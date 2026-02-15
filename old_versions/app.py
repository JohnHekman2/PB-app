import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os 
from datetime import datetime
import pandas as pd 

# --- UTILITY FUNCTION ---
def get_area_name(file_path: str) -> str:
    """
    Extracts the clean protected area name from a file path.
    
    Assumes file names are like 'data/AreaName_assessment.pdf'
    """
    # 1. Get the base filename (e.g., 'Vechtplassen_assessment.pdf')
    filename = os.path.basename(file_path)
    
    # 2. Remove the extension (.pdf)
    name_without_ext = os.path.splitext(filename)[0]
    
    # 3. Remove the suffix '_assessment'
    clean_name = name_without_ext.replace('_assessment', '').replace('_ASSESSMENT', '')
    
    # 4. Replace underscores with spaces for a clean display name
    display_name = clean_name.replace('_', ' ')
    
    return display_name


# --- CONFIGURATION ---
VECTOR_STORE_DIRECTORY = "vector_store" 
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 

# --- LLM API DETAILS (LOADED FROM SECRETS.TOML) ---
try:
    # Load credentials securely from .streamlit/secrets.toml
    # Accessing keys directly from the root level, e.g., st.secrets["BASE_URL"]
    YOUR_API_BASE_URL = st.secrets["BASE_URL"]
    YOUR_API_KEY = st.secrets["API_KEY"]
except KeyError:
    st.error("API keys (API_KEY or BASE_URL) not found in `.streamlit/secrets.toml`. Please configure your file to match the simple 'KEY = \"value\"' structure.")
    st.stop()


# --- PROMPT TEMPLATE ---
SYSTEM_TEMPLATE = """
Je bent een behulpzame, deskundige assistent voor het analyseren van documenten over biodiversiteitsdoelstellingen. 
Je taak is om de geleverde context strikt te analyseren en eerst de aanwezigheid van vier sleutelconcepten vast te stellen.
Vervolgens geef je een beknopte samenvatting, uitsluitend op basis van de gevonden context.
Gebruik geen informatie die niet in de geleverde context staat. Wees beknopt en gezaghebbend.

Context:
{context}
"""

# De vaste prompt voor de analysetaak, aangepast om de 4 concepten te checken
ANALYSIS_PROMPT = """
Analyseer de geleverde context en geef eerst, in een Markdown lijst, de aanwezigheid van de volgende concepten aan met de status **Aanwezig** of **Niet aanwezig**:
1. Habitattype
2. Habitatrichtlijnsoorten
3. Broedvogels
4. Niet-broedvogels

Geef daarna een algemene samenvatting van **maximaal 10 zinnen** over de belangrijkste bevindingen van dit document. Focus op de categorieën die 'Aanwezig' zijn, maar neem ook de bevindingen over knelpunten, beheer, of algemeen oordeel mee in de samenvatting.
"""


# --- HELPER FUNCTIONS (CACHED) ---

@st.cache_resource
def get_embedding_model():
    """
    Loads the embedding model (cached)
    """
    print("Loading embedding model...")
    model_kwargs = {'device': 'cpu'} 
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs
    )

@st.cache_resource
def get_vector_store():
    """
    Loads the persistent vector store from disk (cached)
    """
    print("Loading vector store...")
    _embeddings = get_embedding_model() 
    return Chroma(
        persist_directory=VECTOR_STORE_DIRECTORY,
        embedding_function=_embeddings
    )

@st.cache_resource
def get_custom_llm():
    """
    Initializes a connection to your custom LLM API (cached)
    """
    print("Initializing custom LLM connection...")
    return ChatOpenAI(
        model="gpt-5-mini",
        api_key=YOUR_API_KEY,
        base_url=YOUR_API_BASE_URL,
        temperature=0.2 
    )

@st.cache_resource
def get_all_area_names():
    """
    Retrieves and cleans all unique area names from the vector store metadata.
    """
    vector_store = get_vector_store()
    documents = vector_store.get(include=['metadatas'])
    
    unique_area_names = set()
    for metadata in documents['metadatas']:
        if 'area_name' in metadata:
            unique_area_names.add(metadata['area_name'])
            
    return sorted(list(unique_area_names))

def create_filtered_retriever(vector_store, selected_areas):
    """
    Creates a retriever that filters documents based on the selected area names.
    This function is used primarily to define the filter logic for the batch analysis.
    """
    RETRIEVAL_COUNT = 15

    # If no areas are selected, return a retriever that queries everything (no filter)
    if not selected_areas:
        return vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_COUNT})

    # If a single area is selected, use a simple equality filter
    if len(selected_areas) == 1:
        chroma_filter = {"area_name": selected_areas[0]}
    # If multiple areas are selected, use the '$or' operator
    else:
        filter_conditions = [{"area_name": area} for area in selected_areas]
        chroma_filter = {"$or": filter_conditions}

    return vector_store.as_retriever(
        search_kwargs={
            "k": RETRIEVAL_COUNT, 
            "filter": chroma_filter
        }
    )

def get_rag_chain(_retriever, _llm, system_template):
    """
    Creates the full RAG chain using a fixed system prompt.
    """
    print("Creating RAG chain...")
    
    # 1. Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}") 
    ])
    
    # 2. Define the RAG chain
    rag_chain = (
        {"context": _retriever, "question": RunnablePassthrough()}
        | prompt
        | _llm
        | StrOutputParser()
    )
    
    return rag_chain

# --- NEW BATCH ANALYSIS FUNCTION ---
def run_batch_analysis(vector_store, llm, selected_areas, analysis_prompt, system_template):
    """
    Runs the analysis prompt individually against each selected area document.
    """
    if not selected_areas:
        return "Geen documenten geselecteerd voor analyse."

    # Use a dictionary to store the results
    results = {}
    
    # Initialize Streamlit progress bar
    progress_bar = st.progress(0, text="Analyse wordt uitgevoerd...")
    total_areas = len(selected_areas)

    for i, area in enumerate(selected_areas):
        progress_text = f"Bezig met document {i+1} van {total_areas}: **{area}**"
        progress_bar.progress((i + 1) / total_areas, text=progress_text)
        
        # 1. Create a RETRIEVER filtered ONLY for the current area (k=15)
        # We pass [area] as a list to filter for only this single document.
        retriever = create_filtered_retriever(vector_store, [area])
        
        # 2. Create the RAG chain for this specific retriever
        rag_chain = get_rag_chain(retriever, llm, system_template)
        
        try:
            # 3. Invoke the chain with the fixed analysis prompt
            # The retriever now only fetches chunks from the current 'area'.
            response = rag_chain.invoke(analysis_prompt)
            results[area] = response
        except Exception as e:
            results[area] = f"**Fout bij analyse van {area}** (LLM API Fout): {e}"
            print(f"Error processing {area}: {e}")
            
    progress_bar.empty()
    st.success("Analyse voltooid!")
    return results

# --- MAIN STREAMLIT APP ---

st.set_page_config(page_title="Biodiversity Assessment Batch Analyzer", layout="wide")
st.title("🌱 Batch Analyzer: Natuurdoel Analyse Samenvattingen")
st.markdown("Selecteer de documenten waarvoor u een samenvatting op hoog niveau wilt genereren. De app zal de analyse uitvoeren voor **elk document afzonderlijk** en de resultaten samenvoegen in één downloadbaar Markdown bestand.")


# --- LOAD ALL COMPONENTS ---
try:
    vector_store = get_vector_store()
    llm = get_custom_llm()
    all_areas = get_all_area_names()
except Exception as e:
    st.error(f"Er is een fout opgetreden tijdens de initialisatie: {e}")
    st.error("Controleer of 'build_index.py' correct is uitgevoerd en of de 'vector_store' map aanwezig is.")
    st.stop()


# --- SIDEBAR (Area Selection) ---
st.sidebar.header("Document Selectie")
selected_areas = st.sidebar.multiselect(
    "Selecteer de gebieden voor batch analyse:",
    options=all_areas,
    default=[] # Start met lege selectie
)

# --- MAIN INTERFACE: RUN BUTTON ---
st.info(f"Geselecteerde documenten voor analyse: **{len(selected_areas)}**")

if st.button("▶️ Start Batch Analyse (Samenvatting)", disabled=not selected_areas):
    
    # --- BATCH EXECUTION ---
    with st.container():
        st.info("Start analyse... Dit kan even duren afhankelijk van het aantal geselecteerde documenten en de LLM-latentie.")
        
        analysis_results = run_batch_analysis(
            vector_store=vector_store,
            llm=llm,
            selected_areas=selected_areas,
            analysis_prompt=ANALYSIS_PROMPT,
            system_template=SYSTEM_TEMPLATE
        )
        
        # --- FORMAT OUTPUT ---
        
        # Build the Markdown string for the download file
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        markdown_content = f"# Batch Analyse Rapport\n\n"
        markdown_content += f"**Datum en Tijd:** {now}\n"
        markdown_content += f"**Analyse Vraag:** {ANALYSIS_PROMPT}\n"
        markdown_content += f"**Aantal Geanalyseerde Documenten:** {len(analysis_results)}\n\n"
        markdown_content += "---\n\n"
        
        
        for area, result in analysis_results.items():
            markdown_content += f"## Samenvatting voor: {area}\n\n"
            
            if result.startswith("**Fout bij analyse"):
                # Handle API errors gracefully in the report
                markdown_content += f"{result}\n\n"
            else:
                markdown_content += f"{result}\n\n"
            markdown_content += "---\n\n"
            
        # --- DISPLAY DOWNLOAD BUTTON ---
        st.subheader("Rapport Voltooid")
        
        # Generate filename
        filename = f"batch_analyse_rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        st.download_button(
            label="⬇️ Download Analyse Rapport (.md)",
            data=markdown_content.encode('utf-8'),
            file_name=filename,
            mime="text/markdown"
        )
        
        st.caption("U kunt de inhoud hieronder bekijken:")
        st.code(markdown_content, language='markdown')


st.sidebar.markdown("---")
st.sidebar.markdown(f"Totaal Geïndexeerde Gebieden: **{len(all_areas)}**")