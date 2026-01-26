import streamlit as st
import io
from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
import os 
from datetime import datetime
import pandas as pd 
import markdown
from htmldocx import HtmlToDocx
# NIEUW: Fuzzy matching
from thefuzz import process
from thefuzz import fuzz
# NIEUW: Regulariere expressies voor betere naamschoonmaak
import re

# --- UTILITY FUNCTION ---
def get_area_name(file_path: str) -> str:
    """
    Extracts the clean protected area name from a file path by aggressively 
    removing non-alphanumeric characters, except spaces. This ensures the core
    area name is isolated for robust fuzzy matching.
    
    This function is CRUCIAL for matching CSV input names (e.g., 'Alde Faenen') 
    to the document names (e.g., 'concept_natuurdoelanalyse_alde_feanen').
    """
    # 1. Get base filename without extension
    filename_base = os.path.splitext(os.path.basename(file_path))[0]
    
    # 2. Convert to lowercase
    clean_name = filename_base.lower()

    # 3. Replace all non-alphanumeric characters (including underscores, hyphens, etc.) with a single space
    # This removes all provincial codes, dates, numbers, and most unpredictable delimiters.
    clean_name = re.sub(r'[^a-z0-9\s]+', ' ', clean_name)
    
    # 4. Replace multiple spaces with a single space and strip whitespace
    display_name = re.sub(r'\s+', ' ', clean_name).strip()
    
    return display_name


# --- CONFIGURATION ---
VECTOR_STORE_DIRECTORY = "vector_store" 
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 

# --- LLM API DETAILS (LOADED FROM SECRETS.TOML) ---
try:
    # Load credentials securely from .streamlit/secrets.toml
    YOUR_API_BASE_URL = st.secrets["BASE_URL"]
    YOUR_API_KEY = st.secrets["API_KEY"]
except KeyError:
    st.error("API keys (API_KEY or BASE_URL) not found in `.streamlit/secrets.toml`. Please configure your file to match the simple 'KEY = \"value\"' structure.")
    st.stop()


# --- PROMPT TEMPLATE ---
SYSTEM_TEMPLATE = """
Je bent een behulpzame, deskundige assistent voor het analyseren van documenten over natuurdoelstellingen. Specifiek gaat dit over instandhoudingsdoelstellingen. Het concept 'instandhoudingsdoelstellingen' staat centraal.
Je taak is om de geleverde context strikt te analyseren.

**STAP 1: CONCEPT CHECK**
Je stelt de aanwezigheid van instandhoudingsdoelstellingen vast op 4 verschillende typen natuur.

**STAP 2: TABEL GENERATIE**
Je genereert een gedetailleerde Markdown-tabel. Je mag uitsluitend de resultaten gebruiken van de concept-check om te bepalen welke typen natuur je moet analyseren.

Gebruik geen informatie die niet in de geleverde context staat. Wees beknopt en gezaghebbend.

Context:
{context}
"""

# De vaste prompt voor Stap 1: Concept Check
CONCEPT_CHECK_PROMPT = """
Analyseer de geleverde context en geef aan of er in de bron instandhoudingsdoelstellingen zijn voor de volgende 4 natuurtypen. Status 'Ja' betekent dat er 1 of meer instandhoudingsdoelstellingen zijn voor dit natuurtype.

- Habitattype: [Status]
- Habitatrichtlijnsoorten: [Status]
- Broedvogels: [Status]
- Niet-broedvogels: [Status]

Geef geen andere tekst.
"""

# De vaste prompt voor Stap 2: Tabel Generatie (gebruikt resultaat van Stap 1)
TABLE_GENERATION_PROMPT = """
Jouw taak is om een gedetailleerde Markdown-tabel te genereren over de natuurdoelanalyses.

**GEBRUIK DE VOLGENDE INSTRUCTIES OM TE BEPALEN WELKE CATEGORIEN JE MOET ANALYSEREN. ANALYSEER ALLEEN DE CATEGORIEËN DIE IN DE INSTRUCTIES ALS 'AANWEZIG' ZIJN GEMARKEERD.**
-----------------
INSTRUCTIES/CHECKLIST:
{concept_check_result}
-----------------

Voor elke 'Aanwezige' categorie, en voor elk individueel type binnen die categorie (bijv. elk Habittattype zoals H3150 of elke Broedvogelsoort), genereer je een aparte rij.

De tabel moet de volgende **5 kolommen** bevatten:
1. **Categorie** (Gebruik strikt één van de volgende vier labels: 'Habitattype', 'Habitatrichtlijnsoorten', 'Broedvogels', of 'Niet-broedvogels')
2. **Natuurtype/Soort** (Bijv: 'H3150 Natuurlijke eutrofe meren')
3. **Kwaliteit** (Een korte beschrijving van de huidige kwaliteit.)
4. **Knelpunten** (De belangrijkste knelpunten die de doelen belemmeren.)
5. **Eindoordeel Haalbaarheid** (Drie strikte opties: 'Ja', 'Nee, niet haalbaar', of 'Nee, gebrek aan gegevens'.)

**Constraints:**
- Gebruik alleen de 5 gevraagde kolommen.
- Voeg alle types/soorten toe die je kunt vinden voor de 'Aanwezige' categorieën.
- **Kolom 2:** Gebruik altijd de volledige indicatie (inclusief H-code en de volledige naam van het type/de soort), zonder cursieve tekst.
- **Kolom 5:** Gebruik uitsluitend 'Ja', 'Nee, niet haalbaar', of 'Nee, gebrek aan gegevens'.

Na de tabel genereer je, in een aparte alinea, een algemene samenvatting van **maximaal 5 zinnen** over de meest opvallende bevindingen uit de tabel (bijv. het meest voorkomende eindoordeel).

GEEF UITSLUITEND DE MARKDOWN TABEL EN DE SAMENVATTING TERUG.
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
    """
    RETRIEVAL_COUNT = 15 # Increased k to 15 for more comprehensive analysis

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
    Creates the full RAG chain using RunnableParallel to return both
    the answer and the source documents (context).
    """
    print("Creating RAG chain with source retrieval...")
    
    # 1. Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}") 
    ])
    
    # 2. Define the RAG chain using RunnableParallel
    rag_chain = RunnableParallel(
        {
            "context": _retriever,
            "question": RunnablePassthrough()
        }
    ) | {
        "answer": prompt | _llm | StrOutputParser(),
        "context": lambda x: x["context"] # Pass the documents through for source reference
    }
    
    return rag_chain

def invoke_rag_chain(rag_chain, prompt):
    """
    Invokes the RAG chain and returns the generated answer and unique sources.
    """
    response_dict = rag_chain.invoke(prompt)
    answer = response_dict['answer']
    
    unique_sources = set()
    for doc in response_dict['context']:
        # The source metadata contains the full path, e.g., 'data/file.pdf'
        unique_sources.add(doc.metadata.get('source', 'Onbekende bron'))

    return answer, sorted(list(unique_sources))

def convert_markdown_to_docx_bytes(markdown_string: str) -> io.BytesIO:
    """
    Converts a markdown string to a .docx file in memory.
    Relies on `markdown` and `htmldocx` packages.
    """
    # Convert Markdown to HTML, enabling the 'tables' extension
    html = markdown.markdown(markdown_string, extensions=['tables'])
    
    buffer = io.BytesIO()
    parser = HtmlToDocx()
    # The htmldocx library writes directly to a file path, so we use a temporary file name
    # and then read it back into our buffer.
    doc = parser.parse_html_string(html)
    doc.save(buffer)
    buffer.seek(0)
    return buffer

def match_areas_from_csv(uploaded_file, all_available_areas: list, column_name: str = 'naam_n2k', threshold: int = 40):
    """
    Reads a list of area names from a CSV and performs fuzzy matching 
    against the list of available area names in the vector store.
    
    Uses fuzz.token_sort_ratio for maximum robustness against extra words.
    
    RETURNS:
    - successful_matches_detail: List of dictionaries for names that succeeded the threshold.
    - areas_to_analyze_indexed: List of unique indexed area names to use in the RAG filter.
    - debug_info: List of dictionaries for names that failed the threshold.
    """
    try:
        df = pd.read_csv(uploaded_file)
        if column_name not in df.columns:
            st.error(f"Kolom '{column_name}' niet gevonden in het geüploade CSV-bestand. Zorg ervoor dat de kolomnaam klopt.")
            # Return empty lists for all three components on failure
            return [], [], [] 

        csv_names = df[column_name].astype(str).str.strip().unique().tolist()
    except Exception as e:
        st.error(f"Fout bij het lezen van CSV-bestand: {e}")
        return [], [], []

    areas_to_analyze_indexed = set() # Set of unique indexed names that passed the threshold
    successful_matches_detail = [] # Detailed list of successful matches
    debug_info = [] # List for diagnostic info on failures
    
    # Use the process.extractOne to find the best match for each CSV name
    for csv_name in csv_names:
        # Check if the name exists exactly
        if csv_name in all_available_areas:
            best_match_indexed_name = csv_name
            best_match_score = 100
        else:
            # Fuzzy matching with token_sort_ratio for robustness
            best_match = process.extractOne(
                csv_name, 
                all_available_areas, 
                scorer=fuzz.token_sort_ratio
            )
            
            # Extract score and candidate name
            if best_match:
                best_match_indexed_name = best_match[0]
                best_match_score = best_match[1]
            else:
                best_match_indexed_name = 'Geen Kandidaat Gevonden'
                best_match_score = 0
        
        # Check if a match was found and if the score exceeds the threshold
        if best_match_score >= threshold:
            areas_to_analyze_indexed.add(best_match_indexed_name)
            successful_matches_detail.append({ # Store detailed info for success
                'csv_name': csv_name,
                'indexed_name': best_match_indexed_name,
                'score': best_match_score
            })
        else:
            # Store debug information for names that failed the threshold
            debug_info.append({
                'csv_name': csv_name,
                'best_candidate': best_match_indexed_name,
                'score': best_match_score
            })

    # Return the detailed successful matches, the list of indexed names for the analysis, and the failures
    return successful_matches_detail, sorted(list(areas_to_analyze_indexed)), debug_info


# --- BATCH ANALYSIS FUNCTION ---
def run_batch_analysis(vector_store, llm, selected_areas, concept_check_prompt, table_generation_prompt, system_template):
    """
    Runs the analysis prompt individually against each selected area document in two chained steps.
    """
    if not selected_areas:
        return "Geen documenten geselecteerd voor analyse."

    # Use a dictionary to store the results: {'area': {'summary': str, 'sources': list}}
    results = {}
    
    # Initialize Streamlit progress bar
    progress_bar = st.progress(0, text="Analyse wordt uitgevoerd...")
    total_areas = len(selected_areas)

    for i, area in enumerate(selected_areas):
        progress_text = f"Bezig met document {i+1} van {total_areas}: **{area}**"
        progress_bar.progress((i + 1) / total_areas, text=progress_text)
        
        # 1. Create a RETRIEVER filtered ONLY for the current area
        retriever = create_filtered_retriever(vector_store, [area])
        
        # 2. Create the RAG chain (returns {'answer': str, 'context': list[Document]})
        rag_chain = get_rag_chain(retriever, llm, system_template)
        
        try:
            # --- STEP 1: CONCEPT CHECK ---
            st.markdown(f"**Analyse voor {area}:**")
            st.text(f"  -> Stap 1: Controleren op concepten...")
            
            # The context documents found here are implicitly the sources for Step 2 as well
            concept_check_result, _ = invoke_rag_chain(rag_chain, concept_check_prompt)
            
            # --- STEP 2: TABLE GENERATION (PROMPT CHAINING) ---
            st.text(f"  -> Stap 2: Genereren van de gestructureerde tabel...")
            
            # Construct the final prompt using the result of Step 1
            final_prompt = table_generation_prompt.format(concept_check_result=concept_check_result)
            
            # Invoke the chain for the final output and get sources
            table_and_summary, unique_sources = invoke_rag_chain(rag_chain, final_prompt)
            
            # Store results
            results[area] = {
                'summary': table_and_summary, # Contains the markdown table + 5-sentence summary
                'sources': unique_sources
            }
            
        except Exception as e:
            results[area] = {
                'summary': f"**Fout bij analyse van {area}** (LLM API Fout): {e}",
                'sources': []
            }
            st.error(f"Fout bij verwerken {area}: {e}")
            print(f"Error processing {area}: {e}")
            
    progress_bar.empty()
    st.success("Analyse voltooid! Scroll naar beneden voor het rapport.")
    return results

# --- MAIN STREAMLIT APP ---

st.set_page_config(page_title="Biodiversity Assessment Batch Analyzer", layout="wide")
st.title("🌱 Batch Analyzer: Natuurdoel Analyse")
st.markdown("Selecteer de documenten voor een geavanceerde analyse in twee stappen: eerst een concept-check, dan een gestructureerde tabel. Het resultaat is één downloadbaar Markdown bestand.")


# --- LOAD ALL COMPONENTS ---
try:
    vector_store = get_vector_store()
    llm = get_custom_llm()
    all_areas = get_all_area_names()
except Exception as e:
    st.error(f"Er is een fout opgetreden tijdens de initialisatie: {e}")
    st.error("Controleer of 'build_index.py' correct is uitgevoerd en of de 'vector_store' map aanwezig is.")
    st.stop()


# --- SIDEBAR (Area Selection/CSV Upload) ---
st.sidebar.header("1. Document Selectie")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV met 'naam_n2k' kolom (Optioneel)",
    type=['csv']
)

# Initialize the list of areas to be analyzed
areas_to_analyze = []

if uploaded_file is not None:
    # --- Process CSV and perform Fuzzy Matching ---
    # Update: Now receives detailed success list, the final indexed list, and debug info
    successful_matches_detail, csv_matched_areas, debug_info = match_areas_from_csv(uploaded_file, all_areas)
    
    if csv_matched_areas:
        st.sidebar.success(f"CSV succesvol verwerkt! {len(successful_matches_detail)} van {len(successful_matches_detail) + len(debug_info)} namen gematcht met >85% zekerheid.")
        
        # Set the matched list as the final list for analysis
        areas_to_analyze = csv_matched_areas
        
        # --- NEW DISPLAY FOR SUCCESSFUL MATCHES ---
        st.sidebar.subheader("Matching Resultaten:")
        
        # Gebruik expander en compacte Markdown lijst voor leesbaarheid en ruimte-efficiëntie
        with st.sidebar.expander(f"Toon succesvol gekoppelde namen ({len(successful_matches_detail)} matches)"):
            st.markdown("De volgende **CSV-namen** zijn met succes gekoppeld aan een **Geïndexeerd Document** (score $\ge 85$).")
            
            match_output = ""
            for item in successful_matches_detail:
                # Compacte bullet point notatie die tekstterugloop ondersteunt
                match_output += (
                    f"- **CSV:** `{item['csv_name']}`\n"
                    f"  - **Index:** `{item['indexed_name']}` (Score: {item['score']}%) \n"
                )
            
            st.markdown(match_output) 
            
        # Display the final list of indexed names to be used for RAG filtering
        st.sidebar.markdown(f"**Geïndexeerde Namen voor Analyse ({len(csv_matched_areas)}):**")
        st.sidebar.code(", ".join(csv_matched_areas), language='text')

        # Toon de diagnostische informatie voor mislukte matches
        if debug_info:
            st.sidebar.warning(f"{len(debug_info)} namen konden niet gematcht worden met de drempel (85).")
            with st.sidebar.expander("Toon Diagnostische Matchpogingen (laagste scores)"):
                st.markdown("Dit zijn de namen uit de CSV die de drempel niet haalden, samen met hun beste kandidaat en behaalde score.")
                
                # Sorteer op score om de slechtste matches bovenaan te tonen
                sorted_debug_info = sorted(debug_info, key=lambda x: x['score'])
                
                for item in sorted_debug_info:
                    st.markdown(f"- **CSV:** `{item['csv_name']}`")
                    st.markdown(f"  - **Kandidaat (Indexed):** `{item['best_candidate']}`")
                    st.markdown(f"  - **Score:** `{item['score']}%`")
                    st.markdown("---")


    else:
        st.sidebar.warning("CSV geüpload, maar geen geldige matches gevonden met de bestanden.")
else:
    # --- Manual Selection (Fallback/Alternative) ---
    st.sidebar.markdown("Of **Selecteer Handmatig** de documenten:")
    manual_selection = st.sidebar.multiselect(
        "Geselecteerde gebieden voor batch analyse:",
        options=all_areas,
        default=[] # Start met lege selectie
    )
    areas_to_analyze = manual_selection


# --- MAIN INTERFACE: RUN BUTTON ---
st.info(f"Geselecteerde documenten voor analyse: **{len(areas_to_analyze)}**")

if st.button("▶️ Start Batch Analyse (Gestructureerde Tabel)", disabled=not areas_to_analyze):
    
    st.session_state.analysis_results = None  # Reset previous results
    # --- BATCH EXECUTION ---
    with st.container():
        st.info("Start geavanceerde analyse... Dit kan even duren afhankelijk van het aantal geselecteerde documenten.")
        
        analysis_results = run_batch_analysis(
            vector_store=vector_store,
            llm=llm,
            selected_areas=areas_to_analyze, # Use the list from CSV or manual selection
            concept_check_prompt=CONCEPT_CHECK_PROMPT,
            table_generation_prompt=TABLE_GENERATION_PROMPT,
            system_template=SYSTEM_TEMPLATE
        )
        
        # --- FORMAT OUTPUT AND STORE IN SESSION STATE---
        
        # Build the Markdown string for the download file
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        markdown_content = f"# Batch Analyse Rapport\n\n"
        markdown_content += f"**Datum en Tijd:** {now}\n"
        markdown_content += f"**Analyse Vraag:** De analyse werd uitgevoerd in twee stappen (Concept Check + Tabel Generatie).\n"
        markdown_content += f"**Aantal Geanalyseerde Documenten:** {len(analysis_results)}\n\n"
        markdown_content += "---\n\n"
        
        
        for area, result_dict in analysis_results.items():
            markdown_content += f"## Gedetailleerd Rapport voor: {area}\n\n"
            
            # The summary now contains the table + 5-sentence summary
            markdown_content += f"{result_dict['summary']}\n\n"
            
            # Add Source References
            if result_dict['sources']:
                markdown_content += "\n**Gebruikte Bronnen (Documenten):**\n"
                for source in result_dict['sources']:
                    markdown_content += f"- `{source}`\n"
            elif 'Fout bij analyse' not in result_dict['summary']:
                markdown_content += "*(Geen bronnen gevonden voor deze samenvatting, of de retriever gaf een leeg resultaat.)*\n"

            markdown_content += "\n---\n\n"
            
        st.session_state.markdown_content = markdown_content
        st.success("Analyse voltooid! Scroll naar beneden voor het rapport.")



# --- DISPLAY DOWNLOAD BUTTON & RESULTS (CONDITIONAL) ---
if "markdown_content" in st.session_state and st.session_state.markdown_content:
    st.subheader("Rapport Voltooid")
    
    # Generate filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    md_filename = f"batch_analyse_rapport_gestructureerd_{timestamp}.md"
    docx_filename_dl = f"batch_analyse_rapport_gestructureerd_{timestamp}.docx"

    # --- Markdown Download Button ---
    st.download_button(
        label="⬇️ Download Analyse Rapport (.md)",
        data=st.session_state.markdown_content.encode('utf-8'),
        file_name=md_filename,
        mime="text/markdown"
    )
    
    # --- DOCX Download Button ---
    docx_bytes = convert_markdown_to_docx_bytes(st.session_state.markdown_content)
    st.download_button(
        label="⬇️ Download Analyse Rapport (.docx)",
        data=docx_bytes,
        file_name=docx_filename_dl,
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

    # --- Display Content ---
    st.caption("U kunt de inhoud hieronder bekijken:")
    st.code(st.session_state.markdown_content, language='markdown')


st.sidebar.markdown("---")
st.sidebar.markdown(f"Totaal Geïndexeerde Gebieden: **{len(all_areas)}**")