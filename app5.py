import streamlit as st
import io
from io import StringIO
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
from utils import RUIS_WOORDEN

# NIEUW: Importeer utility functies uit utils.py
from utils import generate_csv_from_municipality, PAD_GEMEENTEN


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

def clean_area_name_for_matching(name: str) -> str:
    """
    Cleans an area name by removing noise words (same logic as get_area_name in utils.py).
    This ensures CSV input names are cleaned the same way as indexed document names.
    """
    # Convert to lowercase
    clean_name = name.lower()
    
    # Remove all non-alphanumeric characters except spaces
    clean_name = re.sub(r'[^a-z0-9\s]+', ' ', clean_name)
    
    # Remove noise words
    words = clean_name.split()
    words = [w for w in words if w not in RUIS_WOORDEN]
    
    # Rejoin and clean up spaces
    clean_name = ' '.join(words)
    clean_name = re.sub(r'\s+', ' ', clean_name).strip()
    
    return clean_name


def match_areas_from_csv(uploaded_file, all_available_areas: list, column_name: str = 'naam_n2k', threshold: int = 60):
    """
    Reads a list of area names from a CSV (or a file-like object/buffer) and 
    performs fuzzy matching against the list of available area names in the vector store.
    
    Uses fuzz.token_sort_ratio for maximum robustness against extra words.
    
    RETURNS:
    - successful_matches_detail: List of dictionaries for names that succeeded the threshold.
    - areas_to_analyze_indexed: List of unique indexed area names to use in the RAG filter.
    - debug_info: List of dictionaries for names that failed the threshold.
    """
    try:
        # Read the CSV from the uploaded file object or in-memory buffer
        df = pd.read_csv(uploaded_file) 
        if column_name not in df.columns:
            st.error(f"Kolom '{column_name}' niet gevonden in het geüploade CSV-bestand. Zorg ervoor dat de kolomnaam klopt.")
            return [], [], [] 

        csv_names = df[column_name].astype(str).str.strip().unique().tolist()
    except Exception as e:
        st.error(f"Fout bij het lezen van CSV-bestand: {e}")
        return [], [], []

    areas_to_analyze_indexed = set()
    successful_matches_detail = []
    debug_info = []
    
    # Use the process.extractOne to find the best match for each CSV name
    for csv_name in csv_names:
        # NIEUW: Clean the CSV name the same way as indexed names
        cleaned_csv_name = clean_area_name_for_matching(csv_name)
        
        # Check if the name exists exactly
        if cleaned_csv_name in all_available_areas:
            best_match_indexed_name = cleaned_csv_name
            best_match_score = 100
        else:
            # Fuzzy matching with token_sort_ratio for robustness
            best_match = process.extractOne(
                cleaned_csv_name, 
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
            successful_matches_detail.append({
                'csv_name': csv_name,
                'indexed_name': best_match_indexed_name,
                'score': best_match_score
            })
        else:
            debug_info.append({
                'csv_name': csv_name,
                'best_candidate': best_match_indexed_name,
                'score': best_match_score
            })

    return successful_matches_detail, sorted(list(areas_to_analyze_indexed)), debug_info


# --- BATCH ANALYSIS FUNCTION (ongewijzigd) ---
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
st.sidebar.header("1. Selectie Methode")

# Initialize state for generated/uploaded file
if 'csv_file_buffer' not in st.session_state:
    st.session_state.csv_file_buffer = None
if 'matching_complete' not in st.session_state:
    st.session_state.matching_complete = False
if 'areas_to_analyze' not in st.session_state:
    st.session_state.areas_to_analyze = []
if 'locked_areas_to_analyze' not in st.session_state:
    st.session_state.locked_areas_to_analyze = []
if 'successful_matches_detail' not in st.session_state:
    st.session_state.successful_matches_detail = []
if 'debug_info' not in st.session_state:
    st.session_state.debug_info = []
if 'analysis_results' not in st.session_state:
    st.session_state['analysis_results'] = None
if 'analysis_running' not in st.session_state:
    st.session_state['analysis_running'] = False

# --- TABS for clearer separation ---
tab1, tab2, tab3 = st.sidebar.tabs(["A. Gemeente", "B. CSV Upload", "C. Handmatig"])

# --- TAB 1: Methode 1: Genereren op basis van Gemeente ---
with tab1:
    st.subheader("Automatisch via Gemeente")
    
    # Get list of all available gemeenten from the GML file
    try:
        import geopandas as gpd
        gemeenten_gdf = gpd.read_file(PAD_GEMEENTEN)
        available_gemeenten = sorted(gemeenten_gdf['Gemeentenaam'].unique().tolist())
    except Exception as e:
        st.warning(f"Kon gemeenten niet laden: {e}")
        available_gemeenten = []
    
    # Use selectbox with autocomplete (search capability)
    gemeente_input = st.selectbox(
        "Selecteer een Gemeente:",
        options=available_gemeenten,
        index=None,
        placeholder="Typ of selecteer een gemeente...",
        key='gemeente_selectbox'
    )
    
    if st.button("Genereer & Match Documenten (via Gemeente)"):
        if gemeente_input:
            
            # BELANGRIJK: Aanroep van de functie uit utils.py
            buffer, message = generate_csv_from_municipality(gemeente_input)
            st.session_state.csv_file_buffer = buffer
            
            if st.session_state.csv_file_buffer:
                # Parse the CSV to get the list of areas
                import pandas as pd
                csv_content = st.session_state.csv_file_buffer.getvalue().decode('utf-8')
                df_generated = pd.read_csv(StringIO(csv_content))
                generated_areas = df_generated['naam_n2k'].tolist()
                
                st.success(f"CSV gegenereerd voor '{gemeente_input}'")
                
                # --- DE CRUCIALE KOPPELSTAP (na match_areas_from_csv-aanroep) ---
                (
                    st.session_state.successful_matches_detail,
                    st.session_state.areas_to_analyze,
                    st.session_state.debug_info
                ) = match_areas_from_csv(st.session_state.csv_file_buffer, all_areas)

                # Lock the selection so later reruns/widgets cannot clear it
                st.session_state.locked_areas_to_analyze = list(st.session_state.areas_to_analyze)
                st.session_state.matching_complete = True

                # Rerun so main UI updates immediately
                st.rerun()

            else:
                st.warning(message)
                st.session_state.matching_complete = False
                st.session_state.areas_to_analyze = []
                st.session_state.successful_matches_detail = []
                st.session_state.debug_info = []
                
        else:
            st.warning("Selecteer een gemeente om te starten.")

# --- TAB 2: Methode 2: Handmatig Uploaden CSV ---
with tab2:
    st.subheader("Upload Bestaande CSV")
    uploaded_file = st.file_uploader(
        "Upload CSV met 'naam_n2k' kolom",
        type=['csv']
    )
    
    if uploaded_file is not None:
        # Voer de matching direct uit met de geüploade Streamlit file object
        (
            st.session_state.successful_matches_detail, 
            st.session_state.areas_to_analyze, 
            st.session_state.debug_info
        ) = match_areas_from_csv(st.session_state.csv_file_buffer, all_areas)
        
        # --- After successful matching (TAB 1 and CSV upload) ---
        # Replace the place where you set st.session_state.areas_to_analyze and matching_complete
        # with the following lines so a locked copy is saved:

        st.session_state.locked_areas_to_analyze = list(st.session_state.areas_to_analyze)  # LOCK the selection
        st.session_state.matching_complete = True
        st.rerun()
        
        st.success(f"✅ CSV geladen! {len(st.session_state.areas_to_analyze)} documenten geselecteerd.")

# --- TAB 3: Methode 3: Handmatig Selecteren ---
with tab3:
    st.subheader("Handmatige Selectie")
    st.markdown("Selecteer gebieden rechtstreeks:")
    manual_selection = st.multiselect(
        "Geselecteerde gebieden voor batch analyse:",
        options=all_areas,
        default=st.session_state.areas_to_analyze,
        key='manual_selection'
    )
    st.session_state.areas_to_analyze = manual_selection
    if manual_selection:
        st.success(f"✅ {len(manual_selection)} documenten geselecteerd.")


# --- Display Match Results (Common section for all methods) ---
areas_to_analyze = st.session_state.areas_to_analyze
successful_matches_detail = st.session_state.successful_matches_detail
debug_info = st.session_state.debug_info

if successful_matches_detail or debug_info:  # Only show if fuzzy matching was used
    
    match_count = len(successful_matches_detail)
    total_input_names = match_count + len(debug_info)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Matching Resultaten:")
    st.sidebar.success(f"✅ {match_count} van {total_input_names} namen gematcht (>85%).")

    # Display successful matches
    with st.sidebar.expander(f"✅ Succesvol gekoppeld ({match_count} matches)", expanded=True):
        if successful_matches_detail:
            match_output = ""
            for item in successful_matches_detail:
                match_output += (
                    f"- **Input:** `{item['csv_name']}`\n"
                    f"  - **Index:** `{item['indexed_name']}` (Score: {item['score']}%) \n"
                )
            st.markdown(match_output)
        else:
            st.info("Geen succesvol gekoppelde namen.")
        
    # Toon de diagnostische informatie voor mislukte matches
    if debug_info:
        st.sidebar.warning(f"❌ {len(debug_info)} namen niet gematcht (<85%).")
        with st.sidebar.expander(f"❌ Niet gekoppeld ({len(debug_info)} pogingen)", expanded=True):
            sorted_debug_info = sorted(debug_info, key=lambda x: x['score'], reverse=True)
            
            for item in sorted_debug_info:
                st.markdown(f"- **Input:** `{item['csv_name']}`")
                st.markdown(f"  - **Beste kandidaat:** `{item['best_candidate']}`")
                st.markdown(f"  - **Score:** `{item['score']}%` (te laag voor match)")


# Display the final list of indexed names to be used for RAG filtering
if areas_to_analyze:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Geïndexeerde Namen ({len(areas_to_analyze)}):**")
    st.sidebar.code(", ".join(areas_to_analyze), language='text')


# --- MAIN INTERFACE: RUN BUTTON ---
# Deze info-box reflecteert nu altijd de staat van de geselecteerde gebieden (of 0 als niets is geselecteerd).
st.info(f"Geselecteerde documenten voor analyse: **{len(areas_to_analyze)}**")

# --- Start Batch Analyse handler: use locked copy as fallback ---
# Replace your current Start button handler with this safe version:

if st.button("▶️ Start Batch Analyse"):
    # Prefer the current live selection, fall back to the locked snapshot
    areas = list(st.session_state.get('areas_to_analyze') or st.session_state.get('locked_areas_to_analyze') or [])
    # Debug helper (remove later)
    st.sidebar.write(f"DEBUG — type areas: {type(areas)}, len: {len(areas)}")
    if not areas:
        st.error("Geen geselecteerde documenten om te analyseren. Controleer selectie (gemeente/CSV/handmatig).")
    else:
        st.session_state['analysis_running'] = True
        try:
            # Pass the locked list into your analysis function to avoid mid-run changes
            analysis_results = run_batch_analysis(areas)  # pas aan naar jouw functie-signature
            st.session_state['analysis_results'] = analysis_results
            if isinstance(analysis_results, dict) and analysis_results:
                st.success("Analyse voltooid.")
            else:
                st.warning("Analyse voltooid, maar geen resultaten terug (leeg of fout).")
                st.sidebar.write("DEBUG — analysis_results type:", type(analysis_results))
                st.sidebar.write("DEBUG — analysis_results inhoud:", analysis_results)
        except Exception as e:
            st.error(f"Fout tijdens analyse: {e}")
            st.exception(e)
        finally:
            st.session_state['analysis_running'] = False