import streamlit as st
import io
from io import StringIO
import os
from datetime import datetime
import pandas as pd
import re

# Third-party imports
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
import markdown
from htmldocx import HtmlToDocx
from thefuzz import process
from thefuzz import fuzz

# Utility imports
from utils import RUIS_WOORDEN, generate_csv_from_municipality, PAD_GEMEENTEN

# --- 1. PAGE CONFIGURATION (MUST BE FIRST) ---
st.set_page_config(page_title="Biodiversity Assessment Batch Analyzer", layout="wide")

# --- 2. CONFIGURATION & SECRETS ---
VECTOR_STORE_DIRECTORY = "vector_store"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

try:
    # Load credentials securely from .streamlit/secrets.toml
    YOUR_API_BASE_URL = st.secrets["BASE_URL"]
    YOUR_API_KEY = st.secrets["API_KEY"]
except KeyError:
    st.error("API keys (API_KEY or BASE_URL) not found in `.streamlit/secrets.toml`.")
    st.stop()

# --- 3. PROMPT TEMPLATES ---
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

CONCEPT_CHECK_PROMPT = """
Analyseer de geleverde context en geef aan of er in de bron instandhoudingsdoelstellingen zijn voor de volgende 4 natuurtypen. Status 'Ja' betekent dat er 1 of meer instandhoudingsdoelstellingen zijn voor dit natuurtype.

- Habitattype: [Status]
- Habitatrichtlijnsoorten: [Status]
- Broedvogels: [Status]
- Niet-broedvogels: [Status]

Geef geen andere tekst.
"""

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

# --- 4. CACHED HELPER FUNCTIONS ---

@st.cache_resource
def get_embedding_model():
    print("Loading embedding model...")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})

@st.cache_resource
def get_vector_store():
    print("Loading vector store...")
    return Chroma(persist_directory=VECTOR_STORE_DIRECTORY, embedding_function=get_embedding_model())

@st.cache_resource
def get_custom_llm():
    print("Initializing custom LLM connection...")
    return ChatOpenAI(
        model="gpt-5-mini",
        api_key=YOUR_API_KEY,
        base_url=YOUR_API_BASE_URL,
        temperature=0.2
    )

@st.cache_data
def get_all_area_names():
    # Helper to prevent re-reading DB on every rerun
    vector_store = get_vector_store()
    documents = vector_store.get(include=['metadatas'])
    unique_area_names = set()
    for metadata in documents['metadatas']:
        if 'area_name' in metadata:
            unique_area_names.add(metadata['area_name'])
    return sorted(list(unique_area_names))

@st.cache_data
def load_gemeenten():
    """Caches the heavy GML file loading to prevent app slowness."""
    try:
        import geopandas as gpd
        if not os.path.exists(PAD_GEMEENTEN):
            return []
        gemeenten_gdf = gpd.read_file(PAD_GEMEENTEN)
        return sorted(gemeenten_gdf['Gemeentenaam'].unique().tolist())
    except Exception as e:
        print(f"Error loading gemeenten: {e}")
        return []

def create_filtered_retriever(vector_store, selected_areas):
    RETRIEVAL_COUNT = 15
    if not selected_areas:
        return vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_COUNT})
    
    if len(selected_areas) == 1:
        chroma_filter = {"area_name": selected_areas[0]}
    else:
        chroma_filter = {"$or": [{"area_name": area} for area in selected_areas]}

    return vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_COUNT, "filter": chroma_filter})

def get_rag_chain(_retriever, _llm, system_template):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}") 
    ])
    return RunnableParallel({
        "context": _retriever,
        "question": RunnablePassthrough()
    }) | {
        "answer": prompt | _llm | StrOutputParser(),
        "context": lambda x: x["context"]
    }

def invoke_rag_chain(rag_chain, prompt):
    response_dict = rag_chain.invoke(prompt)
    unique_sources = set()
    for doc in response_dict['context']:
        unique_sources.add(doc.metadata.get('source', 'Onbekende bron'))
    return response_dict['answer'], sorted(list(unique_sources))

def convert_markdown_to_docx_bytes(markdown_string: str) -> io.BytesIO:
    html = markdown.markdown(markdown_string, extensions=['tables'])
    buffer = io.BytesIO()
    parser = HtmlToDocx()
    doc = parser.parse_html_string(html)
    doc.save(buffer)
    buffer.seek(0)
    return buffer

def clean_area_name_for_matching(name: str) -> str:
    clean_name = name.lower()
    clean_name = re.sub(r'[^a-z0-9\s]+', ' ', clean_name)
    words = clean_name.split()
    words = [w for w in words if w not in RUIS_WOORDEN]
    clean_name = ' '.join(words)
    return re.sub(r'\s+', ' ', clean_name).strip()

def match_areas_from_csv(uploaded_file, all_available_areas: list, column_name: str = 'naam_n2k', threshold: int = 60):
    try:
        df = pd.read_csv(uploaded_file) 
        if column_name not in df.columns:
            st.error(f"Kolom '{column_name}' niet gevonden in CSV.")
            return [], [], [] 
        csv_names = df[column_name].astype(str).str.strip().unique().tolist()
    except Exception as e:
        st.error(f"Fout bij lezen CSV: {e}")
        return [], [], []

    areas_to_analyze_indexed = set()
    successful_matches_detail = []
    debug_info = []
    
    for csv_name in csv_names:
        cleaned_csv_name = clean_area_name_for_matching(csv_name)
        
        if cleaned_csv_name in all_available_areas:
            best_match = (cleaned_csv_name, 100)
        else:
            best_match = process.extractOne(cleaned_csv_name, all_available_areas, scorer=fuzz.token_sort_ratio)
            
        score = best_match[1] if best_match else 0
        candidate = best_match[0] if best_match else 'Geen'

        if score >= threshold:
            areas_to_analyze_indexed.add(candidate)
            successful_matches_detail.append({'csv_name': csv_name, 'indexed_name': candidate, 'score': score})
        else:
            debug_info.append({'csv_name': csv_name, 'best_candidate': candidate, 'score': score})

    return successful_matches_detail, sorted(list(areas_to_analyze_indexed)), debug_info

# --- 5. CORE ANALYSIS FUNCTION ---
def run_batch_analysis(vector_store, llm, selected_areas, concept_check_prompt, table_generation_prompt, system_template):
    if not selected_areas:
        return "Geen documenten geselecteerd."

    results = {}
    progress_bar = st.progress(0, text="Analyse wordt uitgevoerd...")
    total_areas = len(selected_areas)

    for i, area in enumerate(selected_areas):
        progress_bar.progress((i + 1) / total_areas, text=f"Bezig met document {i+1}/{total_areas}: **{area}**")
        
        retriever = create_filtered_retriever(vector_store, [area])
        rag_chain = get_rag_chain(retriever, llm, system_template)
        
        try:
            concept_check_result, _ = invoke_rag_chain(rag_chain, concept_check_prompt)
            final_prompt = table_generation_prompt.format(concept_check_result=concept_check_result)
            table_and_summary, unique_sources = invoke_rag_chain(rag_chain, final_prompt)
            
            results[area] = {'summary': table_and_summary, 'sources': unique_sources}
        except Exception as e:
            results[area] = {'summary': f"**Fout bij analyse van {area}**: {e}", 'sources': []}
            print(f"Error processing {area}: {e}")
            
    progress_bar.empty()
    return results

# --- 6. MAIN APP INTERFACE ---

st.title("🌱 Batch Analyzer: Natuurdoel Analyse")
st.markdown("Selecteer documenten voor analyse. Resultaat is één downloadbaar Markdown bestand.")

# Load Components
try:
    vector_store = get_vector_store()
    llm = get_custom_llm()
    all_areas = get_all_area_names()
except Exception as e:
    st.error(f"Fout bij initialisatie: {e}")
    st.stop()

# Initialize Session State
for key in ['csv_file_buffer', 'areas_to_analyze', 'locked_areas_to_analyze', 
            'successful_matches_detail', 'debug_info', 'analysis_results']:
    if key not in st.session_state:
        st.session_state[key] = None if 'results' in key else []
if 'matching_complete' not in st.session_state: st.session_state.matching_complete = False
if 'analysis_running' not in st.session_state: st.session_state.analysis_running = False

# Sidebar
st.sidebar.header("1. Selectie Methode")
tab1, tab2, tab3 = st.sidebar.tabs(["A. Gemeente", "B. CSV Upload", "C. Handmatig"])

# --- TAB 1: Gemeente ---
with tab1:
    st.subheader("Automatisch via Gemeente")
    available_gemeenten = load_gemeenten() # Uses cached function
    
    gemeente_input = st.selectbox("Selecteer een Gemeente:", options=available_gemeenten, index=None, placeholder="Typ...", key='gemeente_selectbox')
    
    if st.button("Genereer & Match Documenten"):
        if gemeente_input:
            buffer, message = generate_csv_from_municipality(gemeente_input)
            st.session_state.csv_file_buffer = buffer
            
            if buffer:
                st.success(f"CSV gegenereerd voor '{gemeente_input}'")
                matches, areas, debug = match_areas_from_csv(buffer, all_areas)
                st.session_state.successful_matches_detail = matches
                st.session_state.areas_to_analyze = areas
                st.session_state.debug_info = debug
                st.session_state.locked_areas_to_analyze = list(areas)
                st.session_state.matching_complete = True
                st.rerun()
            else:
                st.warning(message)
        else:
            st.warning("Selecteer een gemeente.")

# --- TAB 2: CSV Uploader --- 
with tab2:
    st.subheader("Upload CSV")
    uploaded_file = st.file_uploader("Upload CSV met 'naam_n2k' kolom", type=['csv'])
    if uploaded_file:
        matches, areas, debug = match_areas_from_csv(uploaded_file, all_areas)
        st.session_state.successful_matches_detail = matches
        st.session_state.areas_to_analyze = areas
        st.session_state.debug_info = debug
        st.session_state.locked_areas_to_analyze = list(areas)
        st.session_state.matching_complete = True
        st.success(f"CSV geladen! {len(areas)} documenten.")
        st.rerun()

# --- TAB 3: Handmatig ---
with tab3:
    st.subheader("Handmatig")
    manual_selection = st.multiselect("Selecteer gebieden:", options=all_areas, default=st.session_state.areas_to_analyze, key='manual_selection')
    st.session_state.areas_to_analyze = manual_selection

# Display Results in Sidebar
areas = st.session_state.areas_to_analyze
matches = st.session_state.successful_matches_detail
debug = st.session_state.debug_info

if matches or debug:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Resultaten:")
    st.sidebar.success(f"✅ {len(matches)} matches gevonden.")
    with st.sidebar.expander(f"Details Matches ({len(matches)})"):
        for m in matches:
            st.write(f"**{m['csv_name']}** -> {m['indexed_name']} ({m['score']}%)")
    if debug:
        st.sidebar.warning(f"❌ {len(debug)} niet gevonden.")

if areas:
    st.sidebar.markdown("---")
    st.sidebar.code(", ".join(areas), language='text')

# --- MAIN EXECUTION ---
st.info(f"Geselecteerde documenten: **{len(areas)}**")

if st.button("▶️ Start Batch Analyse", disabled=not areas):
    # Use locked selection if available to prevent UI reset issues
    target_areas = list(st.session_state.get('areas_to_analyze') or st.session_state.get('locked_areas_to_analyze') or [])
    
    if not target_areas:
        st.error("Geen gebieden geselecteerd.")
    else:
        st.session_state.analysis_running = True
        try:
            # RUN ANALYSIS
            results = run_batch_analysis(
                vector_store=vector_store,
                llm=llm,
                selected_areas=target_areas,
                concept_check_prompt=CONCEPT_CHECK_PROMPT,
                table_generation_prompt=TABLE_GENERATION_PROMPT,
                system_template=SYSTEM_TEMPLATE
            )
            st.session_state.analysis_results = results

            # GENERATE REPORT
            if results:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                md_out = f"# Batch Analyse Rapport\n**Datum:** {now}\n\n---\n\n"
                
                for area, data in results.items():
                    md_out += f"## {area}\n\n{data['summary']}\n\n"
                    if data['sources']:
                        md_out += "**Bronnen:**\n" + "\n".join([f"- {s}" for s in data['sources']]) + "\n"
                    md_out += "\n---\n\n"
                
                st.session_state.markdown_content = md_out
                st.success("Klaar!")
                
        except Exception as e:
            st.error(f"Fout: {e}")
        finally:
            st.session_state.analysis_running = False

# Downloads
if "markdown_content" in st.session_state and st.session_state.markdown_content:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Markdown", st.session_state.markdown_content, f"analyse_{timestamp}.md")
    c2.download_button("⬇️ Word", convert_markdown_to_docx_bytes(st.session_state.markdown_content), f"analyse_{timestamp}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    
    with st.expander("Preview"):
        st.markdown(st.session_state.markdown_content)