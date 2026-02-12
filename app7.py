import streamlit as st
import io
from io import StringIO
import os
import tempfile 
from datetime import datetime
import pandas as pd
import re
import json
from collections import Counter

# Third-party imports
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
import markdown
from htmldocx import HtmlToDocx
from thefuzz import process
from thefuzz import fuzz
from openai import OpenAI # NIEUW: Voor directe API call

# Utility imports
from utils import RUIS_WOORDEN, generate_csv_from_municipality, PAD_GEMEENTEN

# --- 1. PAGE CONFIGURATION (MUST BE FIRST) ---
st.set_page_config(page_title="Biodiversity Assessment Batch Analyzer", layout="wide")

# --- 2. CONFIGURATION & SECRETS ---
VECTOR_STORE_DIRECTORY = "vector_store"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

try:
    # Load credentials securely from .streamlit/secrets.toml
    # We slaan ze ook op in session_state voor de directe OpenAI client
    if "openai_base_url" not in st.session_state:
        st.session_state.openai_base_url = st.secrets["BASE_URL"]
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = st.secrets["API_KEY"]
        
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

**STAP 2: GESTRUCTUREERDE ANALYSE (JSON)**
Je genereert een gedetailleerde analyse in JSON-formaat. Je mag uitsluitend de resultaten gebruiken van de concept-check om te bepalen welke typen natuur je moet analyseren.

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
Jouw taak is om een gedetailleerde analyse te genereren over de natuurdoelen.
**Geef je antwoord UITSLUITEND als een valide JSON-object.** Gebruik geen Markdown-opmaak (zoals ```json) rondom het object.

**GEBRUIK DE VOLGENDE INSTRUCTIES OM TE BEPALEN WELKE CATEGORIEN JE MOET ANALYSEREN. ANALYSEER ALLEEN DE CATEGORIEËN DIE IN DE INSTRUCTIES ALS 'AANWEZIG' ZIJN GEMARKEERD.**
-----------------
INSTRUCTIES/CHECKLISTt:
{concept_check_result}
-----------------

Voor elke 'Aanwezige' categorie, en voor elk individueel type binnen die categorie (bijv. elk Habittattype zoals H3150 of elke Broedvogelsoort), voeg je een object toe aan de lijst "bevindingen".

Het JSON-object moet de volgende structuur hebben:
{{
  "bevindingen": [
    {{
      "categorie": "Kies uit: 'Habitattype', 'Habitatrichtlijnsoorten', 'Broedvogels', of 'Niet-broedvogels'",
      "natuurtype": "Bijv: 'H3150 Natuurlijke eutrofe meren' (volledige naam incl code)",
      "kwaliteit": "Korte beschrijving kwaliteit",
      "knelpunten": "Belangrijkste knelpunten",
      "oordeel": "Kies strikt uit: 'Ja', 'Nee, niet haalbaar', of 'Nee, gebrek aan gegevens'"
    }}
  ],
  "samenvatting": "Een algemene samenvatting van maximaal 5 zinnen over de meest opvallende bevindingen."
}}

Als er geen gegevens zijn, laat de lijst "bevindingen" leeg.
"""

# Prompt voor Stap 2 (Omgevingsvisie) - Aangepast voor Full Context met 5 categorieën
IMPACT_PROMPT_FULL = """
Je bent een expert in ruimtelijke ordening en ecologie.
Hieronder volgt de volledige tekst van een Omgevingsvisie (of beleidsdocument).

**JOUW TAAK:**
Analyseer het document grondig op concrete ingrepen, ambities of ontwikkelingen binnen de volgende 5 categorieën die impact kunnen hebben op de natuur:

1. **Woningbouw:** (Implicaties: ruimtegebruik, extra mensen, extra verkeer, extra recreatiedruk)
2. **Recreatie Ontwikkeling:** (Implicaties: extra recreatiedruk, verstoring)
3. **Mobiliteit & Infrastructuur:** (Implicaties: ruimtegebruik, extra verkeer, versnippering)
4. **Landbouwmaatregelen:** (Implicaties: verplaatsing/intensivering/extensivering kan leiden tot verschuivingen in stikstofdepositie)
5. **Bedrijvigheid:** (Implicaties: extra verkeersbewegingen, ruimtegebruik nieuwe terreinen)

**OUTPUT FORMAAT:**
Genereer voor ELK van deze 5 categorieën een samenvattende tekst.
- Gebruik de categorie als tussenkop (bijv. "### 1. Woningbouw").
- Beschrijf concreet wat er in het plan staat (aantallen, locaties, specifieke projecten).
- Benoem expliciet de potentiële risico's voor de natuur zoals hierboven beschreven.
- Citeer waar mogelijk paginanummers of paragraafnamen.

Als er voor een categorie GEEN maatregelen worden genoemd, geef dit dan expliciet aan met "Geen relevante ingrepen gevonden in dit document."

TEKST VAN DOCUMENT:
{context}
"""

# --- 4. CACHED HELPER FUNCTIONS (Models & Data) ---

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
    """LangChain wrapper voor RAG taken"""
    print("Initializing custom LLM connection...")
    return ChatOpenAI(
        model="gpt-5-mini",
        api_key=YOUR_API_KEY,
        base_url=YOUR_API_BASE_URL
    )

# NIEUW: Directe OpenAI Client voor Full Context Analysis
@st.cache_resource
def get_openai_client():
    """
    Maakt en cachet een OpenAI client-instantie.
    Haalt de credentials op uit st.session_state, die worden geinitialiseerd
    via st.secrets.
    """
    api_key = st.session_state.get('openai_api_key')
    base_url = st.session_state.get('openai_base_url')
    
    # Zorg ervoor dat de base_url eindigt op /v1 als dat nodig is voor de API
    if base_url and not base_url.endswith("/v1"):
         # Dit is een heuristiek, de uiteindelijke implementatie hangt af van de AI-proxy
        base_url = base_url.rstrip('/') + '/v1' 
        
    if api_key and base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return None

@st.cache_data
def get_all_area_names():
    vector_store = get_vector_store()
    documents = vector_store.get(include=['metadatas'])
    unique_area_names = set()
    for metadata in documents['metadatas']:
        if 'area_name' in metadata:
            unique_area_names.add(metadata['area_name'])
    return sorted(list(unique_area_names))

@st.cache_data
def load_gemeenten():
    try:
        import geopandas as gpd
        if not os.path.exists(PAD_GEMEENTEN):
            return []
        gemeenten_gdf = gpd.read_file(PAD_GEMEENTEN)
        return sorted(gemeenten_gdf['Gemeentenaam'].unique().tolist())
    except Exception as e:
        print(f"Error loading gemeenten: {e}")
        return []

# --- 5. LOGIC HELPER FUNCTIONS (Matching, Processing, Conversion) ---

def calculate_dynamic_stopwords(all_names: list, frequency_threshold: float = 0.05):
    """Berekent welke woorden te vaak voorkomen in de dataset."""
    word_counter = Counter()
    total_docs = len(all_names)
    if total_docs == 0: return set()

    for name in all_names:
        clean = re.sub(r'[^a-z0-9\s]+', ' ', name.lower())
        words = clean.split()
        word_counter.update(set(words))
    
    dynamic_noise = set()
    cutoff_count = total_docs * frequency_threshold
    for word, count in word_counter.items():
        if count > cutoff_count and len(word) > 1:
            dynamic_noise.add(word)
    return dynamic_noise

def clean_area_name_for_matching(name: str, dynamic_stopwords: set = None) -> str:
    """Maakt namen schoon met statische én dynamische stopwoorden."""
    clean_name = name.lower()
    clean_name = re.sub(r'[^a-z0-9\s]+', ' ', clean_name)
    words = clean_name.split()
    all_stopwords = set(RUIS_WOORDEN)
    if dynamic_stopwords:
        all_stopwords.update(dynamic_stopwords)
    filtered_words = [w for w in words if w not in all_stopwords]
    clean_name = ' '.join(filtered_words)
    return re.sub(r'\s+', ' ', clean_name).strip()

def convert_markdown_to_docx_bytes(markdown_string: str) -> io.BytesIO:
    """Converteert Markdown string naar een Word document buffer."""
    # Zorg dat de tabellen extensie aanstaat
    html = markdown.markdown(markdown_string, extensions=['tables'])
    buffer = io.BytesIO()
    parser = HtmlToDocx()
    doc = parser.parse_html_string(html)
    doc.save(buffer)
    buffer.seek(0)
    return buffer

def parse_json_response(response_text: str):
    try:
        cleaned_text = re.sub(r'```json\s*', '', response_text)
        cleaned_text = re.sub(r'```\s*$', '', cleaned_text)
        cleaned_text = cleaned_text.strip()
        data = json.loads(cleaned_text)
        return data
    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {e}")
        return None

def format_json_to_markdown(json_data):
    if not json_data:
        return "**Fout:** Kon geen gestructureerde data uitlezen uit het model antwoord."
    markdown_output = ""
    findings = json_data.get("bevindingen", [])
    if findings:
        df = pd.DataFrame(findings)
        expected_cols = ["categorie", "natuurtype", "kwaliteit", "knelpunten", "oordeel"]
        for col in expected_cols:
            if col not in df.columns: df[col] = ""
        rename_map = {"categorie": "Categorie", "natuurtype": "Natuurtype/Soort", "kwaliteit": "Kwaliteit", "knelpunten": "Knelpunten", "oordeel": "Eindoordeel"}
        df = df[expected_cols].rename(columns=rename_map)
        markdown_output += df.to_markdown(index=False) + "\n\n"
    else:
        markdown_output += "*Geen specifieke bevindingen gerapporteerd.*\n\n"
    summary = json_data.get("samenvatting", "")
    if summary:
        markdown_output += f"**Samenvatting:**\n{summary}"
    return markdown_output

def match_areas_from_csv(uploaded_file, all_available_areas: list, column_name: str = 'naam_n2k', threshold: int = 60):
    dynamic_stopwords = calculate_dynamic_stopwords(all_available_areas, frequency_threshold=0.05)
    st.session_state.dynamic_stopwords_used = sorted(list(dynamic_stopwords))

    try:
        df = pd.read_csv(uploaded_file) 
        if column_name not in df.columns:
            st.error(f"Kolom '{column_name}' niet gevonden in CSV.")
            return [], [], [] 
        
        distance_map = {}
        if 'afstand_km' in df.columns:
            for _, row in df.iterrows():
                name_key = str(row[column_name]).strip()
                try: distance_map[name_key] = float(row['afstand_km'])
                except: distance_map[name_key] = None

        csv_names = df[column_name].astype(str).str.strip().unique().tolist()
    except Exception as e:
        st.error(f"Fout bij lezen CSV: {e}")
        return [], [], []

    areas_to_analyze_indexed = set()
    successful_matches_detail = []
    debug_info = []
    
    # Pre-clean index
    indexed_map = {}
    for full_name in all_available_areas:
        clean_key = clean_area_name_for_matching(full_name, dynamic_stopwords)
        if clean_key: indexed_map[clean_key] = full_name
    unique_indexed_signatures = list(indexed_map.keys())

    for csv_name in csv_names:
        cleaned_csv_signature = clean_area_name_for_matching(csv_name, dynamic_stopwords)
        
        if cleaned_csv_signature in unique_indexed_signatures:
            best_match_signature = cleaned_csv_signature
            score = 100
        else:
            match_result = process.extractOne(cleaned_csv_signature, unique_indexed_signatures, scorer=fuzz.token_sort_ratio)
            best_match_signature = match_result[0] if match_result else None
            score = match_result[1] if match_result else 0
        
        original_indexed_name = indexed_map.get(best_match_signature, "Onbekend") if best_match_signature else None
        dist = distance_map.get(csv_name)

        if score >= threshold and original_indexed_name:
            areas_to_analyze_indexed.add(original_indexed_name)
            successful_matches_detail.append({'csv_name': csv_name, 'indexed_name': original_indexed_name, 'cleaned_match': f"'{cleaned_csv_signature}' == '{best_match_signature}'", 'score': score, 'distance': dist})
        else:
            debug_info.append({'csv_name': csv_name, 'best_candidate': original_indexed_name, 'cleaned_match': f"'{cleaned_csv_signature}' vs '{best_match_signature}'", 'score': score, 'distance': dist})

    return successful_matches_detail, sorted(list(areas_to_analyze_indexed)), debug_info

# --- NIEUW: Hulpfunctie voor Statistieken Aggregatie ---
def flatten_results_to_df(results_dict):
    """Zet de geneste resultaten-dictionary om naar een vlak DataFrame voor analyse."""
    rows = []
    for area, data in results_dict.items():
        raw = data.get('raw_data')
        if raw and 'bevindingen' in raw:
            for item in raw['bevindingen']:
                rows.append({
                    'Gebied': area,
                    'Categorie': item.get('categorie', 'Onbekend'),
                    'Oordeel': item.get('oordeel', 'Onbekend'),
                    'Soort': item.get('natuurtype', 'Onbekend')
                })
    return pd.DataFrame(rows)

# --- AANGEPAST: Full Context Analyse met Directe OpenAI Client ---
def analyze_local_pdf(uploaded_file, client):
    """
    Laadt PDF, extraheert ALLE tekst en stuurt deze naar OpenAI (Full Context).
    """
    # 1. Tijdelijk opslaan (PyPDFLoader werkt met file paths)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # 2. Extract Text (Full Document)
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        # Voeg alle pagina's samen tot één grote string
        full_text = "\n\n".join([doc.page_content for doc in documents])
        
        # 3. Directe API Call naar OpenAI (gpt-5-mini)
        prompt_content = IMPACT_PROMPT_FULL.format(context=full_text)
        
        response = client.chat.completions.create(
            model="gpt-5-mini", # Gebruik modelnaam uit je config
            messages=[
                {"role": "system", "content": "Je bent een expert in ruimtelijke ordening en ecologie."},
                {"role": "user", "content": prompt_content}
            ],
            temperature=1
        )
        
        return response.choices[0].message.content

    except Exception as e:
        return f"Fout bij analyseren Omgevingsvisie: {e}"
    finally:
        # Opruimen
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# --- 6. RAG CHAIN FUNCTIONS ---

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

def run_batch_analysis(vector_store, llm, selected_areas, concept_check_prompt, table_generation_prompt, system_template):
    if not selected_areas: return "Geen documenten geselecteerd."
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
            json_response_text, unique_sources = invoke_rag_chain(rag_chain, final_prompt)
            json_data = parse_json_response(json_response_text)
            if json_data:
                formatted_markdown = format_json_to_markdown(json_data)
                # NIEUW: Sla ook de ruwe JSON data op in de resultaten
                results[area] = {'summary': formatted_markdown, 'sources': unique_sources, 'raw_data': json_data}
            else:
                results[area] = {'summary': f"**Fout:** Geen valide JSON.\nOutput: {json_response_text}", 'sources': unique_sources, 'raw_data': None}
        except Exception as e:
            results[area] = {'summary': f"Fout: {e}", 'sources': [], 'raw_data': None}
    progress_bar.empty()
    return results

# --- 7. MAIN APP INTERFACE ---

st.title("🌱 Batch Analyzer: Natuurdoel & Omgeving")
st.markdown("Stap 1: Analyseer Natura 2000 doelen. Stap 2: Analyseer impact vanuit Omgevingsvisie.")

try:
    vector_store = get_vector_store()
    llm = get_custom_llm()
    openai_client = get_openai_client() # NIEUW: Laad client
    all_areas = get_all_area_names()
except Exception as e:
    st.error(f"Fout bij initialisatie: {e}")
    st.stop()

# Initialize Session State
for key in ['csv_file_buffer', 'areas_to_analyze', 'locked_areas_to_analyze', 'successful_matches_detail', 'debug_info', 'analysis_results', 'dynamic_stopwords_used', 
            'natuur_analysis_md', 'impact_analysis_md', 'final_report_md']: # NIEUWE keys voor rapporten
    if key not in st.session_state: st.session_state[key] = None if 'results' in key else []
if 'matching_complete' not in st.session_state: st.session_state.matching_complete = False
if 'analysis_running' not in st.session_state: st.session_state.analysis_running = False

st.sidebar.header("1. Selectie Methode")
tab1, tab2, tab3 = st.sidebar.tabs(["A. Gemeente", "B. CSV Upload", "C. Handmatig"])

with tab1:
    st.subheader("Automatisch via Gemeente")
    available_gemeenten = load_gemeenten()
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

with tab3:
    st.subheader("Handmatig")
    manual_selection = st.multiselect("Selecteer gebieden:", options=all_areas, default=st.session_state.areas_to_analyze, key='manual_selection')
    st.session_state.areas_to_analyze = manual_selection

areas = st.session_state.areas_to_analyze
matches = st.session_state.successful_matches_detail
debug = st.session_state.debug_info

if matches or debug:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Resultaten:")
    if matches:
        st.sidebar.success(f"✅ {len(matches)} matches gevonden.")
        with st.sidebar.expander(f"Details Matches ({len(matches)})"):
            for m in matches:
                dist_str = f" ({m['distance']:.1f} km)" if m.get('distance') is not None else ""
                st.write(f"**{m['csv_name']}**{dist_str}\n-> {m['indexed_name']}\n*(Op basis van: {m['cleaned_match']})*")
    
    if debug:
        st.sidebar.warning(f"❌ {len(debug)} niet gevonden.")
        with st.sidebar.expander(f"Details Mislukt ({len(debug)})"):
            for d in debug:
                dist_str = f" ({d['distance']:.1f} km)" if d.get('distance') is not None else ""
                st.write(f"**{d['csv_name']}**{dist_str}\n(Beste gok: *{d['best_candidate']}* - {d['score']}%)")

    if st.session_state.dynamic_stopwords_used:
        with st.sidebar.expander("ℹ️ Automatisch Genegeerde Ruiswoorden"):
            st.write(", ".join(st.session_state.dynamic_stopwords_used))

if areas:
    st.sidebar.markdown("---")
    st.sidebar.code(", ".join(areas), language='text')

st.info(f"Geselecteerde documenten: **{len(areas)}**")

# --- EXECUTION FLOW ---

# STAP 1: NATUURDOEL ANALYSE
st.header("Stap 1: Natura 2000 Doelen Analyse")

if st.button("▶️ Start Stap 1 (Natuurdoelen)", disabled=not areas):
    target_areas = list(st.session_state.get('areas_to_analyze') or st.session_state.get('locked_areas_to_analyze') or [])
    if not target_areas:
        st.error("Geen gebieden geselecteerd.")
    else:
        st.session_state.analysis_running = True
        try:
            results = run_batch_analysis(vector_store, llm, target_areas, CONCEPT_CHECK_PROMPT, TABLE_GENERATION_PROMPT, SYSTEM_TEMPLATE)
            st.session_state.analysis_results = results
            
            # Genereren van initieel rapport
            if results:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                md_out = f"# Natuurdoel & Omgevings Impact Rapport\n**Datum:** {now}\n\n"
                md_out += "# DEEL 1: Natura 2000 Analyse\n\n"
                for area, data in results.items():
                    md_out += f"## {area}\n\n{data['summary']}\n\n"
                    if data['sources']: md_out += "**Bronnen:**\n" + "\n".join([f"- {s}" for s in data['sources']]) + "\n"
                    md_out += "\n---\n\n"
                
                st.session_state.natuur_analysis_md = md_out
                st.session_state.final_report_md = md_out # Voorlopig eindresultaat
                st.success("Stap 1 Voltooid!")
                st.rerun()
        except Exception as e:
            st.error(f"Fout: {e}")
        finally:
            st.session_state.analysis_running = False

# TOON RESULTATEN STAP 1
if st.session_state.analysis_results:
    with st.expander("Bekijk resultaten Stap 1", expanded=False):
        st.markdown(st.session_state.natuur_analysis_md)

    # --- STAP 2: OMGEVINGSVISIE UPLOAD (AANGEPAST) ---
    st.markdown("---")
    st.header("Stap 2: Omgevingsvisie Impact Analyse (Optioneel)")
    st.markdown("Upload de Omgevingsvisie (PDF) om te controleren op ingrepen die de geselecteerde natuurgebieden kunnen raken. **(Full Context Analyse)**")

    omgevings_pdf = st.file_uploader("Upload Omgevingsvisie PDF", type="pdf", key="omgevingsvisie_uploader")

    if omgevings_pdf and st.button("▶️ Start Stap 2 (Impact Analyse)"):
        if not openai_client:
            st.error("Kon OpenAI client niet laden. Controleer je secrets.")
        else:
            with st.spinner("Bezig met analyseren van volledige Omgevingsvisie..."):
                # AANROEP NIEUWE FUNCTIE (MET DIRECTE CLIENT)
                impact_result = analyze_local_pdf(omgevings_pdf, openai_client)
                
                # Markdown opbouwen voor deel 2
                impact_md = "\n# DEEL 2: Impact Analyse uit Omgevingsvisie\n\n"
                impact_md += f"**Geanalyseerd bestand:** {omgevings_pdf.name}\n\n"
                impact_md += impact_result
                
                st.session_state.impact_analysis_md = impact_md
                
                # Samenvoegen rapporten
                st.session_state.final_report_md = st.session_state.natuur_analysis_md + "\n\n---\n\n" + impact_md
                st.success("Stap 2 Voltooid! Rapport bijgewerkt.")
                st.rerun()

    # TOON RESULTATEN STAP 2
    if st.session_state.impact_analysis_md:
        with st.expander("Bekijk resultaten Stap 2", expanded=True):
            st.markdown(st.session_state.impact_analysis_md)


# --- DOWNLOADS & STATS ---

if st.session_state.final_report_md:
    st.markdown("---")
    st.header("📥 Download Eindrapport")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Download Volledig Rapport (.md)", st.session_state.final_report_md, f"rapport_compleet_{timestamp}.md")
    c2.download_button("⬇️ Download Volledig Rapport (.docx)", convert_markdown_to_docx_bytes(st.session_state.final_report_md), f"rapport_compleet_{timestamp}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

# KWANTITATIEVE STATISTIEKEN (Alleen voor Stap 1 data)
if st.session_state.analysis_results:
    df_stats = flatten_results_to_df(st.session_state.analysis_results)
    if not df_stats.empty:
        st.markdown("---")
        st.header("📊 Kwantitatieve Analyse (Natura 2000)")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Aantal beoordelingen per Categorie")
            st.bar_chart(df_stats['Categorie'].value_counts())
        with col2:
            st.subheader("Verdeling van Oordelen")
            st.bar_chart(pd.crosstab(df_stats['Categorie'], df_stats['Oordeel']))