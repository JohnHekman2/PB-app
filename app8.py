import streamlit as st
import io
import os
import tempfile 
from datetime import datetime
import pandas as pd
import re
import json
from collections import Counter
import concurrent.futures
import sys
import tabulate

# Third-party imports
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from openai import OpenAI

# Utility & Service imports
from services.geodata_service import RUIS_WOORDEN, generate_csv_from_municipality, get_geodata_for_municipality, create_map_image, PAD_GEMEENTEN
from services.llm_service import get_embedding_model, get_vector_store, get_custom_llm, get_openai_client
from services.data_processing import get_all_area_names, load_gemeenten, match_areas_from_csv, parse_json_response, format_json_to_markdown, flatten_results_to_df
from services.document_service import load_introduction_from_docx, convert_markdown_to_docx_bytes
from services.llm_analysis_service import analyze_local_pdf, generate_conclusion_paragraph
from services.rag_service import run_batch_analysis
from services.pandoc_converter import convert_md_to_docx_pandoc

# --- 1. PAGE CONFIGURATION (MUST BE FIRST) ---
st.set_page_config(page_title="Passende beoordeling voor omgevingsvisies", layout="wide")

# --- 2. CONFIGURATION & SECRETS ---
VECTOR_STORE_DIRECTORY = "vector_store"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Session State Initialization for AI Provider
if "ai_provider" not in st.session_state:
    st.session_state.ai_provider = "Interne OpenAI"

try:
    # Basic shared secrets (OpenAI Proxy)
    YOUR_API_BASE_URL = st.secrets["BASE_URL"]
    YOUR_API_KEY = st.secrets["API_KEY"]
    
    # Gemini Key (for optional use)
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

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
INSTRUCTIES/CHECKLIST:
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

# Prompt voor Stap 2 (Omgevingsvisie)
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

**OUTPUT FORMAAT & RESTRICTIES:**
Genereer voor ELK van deze 5 categorieën een samenvattende tekst.
- Gebruik de categorie als tussenkop (bijv. "#### 1. Woningbouw").
- Beschrijf concreet wat er in het plan staat (aantallen, locaties, specifieke projecten).
- Benoem expliciet de potentiële risico's voor de natuur zoals hierboven beschreven.
- Citeer waar mogelijk paginanummers of paragraafnamen.
- Maak gebruik van bullet points voor de opsommingen binnen een categorie.

**BELANGRIJK:** Sluit de analyse direct af na de 5e categorie. Geef GEEN suggesties voor verdere analyses, stel GEEN wedervragen en bied GEEN extra hulp aan. De output wordt gebruikt in een statisch rapport zonder mogelijkheid tot interactie.

Als er voor een categorie GEEN maatregelen worden genoemd, geef dit dan expliciet aan met "Geen relevante ingrepen gevonden in dit document."

TEKST VAN DOCUMENT:
{context}
"""

# NIEUW: Prompt voor Stap 3 (Conclusies - alleen op basis van impact)
CONCLUSION_PROMPT_TEMPLATE = """
Je bent een expert in ecologie en ruimtelijke ordening.
Je taak is om een concluderende paragraaf te schrijven over een specifiek onderwerp, gebaseerd op de impactanalyse van een omgevingsvisie.

**ONDERWERP:** {topic}

**ANALYSE IMPACT OMGEVINGSVISIE**
Hieronder staat de analyse van de ingrepen uit de omgevingsvisie.
---
{impact_analyse}
---

**JOUW OPDRACHT:**
Schrijf een beknopte, concluderende paragraaf over het opgegeven **ONDERWERP**. 
Focus op de relatie tussen de vijf impact categorieën en de mogelijke effecten op **ONDERWERP**.

Schrijf 1 paragraaf over hoe de impacts uit de vijf categoriëen kunnen leiden tot een verslechtering in de staat van **ONDERWERP**. 
Bijvoorbeeld, een verslechtering in onderwerp 'stikstofdepositie' betekent een toename van de stikstofdepositie, 
een verslechtering in het onderwerp 'verstoring' betekent een toename van verstoring. Onderbouw wel altijd, met informatie uit de impactanalyse, waarom
je deze conclusie maakt. 

Schrijf ook 1 paragraaf over hoe de impacts uit de vijf categoriëen kunnen leiden tot een verbetering in de staat van **ONDERWERP**.
Bijvoorbeeld, een verbetering in het onderwerp 'stikstofdepositie' betekent een afname van de stikstofdepositie, of
een verbetering in het onderwerp 'verstoring' betekent een afname van de verstoring. Onderbouw wel altijd, met informatie uit de impactanalyse, waarom
je deze conclusie maakt. 

Eindig je analyse met een paragraaf getiteld 'Conclusie en mitigatie'. 
Geef daarin in een paar zinnen de algemene conclusie (overzicht van verslechteringen en verbeteringen), en hoe effecten gemitigeerd kunnen worden.

"""

# --- 4. CACHED HELPER FUNCTIONS (Models & Data) ---
# Models and Vector Store logic have been moved to `services/llm_service.py`
# Area and Gemeenten logic have been moved to `services/data_processing.py`

# --- MOCK DATA FUNCTIES (DEV MODE) ---
def get_mock_natuur_data(selected_areas):
    """Geeft instant dummy data terug voor UI testing."""
    import time
    time.sleep(0.5) # Simuleer een heel klein beetje laadtijd
    
    results = {}
    mock_json = {
        "bevindingen": [
            {"categorie": "Habitattype", "natuurtype": "H3150 Meren", "kwaliteit": "Matig", "knelpunten": "Stikstof", "oordeel": "Ja"},
            {"categorie": "Broedvogels", "natuurtype": "Roerdomp", "kwaliteit": "Slecht", "knelpunten": "Verdroging", "oordeel": "Nee, niet haalbaar"},
            {"categorie": "Habitatrichtlijnsoorten", "natuurtype": "Kamsalamander", "kwaliteit": "Onbekend", "knelpunten": "Versnippering", "oordeel": "Nee, gebrek aan gegevens"}
        ],
        "samenvatting": "Dit is een door de ontwikkelaarsmodus gegenereerde samenvatting. De kwaliteit van de meren staat onder druk door stikstof, en voor de kamsalamander ontbreken momenteel de gegevens."
    }
    
    for area in selected_areas:
        markdown_str = format_json_to_markdown(mock_json)
        results[area] = {
            'summary': markdown_str,
            'sources': ['mock_beheerplan_2024.pdf', 'mock_bijlage_kaarten.pdf'],
            'raw_data': mock_json
        }
    return results

def get_mock_impact_data():
    """Geeft instant dummy data terug voor de Omgevingsvisie UI testing."""
    return """### 1. Woningbouw
- Geplande bouw van 1500 woningen in de wijk 'Nieuw-Noord' (pag. 12).
- **Risico's:** Ruimtegebruik grenzend aan Natura 2000, extra stikstofemissie tijdens de bouwfase, en verhoogde recreatiedruk in omliggende bossen door nieuwe bewoners.

### 2. Recreatie Ontwikkeling
- Aanleg van nieuwe fietspaden en een bezoekerscentrum aan de rand van het plassengebied (pag. 45).
- **Risico's:** Verstoring van broedvogels door toename van dagjesmensen en geluid.

### 3. Mobiliteit & Infrastructuur
- Verbreding van de N-weg om nieuwe wijken te ontsluiten (pag. 50).
- **Risico's:** Extra verkeersbewegingen leiden tot hogere stikstofdepositie. Barrièrewerking (versnippering) voor grondgebonden soorten zoals dassen.

### 4. Landbouwmaatregelen
- Geen relevante ingrepen gevonden in dit document.

### 5. Bedrijvigheid
- Uitbreiding van lokaal bedrijventerrein 'De Haven' met 5 hectare (pag. 60).
- **Risico's:** Extra vrachtverkeer resulteert in stikstofuitstoot, en licht/geluidsverstoring in de nachtelijke uren voor nabijgelegen natuur."""

# --- 5. LOGIC HELPER FUNCTIONS (Matching, Processing, Conversion) ---
# Content from this section has been moved to `services/data_processing.py`, `services/document_service.py`, and `services/llm_analysis_service.py`

# --- 6. RAG CHAIN FUNCTIONS ---
# All RAG functionality has been extracted to `services/rag_service.py`
# --- 7. MAIN APP INTERFACE ---

# --- Service Definition Class ---
# This makes the dependency explicit and clean
class AnalysisServices:
    def __init__(self, run_batch_analysis_func, analyze_local_pdf_func, get_pdf_name_func):
        self.run_batch_analysis = run_batch_analysis_func
        self.analyze_local_pdf = analyze_local_pdf_func
        self.get_pdf_name = get_pdf_name_func

st.title("🌱 Passende Beoordeling voor Omgevingsvisies en programma's")
st.markdown("Stap 1: Analyseer Natura 2000 doelen. Stap 2: Analyseer impact vanuit Omgevingsvisie.")

try:
    # Provider selectie (hier gezet zodat het voor de initialisatie van LLM komt)
    st.sidebar.subheader("🤖 AI Instellingen")
    ai_provider = st.sidebar.radio(
        "Kies AI Provider:",
        options=["Interne OpenAI", "Mijn Gemini"],
        index=0 if st.session_state.ai_provider == "Interne OpenAI" else 1,
        key="ai_provider_radio"
    )
    st.session_state.ai_provider = ai_provider

except Exception as e:
    st.error(f"Fout bij initialisatie: {e}")
    st.stop()

# Initialize Session State
for key in ['csv_file_buffer', 'areas_to_analyze', 'locked_areas_to_analyze', 'successful_matches_detail', 'debug_info', 'analysis_results', 'dynamic_stopwords_used', 
            'natuur_analysis_md', 'impact_analysis_md', 'final_report_md', 'map_image_buffer',
            'conclusion_topics_selected', 'conclusion_results_md']: # NIEUWE KEYS
    if key not in st.session_state: st.session_state[key] = None if 'results' in key or 'buffer' in key or 'md' in key else []
if 'matching_complete' not in st.session_state: st.session_state.matching_complete = False
if 'analysis_running' not in st.session_state: st.session_state.analysis_running = False

st.sidebar.header("1. Selectie Methode")
tab1, tab2 = st.sidebar.tabs(["A. Gemeente", "B. CSV Upload"])

with tab1:
    st.subheader("Automatisch via Gemeente")
    available_gemeenten = load_gemeenten()
    gemeente_input = st.selectbox("Selecteer een Gemeente:", options=available_gemeenten, index=None, placeholder="Typ...", key='gemeente_selectbox')
    if st.button("Genereer & Match Documenten"):
        if gemeente_input:
            # Vorige kaart en selectie wissen
            st.session_state.map_image_buffer = None
            
            message, gemeente_gdf, gebieden_gdf = get_geodata_for_municipality(gemeente_input)

            if gebieden_gdf is not None and not gebieden_gdf.empty:
                st.success(f"Analyse geslaagd: {len(gebieden_gdf)} gebieden gevonden voor '{gemeente_input}'.")

                # 1. Genereer en bewaar de kaart
                with st.spinner("Kaart genereren..."):
                    map_buffer = create_map_image(gemeente_gdf, gebieden_gdf, gemeente_input)
                    st.session_state.map_image_buffer = map_buffer

                # 2. Genereer CSV voor matching-proces
                gebieden_gdf['afstand_km'] = gebieden_gdf['kortste_afstand_m'] / 1000.0
                csv_df = gebieden_gdf[['naam_n2k', 'afstand_km']]
                csv_buffer = io.BytesIO()
                csv_df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                st.session_state.csv_file_buffer = csv_buffer
                
                # 3. Match gebieden
                all_areas = get_all_area_names()
                matches, areas, debug = match_areas_from_csv(io.BytesIO(csv_buffer.getvalue()), all_areas)
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
        # Wis de kaart als er een nieuwe upload is
        st.session_state.map_image_buffer = None
        all_areas = get_all_area_names()
        matches, areas, debug = match_areas_from_csv(uploaded_file, all_areas)
        st.session_state.successful_matches_detail = matches
        st.session_state.areas_to_analyze = areas
        st.session_state.debug_info = debug
        st.session_state.locked_areas_to_analyze = list(areas)
        st.session_state.matching_complete = True
        st.success(f"CSV geladen! {len(areas)} documenten.")
        st.rerun()

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

if st.session_state.get('map_image_buffer'):
    st.subheader("Kaartoverzicht")
    st.session_state.map_image_buffer.seek(0)
    st.image(st.session_state.map_image_buffer, caption="Geselecteerde gemeente en nabijgelegen natuurgebieden.")
    st.markdown("---")

# --- DEV MODE TOGGLE ---
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Development Tools")

# Een enkele schakelaar voor dev mode. De standaardwaarde wordt bepaald door een omgevingsvariabele.
# Dit is een gebruikelijk patroon voor CI/CD of lokale tests.
# Start met: DEV_MODE=true streamlit run app7.py

# Check eerst OS environment variable, daarna secrets.toml
is_dev_env = os.getenv("DEV_MODE", "false").lower() == "true"
if not is_dev_env:
    # Fallback naar secrets.toml als env var niet is gezet
    is_dev_env = str(st.secrets.get("DEV_MODE", "false")).lower() == "true"

dev_mode_active = st.sidebar.checkbox(
    "🛠️ Ontwikkelaarsmodus (gebruik mock data)",
    value=is_dev_env
)

# --- Service Injection ---
# Op basis van de schakelaar "injecteren" we de echte of de mock services.
if dev_mode_active:
    st.sidebar.info("Dev modus is actief.")
    
    # Creëer mock services die voldoen aan de verwachte interface
    def mock_natuur_analyzer(selected_areas, **kwargs):
        st.toast("Gebruik makend van Mock Data voor Stap 1!", icon="🛠️")
        return get_mock_natuur_data(selected_areas)

    def mock_impact_analyzer(uploaded_file, **kwargs):
        return get_mock_impact_data()
        
    def get_mock_filename(uploaded_file):
        return "mock_omgevingsvisie_2024.pdf"

    services = AnalysisServices(
        run_batch_analysis_func=mock_natuur_analyzer,
        analyze_local_pdf_func=mock_impact_analyzer,
        get_pdf_name_func=get_mock_filename
    )
else:
    # Echte services
    services = AnalysisServices(
        run_batch_analysis_func=run_batch_analysis,
        analyze_local_pdf_func=analyze_local_pdf,
        get_pdf_name_func=lambda pdf: pdf.name if pdf else ""
    )

# --- EXECUTION FLOW ---

st.header("Stap 1: Natura 2000 Doelen Analyse")

if st.button("▶️ Start Stap 1 (Natuurdoelen)", disabled=not areas):
    target_areas = list(st.session_state.get('areas_to_analyze') or st.session_state.get('locked_areas_to_analyze') or [])
    if not target_areas:
        st.error("Geen gebieden geselecteerd.")
    else:
        st.session_state.analysis_running = True
        try:
            # Lazy initialize heavy objects just before use
            vector_store = get_vector_store()
            llm = get_custom_llm(st.session_state.ai_provider)
            
            # De code hoeft niet meer te weten of het in dev-modus is.
            results = services.run_batch_analysis(
                vector_store=vector_store, 
                llm=llm, 
                selected_areas=target_areas, 
                concept_check_prompt=CONCEPT_CHECK_PROMPT, 
                table_generation_prompt=TABLE_GENERATION_PROMPT, 
                system_template=SYSTEM_TEMPLATE
            )

            st.session_state.analysis_results = results
            
            if results:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Inleiding laden uit docx
                intro_path = os.path.join("rapport_tekst", "INLEIDING.docx")
                intro_text = load_introduction_from_docx(intro_path)
                
                md_out = f"# Natuurdoel & Omgevings Impact Rapport\n**Datum:** {now}\n\n"
                if intro_text:
                    md_out += f"{intro_text}\n\n---\n\n"
                
                md_out += "# DEEL 1: Natura 2000 Analyse\n\n"
                for area, data in results.items():
                    md_out += f"## {area}\n\n{data['summary']}\n\n"
                    if data['sources']: md_out += "**Bronnen:**\n" + "\n".join([f"- {s}" for s in data['sources']]) + "\n"
                    md_out += "\n---\n\n"
                
                st.session_state.natuur_analysis_md = md_out
                st.session_state.final_report_md = md_out 
                st.success("Stap 1 Voltooid!")
                st.rerun()
        except Exception as e:
            st.error(f"Fout: {e}")
        finally:
            st.session_state.analysis_running = False

if st.session_state.analysis_results:
    with st.expander("Bekijk resultaten Stap 1", expanded=False):
        st.markdown(st.session_state.natuur_analysis_md)

    st.markdown("---")
    st.header("Stap 2: Omgevingsvisie Impact Analyse (Optioneel)")
    st.markdown("Upload de Omgevingsvisie (PDF) om te controleren op ingrepen die de geselecteerde natuurgebieden kunnen raken.")

    omgevings_pdf = st.file_uploader("Upload Omgevingsvisie PDF", type="pdf", key="omgevingsvisie_uploader")

    # Als DEV MODE aanstaat, sta toe dat we Stap 2 draaien zónder een PDF te hoeven uploaden
    can_run_step_2 = (omgevings_pdf is not None) or dev_mode_active

    if can_run_step_2 and st.button("▶️ Start Stap 2 (Impact Analyse)"):
        # Lazy initialize openai client
        openai_client = get_openai_client(st.session_state.ai_provider)

        if not openai_client and not dev_mode_active:
            st.error("Kon OpenAI client niet laden. Controleer je secrets.")
        else:
            with st.spinner("Bezig met analyseren van volledige Omgevingsvisie..."):
                # Roep de juiste service aan (echt of mock)
                impact_result = services.analyze_local_pdf(
                    uploaded_file=omgevings_pdf, 
                    client=openai_client,
                    system_prompt="Je bent een expert in ruimtelijke ordening en ecologie.",
                    impact_prompt=IMPACT_PROMPT_FULL
                )
                bestandsnaam = services.get_pdf_name(omgevings_pdf)

                impact_md = "\n# DEEL 2: Ingreep-effect relaties\n\n"
                impact_md += f"**Geanalyseerd bestand:** {bestandsnaam}\n\n"
                impact_md += impact_result

                st.session_state.impact_analysis_md = impact_md
                
                st.session_state.final_report_md = st.session_state.natuur_analysis_md + "\n\n---\n\n" + impact_md
                st.success("Stap 2 Voltooid! Rapport bijgewerkt.")
                st.rerun()

    if st.session_state.impact_analysis_md:
        with st.expander("Bekijk resultaten Stap 2", expanded=True):
            st.markdown(st.session_state.impact_analysis_md)

        # --- NIEUW: STAP 3 ---
        st.markdown("---")
        st.header("Stap 3: Conclusies Genereren (Optioneel)")
        st.markdown("Selecteer de onderwerpen waarover u een concluderende paragraaf wilt genereren op basis van de impactanalyse (Stap 2).")

        # NIEUW: Lijst met onderwerpen voor de conclusie
        CONCLUSION_TOPICS = [
            "Stikstofdepositie", "Recreatiedruk en toename verstoring", "Behoud en versterken van natuur buiten Natura 2000 gebieden",
            "Algemene Samenvatting & Aanbevelingen"
        ]

        selected_topics = st.multiselect(
            "Kies onderwerpen voor de conclusie:",
            options=CONCLUSION_TOPICS,
            key='conclusion_topics_selected'
        )

        if st.button("▶️ Start Stap 3 (Genereer Conclusies)", disabled=not selected_topics):
            # Lazy initialize openai client
            openai_client = get_openai_client(st.session_state.ai_provider)

            if not openai_client and not dev_mode_active:
                st.error("Kon OpenAI client niet laden. Controleer je secrets.")
            else:
                conclusion_results = {}
                total_topics = len(selected_topics)
                progress_bar = st.progress(0, text="Voorbereiden van conclusies...")

                with st.spinner("Bezig met genereren van conclusies..."):
                    for i, topic in enumerate(selected_topics):
                        progress_bar.progress((i + 1) / total_topics, text=f"Bezig met: **{topic}** ({i+1}/{total_topics})")
                        
                        # In dev mode, gebruik een mock antwoord
                        if dev_mode_active:
                            import time
                            time.sleep(0.5)
                            conclusion_text = f"Dit is een mock-conclusie voor het onderwerp **{topic}**. De impactanalyse toont aan dat de geplande ontwikkelingen significante risico's met zich meebrengen."
                        else:
                            conclusion_text = generate_conclusion_paragraph(
                                topic=topic,
                                impact_analyse=st.session_state.impact_analysis_md,
                                client=openai_client,
                                conclusion_prompt_template=CONCLUSION_PROMPT_TEMPLATE
                            )
                        conclusion_results[topic] = conclusion_text

                progress_bar.empty()
                
                # Formatteer de resultaten naar Markdown
                conclusion_md = "\n# DEEL 3: Conclusies\n\n"
                for topic, text in conclusion_results.items():
                    conclusion_md += f"### Conclusie: {topic}\n"
                    conclusion_md += f"{text}\n\n"
                
                st.session_state.conclusion_results_md = conclusion_md
                
                # Werk het volledige rapport bij
                st.session_state.final_report_md = (st.session_state.natuur_analysis_md + "\n\n---\n\n" + st.session_state.impact_analysis_md + "\n\n---\n\n" + conclusion_md)
                st.success("Stap 3 Voltooid! Rapport bijgewerkt met conclusies.")
                st.rerun()

    if st.session_state.get('conclusion_results_md'):
        with st.expander("Bekijk resultaten Stap 3", expanded=True):
            st.markdown(st.session_state.conclusion_results_md)

# --- DOWNLOADS & STATS ---

if st.session_state.final_report_md:
    st.markdown("---")
    st.header("📥 Download Eindrapport")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇️ Download Volledig Rapport (.md)", st.session_state.final_report_md, f"rapport_compleet_{timestamp}.md")
    
    docx_data = convert_markdown_to_docx_bytes(st.session_state.final_report_md, st.session_state.get('map_image_buffer'))
    c2.download_button("⬇️ Download Volledig Rapport (.docx)", docx_data, f"rapport_compleet_{timestamp}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

    with c3:
        if st.button("Genereer Pandoc Document (Beta)"):
            with st.spinner("Document genereren met Pandoc..."):
                from services.pandoc_converter import convert_md_to_docx_pandoc
                st.session_state.pandoc_docx_data = convert_md_to_docx_pandoc(st.session_state.final_report_md)
        
        if st.session_state.get('pandoc_docx_data'):
            st.download_button("⬇️ Download Pandoc (.docx)", st.session_state.pandoc_docx_data, f"rapport_pandoc_{timestamp}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", key="pandoc_dl")

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