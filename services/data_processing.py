import os
import json
import re
from collections import Counter
import pandas as pd
import streamlit as st

# Local imports
from services.llm_service import get_vector_store, VECTOR_STORE_DIRECTORY
from services.geodata_service import PAD_GEMEENTEN, RUIS_WOORDEN

@st.cache_data
def get_all_area_names():
    # Probeer eerst de JSON cache te laden
    area_names_path = os.path.join(VECTOR_STORE_DIRECTORY, "area_names.json")
    if os.path.exists(area_names_path):
        try:
            with open(area_names_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading area_names.json: {e}")

    # Fallback: Ouderwetse methode (traag)
    print("Fallback: Loading unique area names from vector store...")
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

def calculate_dynamic_stopwords(all_names: list, frequency_threshold: float = 0.05):
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
    clean_name = name.lower()
    clean_name = re.sub(r'[^a-z0-9\s]+', ' ', clean_name)
    words = clean_name.split()
    all_stopwords = set(RUIS_WOORDEN)
    if dynamic_stopwords:
        all_stopwords.update(dynamic_stopwords)
    filtered_words = [w for w in words if w not in all_stopwords]
    clean_name = ' '.join(filtered_words)
    return re.sub(r'\s+', ' ', clean_name).strip()

def match_areas_from_csv(uploaded_file, all_available_areas: list, column_name: str = 'naam_n2k', threshold: int = 60):
    from thefuzz import process, fuzz
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

def parse_json_response(response_text: str):
    try:
        cleaned_text = re.sub(r'```json\s*', '', response_text)
        cleaned_text = re.sub(r'```\s*$', '', cleaned_text)
        
        match = re.search(r'(\{.*\})', cleaned_text, re.DOTALL)
        if match:
            cleaned_text = match.group(1)
        else:
            start_idx = cleaned_text.find('{')
            if start_idx != -1:
                cleaned_text = cleaned_text[start_idx:]
            
        cleaned_text = cleaned_text.strip()
            
        if not cleaned_text.endswith('}'):
            cleaned_text += '}'

        data = json.loads(cleaned_text, strict=False)
        return data
    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {e}")
        try:
            import ast
            return ast.literal_eval(cleaned_text)
        except Exception:
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

def flatten_results_to_df(results_dict):
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
