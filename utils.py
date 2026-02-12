import os
import re
import io
import pandas as pd
import geopandas as gpd
import warnings

# --- Configuratie (Paden naar Geo-bestanden) ---
PAD_GEMEENTEN = "supportdata/Gemeentegrenzen.gml" 
PAD_NATURA2000 = "supportdata/natura2000.gpkg" 

# De afstand voor de selectie (in meters)
ANALYSE_AFSTAND = 25000 
# Het Nederlandse Coördinaat Referentie Systeem (CRS)
DOEL_CRS = "EPSG:28992"

# --- NIEUW: Lijst met woorden die we negeren bij matching ---
RUIS_WOORDEN = [
    'landgoederen', 'concept', 'natuurdoelanalyse', 'nda', 'ov', 'gelderland', 
    'overijssel', 'utrecht', 'noord', 'zuid', 'holland', 'brabant', 'limburg',
    'zeeland', 'flevoland', 'groningen', 'friesland', 'drenthe', 'beheerplan','doelenanalyse',
    'assessment', 'bijlage', 'definitief', 'versie', 'v1', 'v2', 'v3','compressed','natura2000','natura',
]

def get_area_name(file_path: str) -> str:
    """
    Extraheert de unieke gebiedsnaam door ruis-woorden, getallen en 
    niet-alphanumerieke tekens agressief te verwijderen.
    """
    # 1. Haal de basisnaam op (zonder extensie)
    filename_base = os.path.splitext(os.path.basename(file_path))[0]
    
    # 2. Naar lowercase
    clean_name = filename_base.lower()

    # 3. Verwijder jaartallen en versienummers (bijv. 20221213 of 050)
    # We verwijderen alle woorden die alleen uit getallen bestaan
    clean_name = re.sub(r'\b\d+\b', ' ', clean_name)

    # 4. Vervang alle niet-letters door spaties
    clean_name = re.sub(r'[^a-z\s]+', ' ', clean_name)
    
    # 5. Verwijder de specifieke RUIS_WOORDEN
    # We gebruiken \b om alleen hele woorden te matchen
    for woord in RUIS_WOORDEN:
        clean_name = re.sub(rf'\b{woord}\b', ' ', clean_name, flags=re.IGNORECASE)
    
    # 6. Verwijder dubbele spaties en strip
    display_name = re.sub(r'\s+', ' ', clean_name).strip()
    
    return display_name


def generate_csv_from_municipality(municipality: str) -> tuple[io.BytesIO | None, str]:
    """
    Voert een GeoPandas spatial join uit om Natura 2000 gebieden te selecteren.
    """
    DOEL_GEMEENTE = municipality.strip()
    
    if not DOEL_GEMEENTE:
        return None, "Voer alstublieft een geldige gemeentenaam in."

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            
            if not os.path.exists(PAD_GEMEENTEN) or not os.path.exists(PAD_NATURA2000):
                return None, "FOUT: Geo-bestanden niet gevonden in 'supportdata/'."

            gemeenten_gdf = gpd.read_file(PAD_GEMEENTEN)
            natura2000_gdf = gpd.read_file(PAD_NATURA2000)

            gemeenten_gdf = gemeenten_gdf.to_crs(DOEL_CRS)
            natura2000_gdf = natura2000_gdf.to_crs(DOEL_CRS)

            doel_gemeente_gdf = gemeenten_gdf[gemeenten_gdf['Gemeentenaam'] == DOEL_GEMEENTE].copy()

            if doel_gemeente_gdf.empty:
                return None, f"FOUT: Gemeente '{DOEL_GEMEENTE}' niet gevonden."

            doel_gemeente_gdf = gpd.GeoDataFrame(geometry=[doel_gemeente_gdf.unary_union], crs=DOEL_CRS)

            selectie_met_afstand = gpd.sjoin_nearest(
                natura2000_gdf,  
                doel_gemeente_gdf, 
                how="left",
                distance_col="kortste_afstand_m"
            )

            selectie_finaal_gefilterd = selectie_met_afstand[selectie_met_afstand['kortste_afstand_m'] <= ANALYSE_AFSTAND].copy()

            if selectie_finaal_gefilterd.empty:
                return None, f"Geen gebieden binnen {ANALYSE_AFSTAND / 1000} km van '{DOEL_GEMEENTE}'."

            selectie_finaal_gefilterd['afstand_km'] = selectie_finaal_gefilterd['kortste_afstand_m'] / 1000.0
            selectie_df = selectie_finaal_gefilterd[['naam_n2k', 'afstand_km']].copy()
            
            buffer = io.BytesIO()
            selectie_df.to_csv(buffer, index=False)
            buffer.seek(0)
            
            return buffer, f"CSV gegenereerd voor '{DOEL_GEMEENTE}' ({len(selectie_df)} gebieden)."

    except Exception as e:
        return None, f"FOUT bij geo-analyse: {e}"