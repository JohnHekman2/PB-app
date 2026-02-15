import os
import re
import io
import pandas as pd
import geopandas as gpd
import warnings
import matplotlib.pyplot as plt # type: ignore
import contextily as cx # pyright: ignore[reportMissingImports]

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

def create_map_image(municipality_gdf: gpd.GeoDataFrame, areas_gdf: gpd.GeoDataFrame, municipality_name: str) -> io.BytesIO | None:
    """
    Genereert een kaart met de gemeente en de geselecteerde natuurgebieden.

    Args:
        municipality_gdf: GeoDataFrame met de geometrie van de gemeente.
        areas_gdf: GeoDataFrame met de geometrieën van de natuurgebieden.
        municipality_name: Naam van de geselecteerde gemeente voor de titel.

    Returns:
        Een io.BytesIO object met de PNG-afbeelding van de kaart, of None bij een fout.
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot de natuurgebieden
        areas_gdf.plot(ax=ax, color='green', edgecolor='black', alpha=0.7, label='Natura 2000 Gebieden')

        # Labels toevoegen voor elk natuurgebied
        placed_labels = set()
        for _, row in areas_gdf.iterrows():
            name = row['naam_n2k']
            if name in placed_labels:
                continue
            
            centroid = row.geometry.centroid
            ax.text(
                centroid.x, centroid.y, 
                name, 
                fontsize=7, ha='center', va='center', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1)
            )
            placed_labels.add(name)

        # Plot de gemeentegrens
        municipality_gdf.plot(ax=ax, color='red', edgecolor='darkred', alpha=0.5, label=f'Gemeente {municipality_name}')

        # Stel de limieten van de kaart in om alles te tonen
        total_bounds = pd.concat([municipality_gdf, areas_gdf]).total_bounds
        ax.set_xlim(total_bounds[0] - 1000, total_bounds[2] + 1000)
        ax.set_ylim(total_bounds[1] - 1000, total_bounds[3] + 1000)

        # Voeg een achtergrondkaart toe
        cx.add_basemap(ax, crs=areas_gdf.crs.to_string(), source=cx.providers.CartoDB.Positron)

        # Esthetische aanpassingen
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"Natuurgebieden nabij {municipality_name}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend()
        plt.tight_layout()

        # Sla de afbeelding op in een buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        plt.close(fig)  # Sluit de figuur om geheugen vrij te maken
        buf.seek(0)
        return buf

    except Exception as e:
        print(f"Fout bij het maken van de kaart: {e}")
        return None

def get_geodata_for_municipality(municipality: str) -> tuple[str, gpd.GeoDataFrame | None, gpd.GeoDataFrame | None]:
    """
    Selecteert de geo-data voor een gemeente en de nabijgelegen Natura 2000 gebieden.

    Args:
        municipality: De naam van de gemeente.

    Returns:
        Een tuple met (bericht, gemeente_gdf, natura2000_gebieden_gdf).
        Bij een fout zijn de GeoDataFrames None.
    """
    DOEL_GEMEENTE = municipality.strip()
    
    if not DOEL_GEMEENTE:
        return "Voer alstublieft een geldige gemeentenaam in.", None, None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            
            if not os.path.exists(PAD_GEMEENTEN) or not os.path.exists(PAD_NATURA2000):
                return "FOUT: Geo-bestanden niet gevonden in 'supportdata/'.", None, None

            gemeenten_gdf = gpd.read_file(PAD_GEMEENTEN)
            natura2000_gdf = gpd.read_file(PAD_NATURA2000)

            gemeenten_gdf = gemeenten_gdf.to_crs(DOEL_CRS)
            natura2000_gdf = natura2000_gdf.to_crs(DOEL_CRS)

            doel_gemeente_gdf_full = gemeenten_gdf[gemeenten_gdf['Gemeentenaam'] == DOEL_GEMEENTE].copy()

            if doel_gemeente_gdf_full.empty:
                return f"FOUT: Gemeente '{DOEL_GEMEENTE}' niet gevonden.", None, None

            # Maak één geometrie voor de hele gemeente
            doel_gemeente_gdf_unified = gpd.GeoDataFrame(geometry=[doel_gemeente_gdf_full.unary_union], crs=DOEL_CRS)

            selectie_met_afstand = gpd.sjoin_nearest(
                natura2000_gdf,  
                doel_gemeente_gdf_unified, 
                how="left",
                distance_col="kortste_afstand_m"
            )

            selectie_finaal_gefilterd = selectie_met_afstand[selectie_met_afstand['kortste_afstand_m'] <= ANALYSE_AFSTAND].copy()

            if selectie_finaal_gefilterd.empty:
                return f"Geen gebieden binnen {ANALYSE_AFSTAND / 1000} km van '{DOEL_GEMEENTE}'.", doel_gemeente_gdf_unified, None

            return f"Analyse geslaagd voor '{DOEL_GEMEENTE}'.", doel_gemeente_gdf_unified, selectie_finaal_gefilterd

    except Exception as e:
        return f"FOUT bij geo-analyse: {e}", None, None


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
    Genereert een CSV van Natura 2000 gebieden op basis van een gemeente.
    Dit is nu een wrapper om de `get_geodata_for_municipality` functie.
    """
    message, _, selectie_gdf = get_geodata_for_municipality(municipality)

    if selectie_gdf is None or selectie_gdf.empty:
        return None, message
    
    try:
        selectie_gdf['afstand_km'] = selectie_gdf['kortste_afstand_m'] / 1000.0
        selectie_df = selectie_gdf[['naam_n2k', 'afstand_km']].copy()

        # Debug: Sla de CSV ook lokaal op in de map
        debug_filename = f"debug_{municipality}.csv"
        selectie_df.to_csv(debug_filename, index=False)

        buffer = io.BytesIO()
        selectie_df.to_csv(buffer, index=False)
        buffer.seek(0)
        
        return buffer, f"CSV gegenereerd voor '{municipality}' ({len(selectie_df)} gebieden)."

    except Exception as e:
        return None, f"FOUT bij genereren CSV: {e}"