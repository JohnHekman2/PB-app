import geopandas as gpd
import warnings
import pandas as pd
import sys # Nodig voor het netjes afsluiten bij fouten

# --- Configuratie ---

# TODO: Vervang deze paden met de locaties van je gedownloade bestanden.
PAD_GEMEENTEN = "supportdata/Gemeentegrenzen.gml"  # GML of ander formaat voor gemeentegrenzen
PAD_NATURA2000 = "supportdata/natura2000.gpkg"  # GPKG of ander formaat voor Natura 2000
PAD_RESULTAAT = "resultaat_selectie.csv"  # Output bestand

# DE GEMEENTE VOOR DE ANALYSE (MOET EXACT overeenkomen met de waarde in de 'Gemeentenaam' kolom van je GML)
DOEL_GEMEENTE = "Deventer" # <-- PAS DEZE WAARDE AAN!

# Het Nederlandse Coördinaat Referentie Systeem (CRS)
DOEL_CRS = "EPSG:28992"

# De afstand voor de selectie (in meters)
ANALYSE_AFSTAND = 25000  # 25 kilometer

# --- Stap 1: Data Inladen ---

print(f"Stap 1: Data inladen...")
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        
        gemeenten_gdf = gpd.read_file(PAD_GEMEENTEN)
        natura2000_gdf = gpd.read_file(PAD_NATURA2000)

    print(f"  > Gemeenten ingeladen: {len(gemeenten_gdf)} records")
    print(f"  > Natura 2000 ingeladen: {len(natura2000_gdf)} records")

except Exception as e:
    print(f"FOUT: Kon bestanden niet inladen. Controleer de 'PAD_...' variabelen.")
    print(f"Error details: {e}")
    sys.exit(1)

# --- Stap 2: CRS Controleren en Transformeren ---

print(f"\nStap 2: Coördinatensystemen (CRS) controleren...")

try:
    gemeenten_gdf = gemeenten_gdf.to_crs(DOEL_CRS)
    natura2000_gdf = natura2000_gdf.to_crs(DOEL_CRS)
    print(f"  > Beide datasets zijn nu in CRS: {DOEL_CRS}")
except Exception as e:
    print(f"FOUT: Kon CRS niet transformeren. Error: {e}")
    sys.exit(1)

# --- Stap 2a: Filteren op Doelgemeente ---

print(f"\nStap 2a: Selecteren van doelgemeente '{DOEL_GEMEENTE}'...")

# Filter de gemeenten GeoDataFrame op de ingevoerde naam (gebruik nu 'Gemeentenaam' met hoofdletter)
doel_gemeente_gdf = gemeenten_gdf[gemeenten_gdf['Gemeentenaam'] == DOEL_GEMEENTE]

if doel_gemeente_gdf.empty:
    print(f"FOUT: Gemeente '{DOEL_GEMEENTE}' niet gevonden in de data. Controleer de spelling en de kolomnaam ('Gemeentenaam') in de GML file.")
    sys.exit(1)

# Optioneel: Combineer eventuele meerdere rijen (bijvoorbeeld eilanden) tot één contourobject
# Dit is veiliger voor sjoin_nearest, maar de functie werkt ook met meerdere kleine geometrieën.
doel_gemeente_gdf = gpd.GeoDataFrame(geometry=[doel_gemeente_gdf.unary_union], crs=DOEL_CRS)

print(f"  > Gemeente '{DOEL_GEMEENTE}' geselecteerd.")

# --- Stap 3: Bereken de *exacte* kortste afstand ---

print("\nStap 3: Kortste afstanden berekenen tussen alle N2000 gebieden en de geselecteerde gemeente...")

# Gebruik sjoin_nearest om de kortste afstand voor *alle* N2000-gebieden te vinden
# tot de ENE geselecteerde gemeente-contour.
selectie_met_afstand = gpd.sjoin_nearest(
    natura2000_gdf,  # Alle N2000 gebieden
    doel_gemeente_gdf, # De enkele gemeente contour
    how="left",
    distance_col="kortste_afstand_m"
)

# Er zijn nu geen duplicaten te verwachten, omdat er maar één doelcontour is.
# We hoeven de drop_duplicates stap niet te herhalen.

print(f"  > Afstanden berekend voor alle {len(selectie_met_afstand)} N2000-gebieden tot {DOEL_GEMEENTE}.")


# --- Stap 4: Filteren op de 25 km eis ---

print(f"\nStap 4: Filteren op maximale afstand van {ANALYSE_AFSTAND / 1000} km...")

# Filter de berekende resultaten op de afstandseis.
selectie_finaal_gefilterd = selectie_met_afstand[selectie_met_afstand['kortste_afstand_m'] <= ANALYSE_AFSTAND].copy()

print(f"  > Filter voltooid: {len(selectie_finaal_gefilterd)} Natura 2000-gebieden liggen binnen {ANALYSE_AFSTAND / 1000} km van de gemeente {DOEL_GEMEENTE}.")


# --- Stap 5: Resultaat Opslaan (ALS CSV) ---

print(f"\nStap 5: Resultaat opslaan (als CSV)...")
try:
    # 1. Bereken afstand in kilometers
    selectie_finaal_gefilterd['afstand_km'] = selectie_finaal_gefilterd['kortste_afstand_m'] / 1000.0
    
    # 2. Selecteer alleen de gewenste output kolommen: naam en afstand in km
    selectie_df = selectie_finaal_gefilterd[[
        'naam_n2k',          # Naam van het Natura 2000 gebied (zoals gevraagd)
        'afstand_km',        # Berekende afstand in kilometers
    ]].copy()
    
    # Opslaan in CSV-formaat
    selectie_df.to_csv(PAD_RESULTAAT, index=False)
    print(f"  > Selectie succesvol opgeslagen in: {PAD_RESULTAAT}. Kolommen: naam_n2k, afstand_km.")

except Exception as e:
    print(f"FOUT: Kon resultaat niet opslaan. Error: {e}")


print("\n--- Script voltooid ---")