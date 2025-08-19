import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from shapely import wkb
import folium
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Configuration & Setup ---
NHPD_FILENAME = "Active and Inconclusive Properties.xlsx"
ACS_FILENAME = "data/acs5_2023_tract_preservation_risk_data.parquet"
OUTPUT_FILENAME = "data/tract_vulnerability_scores_final.csv"
MAP_OUTPUT = "index.html"
BOXPLOT_OUTPUT = "tier3_by_risk_tier.png"
SCATTERPLOT_OUTPUT = "rent_gap_vs_diversity.png"

# Define PBRA programs explicitly
PBRA_PROGRAMS = ['Sec 8 NC', 'Sec 8 SR', 'HFDA/8 NC', 'HFDA/8 SR', '515/8 NC', '515/8 SR']

# Set analysis timeframe: 10-year window from current date (June 25, 2025)
start_date = datetime(2025, 6, 25)
end_date = start_date + timedelta(days=10*365.25)
print(f"Analyzing PBRA contracts expiring between {start_date.date()} and {end_date.date()}")

# --- Step 1: Load and Prepare ACS Data ---
print("\n--- Loading and Preparing ACS Data ---")
try:
    acs_df = pd.read_parquet(ACS_FILENAME)
    print(f"Loaded ACS data with {len(acs_df)} tracts.")
except FileNotFoundError:
    print(f"ERROR: ACS data file not found at '{ACS_FILENAME}'. Please run the download script first.")
    exit()

acs_df['geometry'] = acs_df['geometry'].apply(lambda x: wkb.loads(x) if isinstance(x, bytes) else None)
acs_gdf = gpd.GeoDataFrame(acs_df, geometry='geometry', crs="EPSG:4326")

# Create CensusTract ID (11-digit GEOID) - ROBUST METHOD
if 'GEOID' in acs_gdf.columns:
    acs_gdf['CensusTract'] = acs_gdf['GEOID'].astype(str).str.zfill(11)
elif all(col in acs_gdf.columns for col in ['STATE', 'COUNTY', 'TRACT']):
    acs_gdf['CensusTract'] = (acs_gdf['STATE'].astype(str).str.zfill(2) +
                             acs_gdf['COUNTY'].astype(str).str.zfill(3) +
                             acs_gdf['TRACT'].astype(str).str.zfill(6))
else:
    print("ERROR: Cannot create CensusTract ID from ACS data. Missing required columns.")
    exit()

state_fips_to_name = {
    '01': 'Alabama', '02': 'Alaska', '04': 'Arizona', '05': 'Arkansas', '06': 'California',
    '08': 'Colorado', '09': 'Connecticut', '10': 'Delaware', '11': 'District of Columbia',
    '12': 'Florida', '13': 'Georgia', '15': 'Hawaii', '16': 'Idaho', '17': 'Illinois',
    '18': 'Indiana', '19': 'Iowa', '20': 'Kansas', '21': 'Kentucky', '22': 'Louisiana',
    '23': 'Maine', '24': 'Maryland', '25': 'Massachusetts', '26': 'Michigan', '27': 'Minnesota',
    '28': 'Mississippi', '29': 'Missouri', '30': 'Montana', '31': 'Nebraska', '32': 'Nevada',
    '33': 'New Hampshire', '34': 'New Jersey', '35': 'New Mexico', '36': 'New York',
    '37': 'North Carolina', '38': 'North Dakota', '39': 'Ohio', '40': 'Oklahoma',
    '41': 'Oregon', '42': 'Pennsylvania', '44': 'Rhode Island', '45': 'South Carolina',
    '46': 'South Dakota', '47': 'Tennessee', '48': 'Texas', '49': 'Utah', '50': 'Vermont',
    '51': 'Virginia', '53': 'Washington', '54': 'West Virginia', '55': 'Wisconsin',
    '56': 'Wyoming', '72': 'Puerto Rico'
}

# Calculate Tier 2 and Tier 3 variables
acs_gdf['Rental_Vacancy_Rate'] = acs_gdf['Vacant_Units_For_Rent'] / (acs_gdf['Renter_Occupied_Units'] + acs_gdf['Vacant_Units_For_Rent'])
cost_burden_cols = ['Rent_Burden_30_35_Pct', 'Rent_Burden_35_40_Pct', 'Rent_Burden_40_50_Pct', 'Rent_Burden_50_Plus_Pct']
acs_gdf['Cost_Burdened_Renters_Count'] = acs_gdf[cost_burden_cols].sum(axis=1)
acs_gdf['Renter_Cost_Burden_30_Plus_Pct'] = acs_gdf['Cost_Burdened_Renters_Count'] / acs_gdf['Total_Renter_HH_For_Burden']
acs_gdf['Poverty_Rate'] = acs_gdf['Pop_Below_Poverty'] / acs_gdf['Total_Pop_For_Poverty']
acs_gdf['Pct_Non_White'] = (acs_gdf['Total_Population'] - acs_gdf['Pop_Not_Hispanic_White_Alone']) / acs_gdf['Total_Population']
acs_gdf['Pct_Renter_Occupied'] = acs_gdf['Renter_Occupied_Units'] / acs_gdf['Total_Occupied_Units']
acs_gdf.replace([np.inf, -np.inf], np.nan, inplace=True)

# Select and rename final ACS columns
acs_final_cols = {
    'CensusTract': 'CensusTract', 'Median_Gross_Rent': 'Market_Rent_ACS', 'Rental_Vacancy_Rate': 'Rental_Vacancy_Rate',
    'Renter_Cost_Burden_30_Plus_Pct': 'Renter_Cost_Burden_Pct', 'Median_Household_Income': 'Median_Household_Income',
    'Poverty_Rate': 'Poverty_Rate', 'Pct_Non_White': 'Pct_Non_White', 'Pct_Renter_Occupied': 'Pct_Renter_Occupied',
    'Median_Year_Structure_Built': 'Median_Year_Built', 'geometry': 'geometry'
}
acs_prepared_df = acs_gdf[list(acs_final_cols.keys())].rename(columns=acs_final_cols)
print("ACS data prepared with market, community, and geometry variables.")

# --- Step 2: Load and Prepare NHPD Data ---
print("\n--- Loading and Preparing NHPD Data ---")
try:
    nhpd_df = pd.read_excel(NHPD_FILENAME)
    print(f"Loaded NHPD data with {len(nhpd_df)} properties.")
except FileNotFoundError:
    print(f"ERROR: NHPD data file not found at '{NHPD_FILENAME}'.")
    exit()

if 'FairMarketRent_2BR' not in nhpd_df.columns:
    print("ERROR: FairMarketRent_2BR not found. You may need to merge HUD FMR data.")
    exit()

try:
    max_contract_num = max([int(re.search(r'S8_(\d+)_', col).group(1)) for col in nhpd_df.columns if re.search(r'S8_(\d+)_', col)])
except ValueError:
    print("ERROR: No Section 8 contract columns found. Exiting.")
    exit()

all_contracts_list = []
for i in range(1, max_contract_num + 1):
    contract_cols = {
        'NHPDPropertyID': 'NHPDPropertyID', 'CensusTract': 'CensusTract', 'FairMarketRent_2BR': 'FairMarketRent_2BR',
        f'S8_{i}_Status': 'Status', f'S8_{i}_ProgramName': 'ProgramName', f'S8_{i}_EndDate': 'EndDate',
        f'S8_{i}_AssistedUnits': 'AssistedUnits', f'S8_{i}_RentToFMR': 'RentToFMR'
    }
    if f'S8_{i}_Status' in nhpd_df.columns:
        contract_df = nhpd_df[list(contract_cols.keys())].dropna(subset=[f'S8_{i}_Status']).copy()
        contract_df.rename(columns=contract_cols, inplace=True)
        all_contracts_list.append(contract_df)

nhpd_long = pd.concat(all_contracts_list, ignore_index=True)
nhpd_long['EndDate'] = pd.to_datetime(nhpd_long['EndDate'], errors='coerce')

at_risk_contracts = nhpd_long[
    (nhpd_long['Status'] == 'Active') &
    (nhpd_long['ProgramName'].isin(PBRA_PROGRAMS)) &
    (nhpd_long['EndDate'] >= start_date) &
    (nhpd_long['EndDate'] <= end_date)
].copy()

at_risk_contracts = at_risk_contracts.dropna(subset=['EndDate', 'AssistedUnits', 'RentToFMR', 'FairMarketRent_2BR'])
print(f"\nIdentified {len(at_risk_contracts)} at-risk PBRA contracts after dropping missing data.")

if at_risk_contracts.empty:
    print("No at-risk PBRA contracts found. Exiting.")
    exit()

if at_risk_contracts['RentToFMR'].max() > 10:
    at_risk_contracts['Subsidized_Rent_Est'] = (at_risk_contracts['RentToFMR'] / 100) * at_risk_contracts['FairMarketRent_2BR']
else:
    at_risk_contracts['Subsidized_Rent_Est'] = at_risk_contracts['RentToFMR'] * at_risk_contracts['FairMarketRent_2BR']

# --- Step 3: Aggregate to Census Tract Level ---
print("\n--- Aggregating At-Risk Data to Census Tract Level ---")
# ROBUST METHOD: Ensure CensusTract is a cleaned string before aggregation
at_risk_contracts['CensusTract'] = at_risk_contracts['CensusTract'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(11)

at_risk_contracts['rent_x_units'] = at_risk_contracts['Subsidized_Rent_Est'] * at_risk_contracts['AssistedUnits']

tract_agg = at_risk_contracts.groupby('CensusTract').agg(
    Total_Expiring_Units=('AssistedUnits', 'sum'),
    rent_x_units_sum=('rent_x_units', 'sum')
).reset_index()

tract_agg['Avg_Subsidized_Rent'] = np.divide(
    tract_agg['rent_x_units_sum'], tract_agg['Total_Expiring_Units'],
    out=np.zeros_like(tract_agg['rent_x_units_sum'], dtype=float),
    where=tract_agg['Total_Expiring_Units'] != 0
)
tract_summary_df = tract_agg[['CensusTract', 'Total_Expiring_Units', 'Avg_Subsidized_Rent']]

# --- Step 4: Merge with ACS Data ---
print("\n--- Merging NHPD and ACS Data ---")
merged_df = pd.merge(tract_summary_df, acs_prepared_df, on='CensusTract', how='left')

# CRITICAL CHECK: Ensure the merge was successful before proceeding
if merged_df['Market_Rent_ACS'].isna().all():
    print("\nFATAL ERROR: Merge between NHPD and ACS data failed.")
    print("No matching Census Tracts found. Check for data type or formatting mismatches in 'CensusTract' columns.")
    exit()

# Filter for valid data *after* confirming merge success
merged_df = merged_df[(merged_df['Market_Rent_ACS'] > 0) & (merged_df['Avg_Subsidized_Rent'] > 0)]

# CRITICAL CHECK #2: Ensure dataframe is not empty after filtering
if merged_df.empty:
    print("\nFATAL ERROR: The dataframe is empty after filtering for valid rent data.")
    print("This means no at-risk tracts had valid, positive market and subsidized rent values.")
    exit()

merged_df['State_FIPS'] = merged_df['CensusTract'].str[:2]
merged_df['State_Name'] = merged_df['State_FIPS'].map(state_fips_to_name)
merged_df['Rent_Gap'] = merged_df['Market_Rent_ACS'] - merged_df['Avg_Subsidized_Rent']
merged_df['Rent_Gap'] = merged_df['Rent_Gap'].clip(lower=0).fillna(0)
print(f"Successfully merged and filtered data for {len(merged_df)} at-risk tracts.")

# --- Step 5: Calculate Vulnerability Score ---
print("\n--- Calculating Final Vulnerability Score ---")
def min_max_scaler(series):
    if series.max() == series.min() or pd.isna(series.max()):
        return pd.Series(0.5, index=series.index)
    return (series - series.min()) / (series.max() - series.min())

merged_df['Financial_Incentive_Score'] = min_max_scaler(merged_df['Rent_Gap'])
merged_df['Vacancy_Pressure_Score'] = 1 - min_max_scaler(merged_df['Rental_Vacancy_Rate'])
merged_df['Cost_Burden_Pressure_Score'] = min_max_scaler(merged_df['Renter_Cost_Burden_Pct'])
merged_df['Market_Pressure_Score'] = (merged_df['Vacancy_Pressure_Score'].fillna(0.5) +
                                     merged_df['Cost_Burden_Pressure_Score'].fillna(0.5)) / 2

weight_financial = 0.5
weight_market = 0.5
merged_df['Vulnerability_Score'] = (weight_financial * merged_df['Financial_Incentive_Score'].fillna(0.5)) + \
                                  (weight_market * merged_df['Market_Pressure_Score'].fillna(0.5))
print(f"Vulnerability Score calculated with weights: Financial={weight_financial}, Market={weight_market}")

# --- Step 6: Profile High-Risk Tracts & Incorporate Ownership Data ---
print("\n--- Profiling Tracts by Risk Tier ---")
merged_df['Risk_Tier'] = pd.qcut(
    merged_df['Vulnerability_Score'], q=3, labels=['Stable', 'Tipping-Point', 'Acute-Risk'], duplicates='drop')

# --- NEW ANALYSIS: Incorporating Owner Type ---
print("\n--- Analyzing Ownership Structure Across Risk Tiers ---")

def clean_owner_type(owner_string):
    if pd.isna(owner_string):
        return 'Unknown'
    owner_string = str(owner_string).upper()
    if 'NON PROFIT' in owner_string or 'NON-PROFIT' in owner_string:
        return 'Non-Profit'
    if 'PROFIT' in owner_string or 'DIVIDEND' in owner_string or 'PARTNERSHIP' in owner_string:
        return 'For-Profit'
    if 'PUBLIC' in owner_string:
        return 'Public Entity'
    return 'Other/Unknown'

nhpd_df['OwnerType_Clean'] = nhpd_df['OwnerType'].apply(clean_owner_type)
print("Cleaned NHPD OwnerType data into 'Non-Profit', 'For-Profit', etc.")

owner_type_mapping = nhpd_df[['NHPDPropertyID', 'OwnerType_Clean']]
at_risk_contracts_with_owner = pd.merge(at_risk_contracts, owner_type_mapping, on='NHPDPropertyID', how='left')

tract_owner_agg = at_risk_contracts_with_owner.groupby(['CensusTract', 'OwnerType_Clean'])['AssistedUnits'].sum().unstack(fill_value=0)

if 'For-Profit' in tract_owner_agg.columns:
    tract_owner_agg['Total_Units_In_Tract'] = tract_owner_agg.sum(axis=1)
    tract_owner_agg['Pct_For_Profit_Units'] = tract_owner_agg['For-Profit'] / tract_owner_agg['Total_Units_In_Tract']
else:
    tract_owner_agg['Pct_For_Profit_Units'] = 0

merged_df = pd.merge(merged_df, tract_owner_agg[['Pct_For_Profit_Units']], on='CensusTract', how='left')
merged_df['Pct_For_Profit_Units'] = merged_df['Pct_For_Profit_Units'].fillna(0)

# --- Final Profile Table Generation ---
profile_cols = [
    'Total_Expiring_Units', 'Rent_Gap', 'Market_Rent_ACS', 'Rental_Vacancy_Rate',
    'Renter_Cost_Burden_Pct', 'Median_Household_Income', 'Poverty_Rate', 'Pct_Non_White',
    'Pct_Renter_Occupied', 'Median_Year_Built', 'Pct_For_Profit_Units'  # Added new column
]
merged_df['Median_Year_Built'] = merged_df['Median_Year_Built'].fillna(0).astype(int)

risk_profile_table = merged_df.groupby('Risk_Tier', observed=False)[profile_cols].mean()
print("\n### Final Risk Profile Table (with Ownership Data) ###")
print(risk_profile_table)

# Statistical analysis: ANOVA for tier differences
print("\nANOVA p-values for Tier Differences:")
for col in profile_cols:
    groups = [merged_df[merged_df['Risk_Tier'] == tier][col].dropna() for tier in ['', '', '']]
    if all(len(g) > 1 for g in groups): # ANOVA requires at least 2 data points per group
        stat, p = f_oneway(*groups)
        print(f"{col}: p-value = {p:.4f}")
    else:
        print(f"{col}: Insufficient data for ANOVA.")

# --- Step 7: Visualizations ---
print("\n--- Generating Visualizations ---")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
community_profile_cols = ['Poverty_Rate', 'Pct_Non_White', 'Pct_Renter_Occupied']
for ax, col in zip(axes.flatten(), community_profile_cols):
    sns.boxplot(x='Risk_Tier', y=col, data=merged_df, ax=ax, order=['Stable', 'Tipping-Point', 'Acute-Risk'])
    ax.set_title(f'Community Profile: {col}')
plt.tight_layout()
plt.savefig(BOXPLOT_OUTPUT)
print(f"Boxplots saved as '{BOXPLOT_OUTPUT}'")

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='Rent_Gap', y='Pct_Non_White', hue='Risk_Tier', size='Total_Expiring_Units',hue_order=['Stable', 'Tipping-Point', 'Acute-Risk'], data=merged_df[merged_df['Rent_Gap'] > 0], ax=ax)
plt.title('Rent Gap (Financial Incentive) vs. Percent Non-White (Community Profile)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.savefig(SCATTERPLOT_OUTPUT)
print(f"Scatterplot saved as '{SCATTERPLOT_OUTPUT}'")

print("\n--- Generating Folium Choropleth Map ---")
if 'geometry' in merged_df.columns and merged_df['geometry'].notna().any():
    gdf = gpd.GeoDataFrame(merged_df, geometry='geometry', crs="EPSG:4326")
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4, tiles='cartodbpositron')
    geojson_data = gdf.to_crs("EPSG:4326").__geo_interface__
    folium.Choropleth(
        geo_data=geojson_data, name='Vulnerability Score', data=gdf,
        columns=['CensusTract', 'Vulnerability_Score'], key_on='feature.properties.CensusTract',
        fill_color='YlOrRd', fill_opacity=0.7, line_opacity=0.2,
        legend_name='Vulnerability Score (Likelihood of Conversion)', highlight=True
    ).add_to(m)
    tooltip = folium.GeoJsonTooltip(
        fields=['CensusTract', 'Vulnerability_Score', 'Risk_Tier', 'Total_Expiring_Units', 'Rent_Gap', 'Pct_For_Profit_Units'],
        aliases=['Tract ID:', 'Vulnerability Score:', 'Risk Tier:', 'Expiring Units:', 'Rent Gap ($):', '% For-Profit Units:'],
        localize=True, sticky=True
    )
    folium.GeoJson(geojson_data, style_function=lambda x: {'fillOpacity': 0, 'weight': 0.1}, tooltip=tooltip).add_to(m)
    folium.LayerControl().add_to(m)
    m.save(MAP_OUTPUT)
    print(f"Folium choropleth map saved as '{MAP_OUTPUT}'")
else:
    print("WARNING: No valid geometry data for choropleth mapping.")

# --- Step 8: Sensitivity Analysis ---
print("\n--- Conducting Weighting Sensitivity Analysis ---")
weight_scenarios = {'Motive_Heavy_70_30': (0.7, 0.3), 'Opportunity_Heavy_30_70': (0.3, 0.7)}
top_tracts_by_scenario = {}
N_TOP_TRACTS = int(len(merged_df) * 0.10)
print(f"Comparing the Top {N_TOP_TRACTS} highest-risk tracts under different model weights...")

for scenario_name, (w_financial, w_market) in weight_scenarios.items():
    merged_df[scenario_name] = (w_financial * merged_df['Financial_Incentive_Score'].fillna(0.5)) + \
                               (w_market * merged_df['Market_Pressure_Score'].fillna(0.5))
    top_tracts = merged_df.nlargest(N_TOP_TRACTS, scenario_name)['CensusTract'].tolist()
    top_tracts_by_scenario[scenario_name] = top_tracts

def jaccard_similarity(list1, list2):
    set1, set2 = set(list1), set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0.0

tracts_motive_heavy = top_tracts_by_scenario['Motive_Heavy_70_30']
tracts_opportunity_heavy = top_tracts_by_scenario['Opportunity_Heavy_30_70']
overlap_score = jaccard_similarity(tracts_motive_heavy, tracts_opportunity_heavy)

print("\n--- Weighting Sensitivity Analysis Results ---")
print(f"Jaccard Similarity between 'Motive-Heavy (70/30)' and 'Opportunity-Heavy (30/70)' scenarios: {overlap_score:.2f}")

if overlap_score > 0.75:
    print("Interpretation: The model is highly robust.")
elif overlap_score > 0.5:
    print("Interpretation: The model is moderately robust, highlighting different 'flavors' of risk.")
else:
    print("Interpretation: The model is sensitive to weighting.")

# --- Step 9: Save Final Data ---
final_output_df = merged_df[[
    'CensusTract', 'State_Name', 'Vulnerability_Score', 'Risk_Tier', 'Total_Expiring_Units', 'Rent_Gap',
    'Financial_Incentive_Score', 'Market_Pressure_Score'
] + profile_cols + ['geometry']].sort_values(by='Vulnerability_Score', ascending=False)
final_output_df = final_output_df.loc[:,~final_output_df.columns.duplicated()] # Remove duplicate columns before saving

if 'geometry' in final_output_df.columns:
    final_output_df['geometry'] = final_output_df['geometry'].map(lambda x: wkb.dumps(x) if x is not None else None)

final_output_df.to_csv(OUTPUT_FILENAME, index=False)
print(f"\nAnalysis complete. Final data with vulnerability scores saved to:\n{OUTPUT_FILENAME}")






# --- Conduct Weighting Sensitivity Analysis ---
print("\n--- Conducting Weighting Sensitivity Analysis ---")

# Define the weight combinations to test [Weight_Financial, Weight_Market]
weight_scenarios = {
    'Motive_Heavy_70_30': (0.7, 0.3),
    'Opportunity_Heavy_30_70': (0.3, 0.7)
}

# Dictionary to store the list of top tracts for each alternative scenario
top_tracts_by_scenario = {}

# Use approximately the top 10% of tracts for comparison
N_TOP_TRACTS = int(len(final_output_df) * 0.10)

print(f"Comparing the Top {N_TOP_TRACTS} highest-risk tracts under different model weights...")

# Loop through each alternative scenario to calculate a new score and get top tracts
for scenario_name, (w_financial, w_market) in weight_scenarios.items():
    # Calculate the Vulnerability Score for the current scenario
    final_output_df[scenario_name] = (w_financial * final_output_df['Financial_Incentive_Score'].fillna(0.5)) + \
                              (w_market * final_output_df['Market_Pressure_Score'].fillna(0.5))
    # Get the list of the top N tracts based on this new score
    top_tracts = final_output_df.nlargest(N_TOP_TRACTS, scenario_name)['CensusTract'].tolist()
    # Store the list of tracts in our dictionary
    top_tracts_by_scenario[scenario_name] = set(top_tracts) # Use sets for efficient comparison

# --- Compare Scenarios and Calculate Metrics for the Paper's Table ---
set_a = top_tracts_by_scenario['Motive_Heavy_70_30']
set_b = top_tracts_by_scenario['Opportunity_Heavy_30_70']

intersection = len(set_a.intersection(set_b))
union = len(set_a.union(set_b))
unique_to_a = len(set_a.difference(set_b))
unique_to_b = len(set_b.difference(set_a))
jaccard_score = intersection / union if union > 0 else 0

# --- Print the Results in a LaTeX-ready format ---
print("\n--- Weighting Sensitivity Analysis Results ---")
print("Copy the following into a new table in your LaTeX document:")
print("\\begin{table}[h!]")
print("\\centering")
print("\\caption{Weighting Sensitivity Analysis Results}")
print("\\begin{tabular}{lr}")
print("\\toprule")
print("\\textbf{Metric} & \\textbf{Value} \\\\")
print("\\midrule")
print(f"Scenario A & Motive-Heavy (70\\% Financial / 30\\% Market) \\\\")
print(f"Scenario B & Opportunity-Heavy (30\\% Financial / 70\\% Market) \\\\")
print("\\midrule")
print(f"Top Tracts Compared (N) & {N_TOP_TRACTS} \\\\")
print(f"Tracts in Both Scenarios (Intersection) & {intersection} \\\\")
print(f"Tracts Unique to Scenario A & {unique_to_a} \\\\")
print(f"Tracts Unique to Scenario B & {unique_to_b} \\\\")
print(f"\\textbf{{Jaccard Similarity}} & \\textbf{{{jaccard_score:.2f}}} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\label{tab:sensitivity}")
print("\\end{table}")