# PBRA Analysis

This repository contains a Python script for analyzing Project-Based Rental Assistance (PBRA) contracts that are at risk of expiring. The analysis combines National Housing Preservation Database (NHPD) data with American Community Survey (ACS) metrics to compute tract-level vulnerability scores and generate visualizations.

## Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Acquire the necessary input files:
   - `Active and Inconclusive Properties.xlsx` from the NHPD.
   - `acs5_2023_tract_preservation_risk_data.parquet` containing ACS data.
2. Run the analysis script:

```bash
python scripts/PBRA_analysis.py
```

The script outputs a tract-level CSV file along with map and plot visualizations.

## Outputs

- `data/tract_vulnerability_scores_final.csv`
- `index.html`
- `tier3_by_risk_tier.png`
- `rent_gap_vs_diversity.png`

## Data Availability

Data is available upon request and can also be accessed via Zenodo: https://zenodo.org/records/16907864
