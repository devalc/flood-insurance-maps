# 🗺️ Flood Insurance Maps


## Visualizing NFIP Flood Insurance Coverage vs. Claim Payments at the U.S. County Level

![Bivariate choropleth map of NFIP netBuildingPaymentAmount (claim payments) and totalBuildingInsuranceCoverage (insurance coverage) across U.S. counties](figures/coverage_vs_payments_figure.png)
*Alt text: U.S. county-level map showing a bivariate color scale comparing net building payment amounts (y-dimension) and total building insurance coverage (x-dimension). Darker purples indicate counties with both high coverage and high payments. Lighter colors show lower values or missing data. Insets include Alaska, Hawaii, Puerto Rico, and U.S. territories.*

---

## 📊 Datasets Used

- **NFIP Redacted Claims Dataset**:  
  https://www.fema.gov/openfema-data-page/fima-nfip-redacted-claims-v2  
- **NFIP Policies Dataset**:  
  https://www.fema.gov/openfema-data-page/fima-nfip-redacted-policies-v2  
- **U.S. County Geometries**:  
  https://www2.census.gov/geo/tiger/TIGER2024/COUNTY/


## 🗂️ Repository Structure

📁 figures/

├── coverage_vs_payments_figure.png      Final bivariate choropleth map

📁 scripts/

├── download_from_openFEMA.py           # Downloads NFIP datasets from FEMA

└── plot_bivariate_choropleth.py        # Processes data and creates the map




---

## 💡 Inspiration 

This work was inspired by [@Marc's](https://www.linkedin.com/in/markebauer/) [deep dive](https://github.com/mebauer/duckdb-fema-nfip) into  NFIP datasets that got me thinking about how coverage and claims vary spatially.