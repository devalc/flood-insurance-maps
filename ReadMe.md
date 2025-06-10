# 🗺️ Flood Insurance Maps


## Visualizing NFIP Flood Insurance Coverage vs. Claim Payments at the U.S. County Level

![Bivariate choropleth map of NFIP netBuildingPaymentAmount (claim payments) and totalBuildingInsuranceCoverage (insurance coverage) across U.S. counties](figures/coverage_vs_payments_figure.png)
*Alt text: U.S. county-level map showing a bivariate color scale comparing net building payment amounts (y-dimension) and total building insurance coverage (x-dimension). Darker purples indicate counties with both high coverage and high payments. Lighter colors show lower values or missing data. Insets include Alaska, Hawaii, Puerto Rico, and U.S. territories.*

---
## 🎞️ Animation of the Costliest Flood Events

A dynamic map showing the **Top 10 Costliest Flood Events** in U.S. history based on total NFIP claim payments, **adjusted to 2025 dollars**.

Each frame highlights one event, mapping both:
- **Total claim payments**, and  
- **Insurance coverage**,  
at the county level using a bivariate choropleth.

![Animated map showing bivariate choropleth for each of the top 10 costliest flood events](figures/top_10_flood_events_animated.gif)  
*Alt text: Animated U.S. county maps showing the geographical distribution of NFIP claim payments and insurance coverage for each of the top 10 flood events by adjusted payouts.*

---

![Geographical distribution of NFIP flood insurance claim payments and total building insurance coverage across U.S. counties for the 10 costliest flood events (adjusted to 2025 dollars). This map highlights spatial patterns of financial risk and protection during major flood disasters.](figures/top_10_all_events_bivariate_map.png)
*Alt text: Geographical distribution of NFIP flood insurance claim payments and total building insurance coverage across U.S. counties for the 10 costliest flood events (adjusted to 2025 dollars). This map highlights spatial patterns of financial risk and protection during major flood disasters.*
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

└──  coverage_vs_payments_figure.png      # Final bivariate choropleth map

└── top_10_flood_events_animated.mp4      # Top 10 costliest flood events animated

└── top_10_flood_events_animated.gif      # Top 10 costliest flood events animated

└── top_10_flood_events_animated.png, event_*.png      # Top 10 costliest flood events static images

📁 scripts/

└── download_from_openFEMA.py           # Downloads NFIP datasets from FEMA

└── plot_bivariate_choropleth.py        # Processes data and creates the map

└── static_plots_bivariate_choropleth_top_10_flood_events.py   # Generate Animation of top 10 costliest flood events




---

## 💡 Inspiration 

This work was inspired by [@Marc's](https://www.linkedin.com/in/markebauer/) [deep dive](https://github.com/mebauer/duckdb-fema-nfip) into  NFIP datasets that got me thinking about how coverage and claims vary spatially.
