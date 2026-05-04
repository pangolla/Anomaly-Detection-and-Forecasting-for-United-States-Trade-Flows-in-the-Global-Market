# Trade Flow Analysis — MSDS696 Practicum II

# Anomaly Detection and Forecasting for United States’ Trade Flows in the Global Market

Globalization has modified supply chains by introducing a high level of interconnections of supply chains across
economies. Consequently, the supply chains of businesses are exposed to uncertainties from different markets; which,
in a competitive global market, exposes businesses to the risks of low profit margins. Predictability of supply chains
offers businesses a competitive advantage. Trade flows provide insight into supply chain functionality at the global
market level and will hence be utilized for the analysis in the project. In order to adequately present a predictability
approach for supply chains, both forecasting and identification of abnormal patterns are important. Consequently,
two Anomaly detection models will first be employed; Isolation Tree and Long Short-Term Memory (LSTM)
Autoencoder. Secondly, two time series forecasting models will be employed; XGBoost and Long Short-Term
Memory (LSTM). Thirdly, Gravity models form the foundation of the analysis into trade flows and hence will be
integrated into the time series forecasting models to create hybrid models for the forecasting. In this project, trade
flows of the United States form indicators for supply chain functionality with anomaly detection and forecasting
of trade flows offering an approach for prediction of supply chain behavior in the global market for American
businesses

## Project Structure

```
Trade_Flow_project/
├── data/
│   ├── raw/                          # Untouched source files (Census + IMF)
│   ├── cleaned/                      # Individual cleaned datasets
│   │   ├── exports_cleaned.csv
│   │   ├── imports_cleaned.csv
│   │   ├── exchange_rate_cleaned.csv  # Includes Euro-area rows
│   │   ├── gdp_cleaned.csv
│   │   ├── inflation_cleaned.csv
│   │   └── interest_rate_cleaned.csv
│   ├── final/
│   │   └── master_trade_flow.csv     # Merged panel (~19.6k × 20 after cleanup + FE; see notebooks)
│   └── country_name_mapping.csv      # Census → IMF name mapping (74 entries)
├── notebooks/
│   ├── Week2_Data_Cleaning.ipynb     # Main ETL from raw → cleaned → first merge
│   ├── Week2_Data_Fix_Inflation.ipynb # Post-merge fixes (inflation, territories, modeling choices)
│   └── Practicum_II_Week_2_Progress.ipynb
│   └── Week3_FE_EDA_IsolationForest.ipynb
│   └── Week4_LSTM_Autoencoder.ipynb
│   └── Week5_XGBoost.ipynb
│   └── Week6_GNN_Neural_Gravity.ipynb
│   └── Week6_ST_GCN.ipynb
│   └── Week7_Cross_Model_Synthesis.ipynb
│   └── Week7_Distance_Fix.ipynb
├── .gitignore
└── README.md
```

## Data 

### Data Sources


Monthly US bilateral trade data merged with IMF macroeconomic indicators for 234 partner countries (2017–2024).


| Dataset | Source | Frequency |
|---------|--------|-----------|
| Exports / Imports | US Census Bureau API | Monthly |
| Exchange Rates | IMF IFS (Domestic currency per USD) | Monthly |
| GDP | IMF WEO | Annual (broadcast to monthly) |
| Inflation (CPI or HICP, YOY) | IMF IFS | Monthly (80 countries in cleaned extract) |
| Interest Rates (Lending) | IMF IFS | Monthly (133 countries) |

### Master Dataset Columns

| Column | Description |
|--------|-------------|
| `country_imf` | Harmonized partner name (IMF-style) |
| `year`, `month` | Time period |
| `exports_usd` | US exports to partner (USD) |
| `imports_usd` | US imports from partner (USD) |
| `exchange_rate` | Partner currency per USD (period average) |
| `gdp_billions` | Partner GDP in billions USD |
| `inflation` | CPI or HICP all-items year-over-year % change |
| `interest_rate` | Not on the modeling master (see fix notebook); `interest_rate_cleaned.csv` kept for documentation |

### Data Gaps (current merged master, approximate)

- **`gdp_billions`:** ~9% missing (annual macro merged to monthly panel)
- **`inflation`:** ~8% missing after WEO-style refresh in the fix notebook (IMF naming / coverage)
- **`exchange_rate`:** ~3% missing; remaining gaps often smaller partners / territories
- **MoM features (`exports_mom_pct`, `imports_mom_pct`):** ~1% missing (edge months)
- **Euro-area:** EUR/USD applied to eurozone partners per cleaning notebook

### Data Quality & Handling Missingness

Pipeline used: 
- Missingness in the merged panel on the order of **~8–9%** for GDP and inflation in the current master is noted and addressed.

- Complete-case analysis for the main multivariate models (Isolation Forest, gradient boosting, LSTM, GNNs) to make inputs fully observed for runs. 

- Linear interpolation is applied only to slow-moving macro series in the data-fix workflow to preserve time continuity.



## Models Implemented 

### Baseline Models

- Isolation Forest (Statistical Baseline Model)

  - Trees: 200 | Contamination: 5% 
  - Features: 6 trade features Per-country z-score normalized 
  
- Long Short-Term Memory (LSTM) Autoencoder (Machine Learning Baseline Model)

  - Layers: LSTM(32) → RepeatVector → LSTM(32) → Dense(4)
  - Sequence: 12 months | Epochs: 50
  - Parameters: ~10,000

### Advanced Machine Learning Models

- Gradient Boosting - XGBoost

  - Trees: 500 | Depth: 6 | Learning rate: 0.05
  - Features: 9 (3 lags + 6 macro) 

- Graphical Neural Networks - GNNs

  - Layers: GCNConv(4,32) → GCNConv(32,32) → MLP 
  - Node features: 4 | Edge features: 5
  - Parameters: ~13,000
  
- Spatially Temporal Graph Convolutional Networks (ST-GCN) 

  - Layers: GCNConv(4,32) → GCNConv(32,32) → LSTM(32) → Linear 
  - Sequence: 6 months 
  - Parameters: ~22,000

## Key Findings

### Where Do Anomalies Cluster

2020 dominates — COVID lockdowns triggered 35 flags in one month. Second spike around Russia-Ukraine war (2022). 
Most months show 0-5 flags — anomalies are rare, which makes the spikes meaningful.

<img width="1239" height="381" alt="image" src="https://github.com/user-attachments/assets/a91f3ea0-f39b-469d-9040-33169d213ba4" />

### Cross Model Agreement

GNN and ST-GCN agree most (24%),  expected since they share similar graph architecture. Isolation Forest and XGBoost overlap at 17%, different methods finding some of the same disruptions. 
Low overall agreement is actually a STRENGTH,  each model catches different types of anomalies, giving broader coverage together.

<img width="1093" height="807" alt="image" src="https://github.com/user-attachments/assets/b8ad9d89-5245-4bc4-acdb-fcdd9d19aa1d" />

### Model Performance Comparison 

All four models cluster near R² = 0.965 — graph models add minimal gain over XGBoost.
XGBoost without lag features: R² = 0.86 — macro features alone carry real signal.
Lag features add 11 percentage points (0.86 → 0.97) — trade is highly autocorrelated.

<img width="1196" height="400" alt="image" src="https://github.com/user-attachments/assets/0c628d95-c977-4571-9c1b-d724becb728b" />



