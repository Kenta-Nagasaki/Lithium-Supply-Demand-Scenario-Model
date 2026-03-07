# Lithium Supply–Demand Scenario Model

![Lithium Supply-Demand Scenario Model](lithium_supply_demand.png)

## Overview

This project is a Python-based scenario model that simulates the lithium supply–demand balance from 2020 to 2030.  
Based on IEA and USGS outlook data, the model assumes fixed growth rates for EV sales and lithium supply to analyze their impact on the overall lithium market.

- Unit: LCE (Lithium Carbonate Equivalent)  
- Historical data are used through 2024 
(USGS supply data are actual through 2023 and estimated for 2024).
- From 2025 onward, projections are based on constant growth rate assumptions  

---

## Motivation

Forecasting lithium prices directly is highly uncertain.  
Instead of attempting price prediction, this model focuses on the structural relationship between demand and supply.  

By explicitly modeling assumptions, it aims to support more consistent and disciplined investment decision-making.

---

## Model Structure

- **EV Demand (kt LCE)**  
  = EV sales × LCE usage per vehicle  

- **Total Lithium Demand**  
  = EV demand × (1 / EV share of total lithium consumption)  

- **Net Lithium Demand**  
  = Total demand × (1 − Recycling rate)  

- **Total Lithium Supply**  
  = Previous year’s supply × (1 + Supply growth rate)

---

## Scenario Assumptions

- EV annual growth rate: 12%  
- Supply annual growth rate: 9%

---

## Key Parameters

- LCE usage per EV: 48 kg  
- Recycling rate: 5%  
- Analysis period: 2020–2030  

---

## How to Run

1. Arrange the files as follows:
├── Lithium_model.py
└── data/
    ├── ev_sales_iea.csv
    └── supply_usgs.csv

2. Run the script:python Lithium_model.py

3. A supply–demand projection chart for 2020–2030 will be generated.

---
## Data Sources

- **EV sales data**: IEA (International Energy Agency), *Global EV Outlook 2025* — World EV sales figures.
- **Lithium supply data**: USGS (U.S. Geological Survey), *Mineral Commodity Summaries 2022~2025* — Lithium production, converted to LCE.

---

## Requirements

- Python 3.9+
- pandas
- numpy
- matplotlib

---

## Author


Kenta Nagasaki


