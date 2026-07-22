# Process Cost Surrogates

This repository contains support vector regression (SVR) surrogate models for estimating optimized membrane process costs in three separation cases:

* CO2 capture from cement calciner flue gas
* CO2 capture from refinery fired-heater flue gas
* Biogas upgrading

## Data

The input variables are log10-scale CO2 permeance and selectivity values. `optimized_cost` is the minimum process cost obtained from process optimization at each membrane-performance grid point.

* CO2 capture: cost in USD2024/tCO2
* Biogas upgrading: cost in USD2024/tCH4
