# Financial Mathematics ABM Modeling Project

Brief
-----
Small agent-based market simulator exploring HFT and market-maker (MM) effects on price discovery and volatility.

Quick start
-----------
- Install dependencies:

  ```bash
  pip install mesa==1.2.1 numpy pandas matplotlib scipy statsmodels
  ```

- Run the main notebook `main.ipynb` 

Scenarios
---------
The code defines four main scenarios:

- `no_hft_no_mm`: with_hft=False, with_mm=False
- `hft_only`:     with_hft=True,  with_mm=False
- `hft_mm`:       with_hft=True,  with_mm=True
- `mm_only`:      with_hft=False, with_mm=True
