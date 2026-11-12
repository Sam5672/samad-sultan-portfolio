"""Pulls macroeconomic and yield-curve series from FRED."""
import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from fredapi import Fred

fred = Fred(api_key=os.environ['FRED_API_KEY'])
# Economic regime indicators
gdp_growth = fred.get_series('A191RL1Q225SBEA')       # Real GDP growth (quarterly)
cpi = fred.get_series('CPIAUCSL')                      # CPI (monthly)
core_pce = fred.get_series('PCEPILFE')                 # Core PCE - the Fed's preferred measure
unemployment = fred.get_series('UNRATE')                # Unemployment rate
nonfarm_payrolls = fred.get_series('PAYEMS')            # Nonfarm payrolls

# Yield curve
ten_year = fred.get_series('DGS10')                    # 10Y Treasury yield
two_year = fred.get_series('DGS2')                     # 2Y Treasury yield
three_month = fred.get_series('DGS3MO')                # 3M T-bill
thirty_year = fred.get_series('DGS30')                 # 30Y Treasury yield
fed_funds = fred.get_series('DFF')                     # Effective fed funds rate
twoy_ff_spread = two_year - fed_funds

# Spread (you can compute this, but FRED also has it)
ten_two_spread = fred.get_series('T10Y2Y')             # 10Y-2Y spread

# Inflation expectations (critical for regime classification)
breakeven_10y = fred.get_series('T10YIE')              # 10Y breakeven inflation
umich_inflation_exp = fred.get_series('MICH')          # UMich inflation expectations

# Financial conditions
chicago_fed_nfci = fred.get_series('NFCI')             # Financial conditions index