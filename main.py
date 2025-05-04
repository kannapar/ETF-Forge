import warnings
warnings.filterwarnings("ignore")

from ETFSelector import DiversifiedETFSelector

# Inputs
existing = ['SPY', 'BND']
candidates = [
    "VEA", "VWO", "IEFA", "IEMG", "EWJ", "EFA",     # International
    "VNQ", "SCHH", "IYR",                           # Real estate
    "GLD", "SLV", "DBC", "USO",                     # Commodities
    "ARKK", "QYLD", "VT"                            # Alternatives
]
start_date = "2015-01-01"
risk_free_rate = 0.04

# Run Program with additional config
selector = DiversifiedETFSelector(existing, candidates,start= start_date,risk_free_rate=risk_free_rate)
suggested = selector.suggest_etfs(top_n=5, num_clusters=3, 
                     apply_vol_filter=True, threshold_percentile=90, 
                     apply_sharpe_filter=True, window=90, min_sharpe=0.2, 
                     apply_drawdown_filter= True ,max_dd_threshold=0.4,
                     show_plots=False)

# Output
print("Suggested diversified ETFs:", suggested)