# 
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from ticker_returns import TickerData

class DiversifiedETFSelector:
    def __init__(self, existing_etfs, candidate_etfs, start="2020-01-01", end=None, risk_free_rate=0.02):
        self.existing = existing_etfs
        self.candidates = candidate_etfs
        self.all_tickers = list(set(existing_etfs + candidate_etfs))
        self.start = start
        self.end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        self.risk_free_rate = risk_free_rate / 252
        self.data = None
        self.returns = None
        self.least_correlated = []
        self.final_selection = []
        self.linkage_matrix = None

    def fetch_data(self):
        t = TickerData(self.all_tickers, self.start, self.end)
        self.data = t.get_prices('daily')
        self.returns = self.data.pct_change().dropna()

    def calculate_volatility(self):
        vol = self.returns[self.candidates].std()
        return vol
    
    def calculate_rolling_sharpe(self, window):
        excess_returns = self.returns.sub(self.risk_free_rate)
        rolling_mean = excess_returns.rolling(window).mean()
        rolling_std = self.returns.rolling(window).std()
        rolling_sharpe = rolling_mean / rolling_std * np.sqrt(252)
        return rolling_sharpe.mean()

    def calculate_max_drawdowns(self):
        max_dd = {}
        for ticker in self.data.columns:
            prices = self.data[ticker].dropna()
            cumulative = (1 + prices.pct_change().fillna(0)).cumprod()
            peak = cumulative.cummax()
            drawdown = (cumulative - peak) / peak
            max_dd[ticker] = drawdown.min()
        return pd.Series(max_dd)

    def filter_by_volatility(self, threshold_percentile):
        vol = self.calculate_volatility()
        cutoff = np.percentile(vol, threshold_percentile)
        self.candidates = vol[vol <= cutoff].index.tolist()

    def filter_by_sharpe(self, window, min_sharpe):
        mean_rolling_sharpe = self.calculate_rolling_sharpe(window)
        self.candidates = mean_rolling_sharpe[mean_rolling_sharpe > min_sharpe].index.intersection(self.candidates).tolist()

    def filter_by_drawdown(self,max_dd_threshold):
        dd = -self.calculate_max_drawdowns()
        self.candidates = dd[dd <= max_dd_threshold].index.intersection(self.candidates).tolist()
    
    def find_least_correlated(self, top_n=5):
        corr_matrix = self.returns.corr()
        candidate_corr = corr_matrix.loc[self.candidates, self.existing].mean(axis=1)
        self.least_correlated = candidate_corr.sort_values().head(top_n).index.tolist()
        return self.least_correlated

    def cluster_and_select(self, num_clusters=3):
        sub_corr = self.returns[self.least_correlated].corr()
        dist = squareform(1 - sub_corr.abs())
        self.linkage_matrix = linkage(dist, method="ward")
        labels = fcluster(self.linkage_matrix, t=num_clusters, criterion="maxclust")

        clustered = pd.DataFrame({"ETF": self.least_correlated, "Cluster": labels})
        self.final_selection = []

        for cluster_id in clustered["Cluster"].unique():
            group = clustered[clustered["Cluster"] == cluster_id]["ETF"].tolist()
            intra_corr = self.returns[group].corr().mean().sort_values()
            self.final_selection.append(intra_corr.index[0])

        return self.final_selection

    def suggest_etfs(self, top_n, num_clusters, 
                     apply_vol_filter, threshold_percentile, 
                     apply_sharpe_filter, window, min_sharpe, 
                     apply_drawdown_filter,max_dd_threshold,
                     show_plots):
        self.fetch_data()
        try:
          if apply_vol_filter:
              self.filter_by_volatility(threshold_percentile)
          if apply_sharpe_filter:
              self.filter_by_sharpe( window, min_sharpe)
          if apply_drawdown_filter:
              self.filter_by_drawdown(max_dd_threshold)

          self.find_least_correlated(top_n=top_n)
          result = self.cluster_and_select(num_clusters=num_clusters)
          
          if show_plots:
              self.plot_correlation_heatmap()
              self.plot_dendrogram()
          return result
        except:
          print("\n \nNo result found. Modify filters \n")  

    def plot_correlation_heatmap(self):
        if not self.least_correlated:
            return
        plt.figure(figsize=(8, 6))
        corr = self.returns[self.least_correlated].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title("Correlation Heatmap of Selected ETFs")
        plt.tight_layout()
        plt.show()

    def plot_dendrogram(self):
        if self.linkage_matrix is None:
            return
        plt.figure(figsize=(8, 5))
        dendrogram(self.linkage_matrix, labels=self.least_correlated)
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("ETFs")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.show()
