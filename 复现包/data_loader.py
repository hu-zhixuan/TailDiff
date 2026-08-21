"""
Data Loader & Preprocessing Pipeline for Authentic Financial Market Time Series.
Supports CSI 300 and S&P 500 OHLCV data with strictly causal feature extraction.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import os
import sys

# Ensure UTF-8 stdout on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.config import TailDiffConfig


class MarketDataLoader:
    def __init__(self, cfg: TailDiffConfig = None):
        self.cfg = cfg or TailDiffConfig()

    def load_raw_csv(self, filepath: str) -> pd.DataFrame:
        """读取标准 OHLCV 行情 CSV 文件。"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"行情数据文件不存在: {filepath}")
        df = pd.read_csv(filepath)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        df = df.set_index("Date")
        return df

    def create_tail_dataset(self, df: pd.DataFrame, market_name: str = "CSI300") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        构建防未来穿越的极端风险预警数据集：
        - 特征窗口 X: 过去 L=32 天的日对数收益率轨迹 (t-L+1 到 t)
        - 物理条件 RV: 过去 20 天滚动年化已实现波动率 (截至 t)
        - 极端标签 y: 未来 H 天前瞻在险跌幅 (Drawdown at Risk) 或累计跌幅是否突破尾部阈值
        - 决策日期 dates: 样本特征窗口截止日 t
        """
        L = self.cfg.L
        H = self.cfg.horizon
        rv_win = self.cfg.rv_window

        # 1. 计算日对数收益率
        close_prices = df["Close"].astype(float)
        log_ret = np.log(close_prices).diff().dropna()

        # 2. 过去 20 日滚动年化已实现波动率 (以交易日为基准)
        rv20 = (log_ret.rolling(rv_win).std() * np.sqrt(252)).dropna()

        # 3. 前瞻 H 日在险回撤 (Forward Maximum Adverse Drawdown)
        # t 时刻评估 (t, t+H] 区间内的最大跌幅
        fwd_drawdowns = []
        fwd_cum_returns = []
        n_days = len(close_prices)

        for i in range(n_days):
            if i + H < n_days:
                curr_price = close_prices.iloc[i]
                future_prices = close_prices.iloc[i+1 : i+1+H]
                max_dd = (future_prices.min() - curr_price) / curr_price
                cum_ret = np.log(future_prices.iloc[-1] / curr_price)
                fwd_drawdowns.append(max_dd)
                fwd_cum_returns.append(cum_ret)
            else:
                fwd_drawdowns.append(np.nan)
                fwd_cum_returns.append(np.nan)

        df_feat = pd.DataFrame({
            "r": log_ret,
            "rv": rv20,
            "fwd_dd": pd.Series(fwd_drawdowns, index=close_prices.index),
            "fwd_cum": pd.Series(fwd_cum_returns, index=close_prices.index)
        }).dropna()

        # 4. 极端尾部风险打标 (DaR 极端事件)
        if self.cfg.use_quantile_label:
            # 严格按照前瞻跌幅的历史最严重分位数打标 (如前 4.5% 严重事件)
            threshold = float(np.quantile(df_feat["fwd_dd"], self.cfg.crash_quantile))
        else:
            threshold = self.cfg.crash_threshold

        y_all = (df_feat["fwd_dd"] <= threshold).astype(int).values

        # 5. 滑动窗口构造时序轨迹 X: (N, L)
        n_samples = len(df_feat) - L + 1
        returns_windows = np.stack([df_feat["r"].values[i : i + L] for i in range(n_samples)]).astype(np.float32)
        dates = df_feat.index[L - 1 :].to_numpy().astype("datetime64[D]")
        rv_values = df_feat["rv"].values[L - 1 :].astype(np.float32)
        y_labels = y_all[L - 1 :].astype(int)

        meta = {
            "market": market_name,
            "total_samples": len(y_labels),
            "crash_samples": int(np.sum(y_labels)),
            "crash_prevalence": float(np.mean(y_labels)),
            "threshold_used": threshold,
            "start_date": str(dates[0]),
            "end_date": str(dates[-1]),
        }

        print(f"[{market_name}] 数据加载成功: 共 {meta['total_samples']} 条样本, "
              f"暴跌样本: {meta['crash_samples']} ({meta['crash_prevalence']*100:.2f}%), "
              f"时间范围: {meta['start_date']} ~ {meta['end_date']}, 阈值: {threshold:.4f}")

        return returns_windows, dates, y_labels, rv_values, meta


def load_datasets(cfg: TailDiffConfig = None):
    """一键加载沪深300与标普500数据集。"""
    cfg = cfg or TailDiffConfig()
    loader = MarketDataLoader(cfg)
    
    df_csi = loader.load_raw_csv(cfg.csi300_path)
    X_csi, d_csi, y_csi, rv_csi, meta_csi = loader.create_tail_dataset(df_csi, "CSI300")

    df_sp = loader.load_raw_csv(cfg.sp500_path)
    X_sp, d_sp, y_sp, rv_sp, meta_sp = loader.create_tail_dataset(df_sp, "SP500")

    return (X_csi, d_csi, y_csi, rv_csi, meta_csi), (X_sp, d_sp, y_sp, rv_sp, meta_sp)


if __name__ == "__main__":
    cfg = TailDiffConfig()
    (X_csi, d_csi, y_csi, rv_csi, meta_csi), (X_sp, d_sp, y_sp, rv_sp, meta_sp) = load_datasets(cfg)
