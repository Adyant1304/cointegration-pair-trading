"""
Pairs Trading — Phase 2: Cointegration Testing (Real-Time Edition)
===================================================================
Reads cleaned prices from Phase 1, runs Engle-Granger cointegration
tests on all pairs, and outputs a ranked list of cointegrated pairs.

Also appends today's real-time price to the formation dataset before
testing, so the hedge ratios reflect the most current relationship.

Requirements:
    pip install pandas numpy statsmodels scipy tqdm
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from itertools import combinations
from tqdm import tqdm
import os
import json
import warnings
warnings.filterwarnings("ignore")

INPUT_DIR  = "pairs_trading_data"
OUTPUT_DIR = "pairs_trading_data"


# ── 1. Load data ──────────────────────────────────────────────────────────────

def load_prices():
    path   = os.path.join(INPUT_DIR, "prices.csv")
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"Loaded prices: {prices.shape[0]} days × {prices.shape[1]} stocks")
    return prices


def append_realtime_row(prices):
    """
    If a real-time snapshot exists (from Phase 1), append today's prices
    as the most recent row so cointegration uses the latest data point.
    """
    snap_path = os.path.join(INPUT_DIR, "realtime_snapshot.json")
    if not os.path.exists(snap_path):
        print("No real-time snapshot found — using historical data only.")
        return prices

    with open(snap_path) as f:
        snap = json.load(f)

    rt_prices = snap.get("prices", {})
    today     = pd.Timestamp(snap["fetched_at"][:10])

    if today in prices.index:
        print(f"Today ({today.date()}) already in dataset — skipping append.")
        return prices

    row = {}
    for ticker in prices.columns:
        if rt_prices.get(ticker):
            row[ticker] = rt_prices[ticker]["price"]

    if row:
        new_row   = pd.DataFrame([row], index=[today])
        prices    = pd.concat([prices, new_row]).sort_index()
        print(f"Appended real-time row for {today.date()} ({len(row)} tickers)")

    return prices


# ── 2. Formation / trading split ──────────────────────────────────────────────

def split_windows(prices, formation_years=3):
    cutoff   = prices.index[int(len(prices) * (formation_years / 5))]
    formation = prices[prices.index <= cutoff]
    trading   = prices[prices.index >  cutoff]
    print(f"\nFormation: {formation.index[0].date()} → {formation.index[-1].date()} ({len(formation)} days)")
    print(f"Trading  : {trading.index[0].date()} → {trading.index[-1].date()} ({len(trading)} days)")
    return formation, trading


# ── 3. Cointegration test ─────────────────────────────────────────────────────

def test_cointegration(s1, s2, significance=0.05):
    score, p_value, _ = coint(s1, s2)
    if p_value >= significance:
        return None

    X           = add_constant(s2)
    result      = OLS(s1, X).fit()
    hedge_ratio = result.params.iloc[1]
    alpha       = result.params.iloc[0]
    spread      = s1 - hedge_ratio * s2 - alpha

    adf_stat    = adfuller(spread, autolag="AIC")[0]

    spread_lag  = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    lam         = OLS(spread_diff, add_constant(spread_lag)).fit().params.iloc[1]
    half_life   = -np.log(2) / np.log(1 + lam) if lam < 0 else np.nan

    return {
        "p_value"    : round(p_value, 5),
        "hedge_ratio": round(hedge_ratio, 4),
        "alpha"      : round(alpha, 4),
        "adf_stat"   : round(adf_stat, 4),
        "half_life"  : round(half_life, 1) if not np.isnan(half_life) else np.nan,
    }


# ── 4. Test all pairs ─────────────────────────────────────────────────────────

def find_cointegrated_pairs(formation, significance=0.05, max_half_life=126, min_half_life=5):
    tickers = formation.columns.tolist()
    pairs   = list(combinations(tickers, 2))
    print(f"\nTesting {len(pairs):,} pairs (p < {significance}, HL: {min_half_life}–{max_half_life}d)...")

    results = []
    for t1, t2 in tqdm(pairs, desc="Testing"):
        s1     = formation[t1].dropna()
        s2     = formation[t2].dropna()
        common = s1.index.intersection(s2.index)
        if len(common) < 252:
            continue
        s1, s2 = s1[common], s2[common]
        r = test_cointegration(s1, s2, significance)
        if r is None:
            continue
        hl = r["half_life"]
        if np.isnan(hl) or not (min_half_life <= hl <= max_half_life):
            continue
        if r["hedge_ratio"] <= 0:
            continue
        results.append({"stock_1": t1, "stock_2": t2, **r})

    if not results:
        print("No cointegrated pairs found. Try relaxing thresholds.")
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values(["p_value", "half_life"]).reset_index(drop=True)
    return df


# ── 5. Results & save ─────────────────────────────────────────────────────────

def summarise(df):
    if df.empty:
        return
    print(f"\n{'─'*60}")
    print(f"  Cointegrated pairs : {len(df)}")
    print(f"  Avg p-value        : {df['p_value'].mean():.4f}")
    print(f"  Avg half-life      : {df['half_life'].mean():.1f} days")
    print(f"{'─'*60}\n")
    top = df.head(15)[["stock_1","stock_2","p_value","hedge_ratio","half_life"]]
    top.index = range(1, len(top)+1)
    print(f"  {'#':<4} {'Pair':<35} {'p-value':<10} {'Hedge ratio':<14} {'Half-life (days)'}")
    print(f"  {'─'*4} {'─'*35} {'─'*10} {'─'*14} {'─'*16}")
    for i, row in top.iterrows():
        pair = f"{row['stock_1']} / {row['stock_2']}"
        print(f"  {i:<4} {pair:<35} {row['p_value']:<10.5f} {row['hedge_ratio']:<14.4f} {row['half_life']:.1f}")


def save_results(df, formation, trading):
    if df.empty:
        return
    df.to_csv(os.path.join(OUTPUT_DIR, "cointegrated_pairs.csv"), index=False)
    formation.to_csv(os.path.join(OUTPUT_DIR, "formation_prices.csv"))
    trading.to_csv(os.path.join(OUTPUT_DIR, "trading_prices.csv"))
    print(f"Saved cointegrated_pairs.csv, formation_prices.csv, trading_prices.csv")


def plot_best_spread(df, formation):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        return

    if df.empty:
        return

    best  = df.iloc[0]
    t1, t2 = best["stock_1"], best["stock_2"]
    common = formation[t1].index.intersection(formation[t2].index)
    spread = formation[t1][common] - best["hedge_ratio"] * formation[t2][common] - best["alpha"]
    z      = (spread - spread.mean()) / spread.std()

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f"Best pair: {t1} / {t2}  (p={best['p_value']:.4f}, HL={best['half_life']:.0f}d)", fontsize=13)

    ax1 = axes[0]
    ax1.plot(formation[t1][common] / formation[t1][common].iloc[0], label=t1, lw=1.2)
    ax1.plot(formation[t2][common] / formation[t2][common].iloc[0], label=t2, lw=1.2, alpha=0.8)
    ax1.set_ylabel("Normalised price"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(z.index, z, color="steelblue", lw=1)
    ax2.axhline(0,    color="black",  lw=0.8, linestyle="--")
    ax2.axhline( 2.0, color="red",    lw=0.8, linestyle="--", label="Entry (±2σ)")
    ax2.axhline(-2.0, color="red",    lw=0.8, linestyle="--")
    ax2.axhline( 3.5, color="orange", lw=0.8, linestyle=":",  label="Stop-loss (±3.5σ)")
    ax2.axhline(-3.5, color="orange", lw=0.8, linestyle=":")
    ax2.fill_between(z.index, -2, 2, alpha=0.07, color="green")
    ax2.set_ylabel("Z-score"); ax2.legend(loc="upper right", fontsize=9); ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "best_pair_spread.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {path}")


# ── 6. Main ───────────────────────────────────────────────────────────────────

def main():
    prices    = load_prices()
    prices    = append_realtime_row(prices)          # inject today's real-time price
    formation, trading = split_windows(prices, formation_years=3)
    pairs_df  = find_cointegrated_pairs(formation, significance=0.05,
                                        max_half_life=126, min_half_life=5)
    summarise(pairs_df)
    save_results(pairs_df, formation, trading)
    plot_best_spread(pairs_df, formation)
    print("\nPhase 2 complete.\n")
    return pairs_df, formation, trading


if __name__ == "__main__":
    pairs_df, formation, trading = main()