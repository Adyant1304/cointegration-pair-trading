Are We A Pair?
Statistical Arbitrage & Cointegration-Based Pairs Trading Engine

"Are We A Pair?" is a systematic quantitative trading framework designed to identify, analyze, and trade cointegrated equity pairs in the Indian market. By moving beyond simple correlation, this project utilizes the Engle-Granger cointegration framework to model long-term price equilibrium and exploit mean-reversion opportunities.

📊 Project Overview
The core objective is to identify pairs of assets that move together in the long run. When the spread between these assets deviates significantly from its historical mean, the engine identifies a potential mean-reversion trade.

Key Methodology
Stationarity Testing: Utilizes the Augmented Dickey-Fuller (ADF) test to ensure the stability of time series data.

Cointegration Modeling: Implements the Engle-Granger two-step method to verify if a linear combination of two assets is stationary.

Signal Generation: Computes rolling Z-scores of the price spread to identify entry and exit signals.

Backtesting: Simulates trade execution and evaluates performance using metrics such as annualized returns, Sharpe ratio, and maximum drawdown.

🛠 Tech Stack
Language: Python 3.14

Data Handling: pandas, numpy, yfinance

Statistical Analysis: statsmodels

Frontend/Dashboard: streamlit, plotly

Deployment: Streamlit Cloud

🚀 Dashboard Features
The integrated dashboard provides real-time market intelligence:

Live Radar: Monitoring of open positions and active signals.

Backtest Summary: Performance analytics of the strategy across the selected universe.

Pair Explorer: Interactive visualization of the spread, allowing for deep-dive analysis into specific cointegrated pairs.

📂 Project Structure
Plaintext
├── dashboard.py             # Main Streamlit application
├── requirements.txt         # Project dependencies
├── pairs_trading_data/      # Data pipeline (CSV/JSON outputs)
│   ├── signals.csv          # Generated trading signals
│   ├── open_positions.json  # Current trade status
│   └── backtest_summary.csv # Strategy performance metrics
└── 01_data_collection.py    # Pipeline scripts (Phase 1-4)
📈 Getting Started
To run the dashboard locally:

Clone the repository:

Bash
git clone https://github.com/Adyant1304/cointegration-pair-trading.git
cd cointegration-pair-trading
Install dependencies:

Bash
pip install -r requirements.txt
Launch the dashboard:

Bash
python -m streamlit run dashboard.py
🤝 Contact & Discussion
I am actively refining the signal generation pipeline and welcome any discussions regarding statistical arbitrage, cointegration methodologies, or model scalability.

LinkedIn: https://www.linkedin.com/in/adyant-vasudeva-404602258/

GitHub: https://github.com/Adyant1304/cointegration-pair-trading

Pro-Tip:
Once you save this file, don't forget to push it to your GitHub:

PowerShell
git add README.md
git commit -m "Add professional README"
git push
