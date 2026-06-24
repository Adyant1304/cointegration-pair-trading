@echo off
python 01_data_collection.py
python 02_cointegration.py
python 03_signals.py
python 04_backtest.py
git add .
git commit -m "Automated data refresh"
git push
echo Dashboard update complete!
pause