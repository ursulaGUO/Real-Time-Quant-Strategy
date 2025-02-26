# Overview

# Steps of usage
1. Run `stockdata.py`, which will write data to `US_Stock_Data` folder and `finance` folder. Then optionally check the result using `train.ipynb`. 
2. Run `model.py`, which will create the several different ML models and store them as pickle files, such as `xgboost_stock_model.pkl`, etc.
3. Run `TCP_server.py` to stream real-time data from `finance` folder using `python3 TCP_server.py -p 8080 -f finance/finance.csv -t 0.01`.
4. Run `trade.py` to see the result of trades. It also saves portfolio trade history data into `portfolios` folder. 
5. Finally, run `portfolioMetrics.py` to see the portfolio metrics comparisons. 
