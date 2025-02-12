# Overview

# Steps of usage
1. Run `stockdata.py`, which will write data to `US_Stock_Data` folder and `finance` folder. Then optionally check the result using `train.ipynb`. 
2. Run `model.py`, which will create the XGBoost model and store it as a pickle file as `xgboost_stock_model.pkl`.
3. Run `TCP_server.py` to stream real-time data from `finance` folder.
4. Run `trade.py` to see the result of trades.
