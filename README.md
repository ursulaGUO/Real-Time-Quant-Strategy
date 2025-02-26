# Overview
Our group tried to use XGBoost, Random Forest, Gradient Boosting, and Vector Autoregressor (VAR) to predict the next day stock price. Given that, we execute buy/sell/hold decision. 

# Steps of usage
1. (OPTIONAL) Run `stockdata.py`, which will write data to `US_Stock_Data` folder and `finance` folder. Then optionally check the result using `train.ipynb`. 
   
2. (OPTIONAL) Run `model.py`, which will create the several different ML models and store them as pickle files, such as `xgboost_stock_model.pkl`, etc.
   
3. Run `TCP_server.py` to stream real-time data from `finance` folder using `python3 TCP_server.py -p 8080 -f finance/finance.csv -t 0.01`.
   
4. Run `trade.py` to see the result of trade using `python3 trade.py -model [modelname]`, where the following modelname are accepted. It also saves portfolio trade history data into `portfolios` folder. 
* `xgboost`
* `Random_Forest`
* `Gradient_Boosting`
* `var`

5. Finally, run `portfolioMetrics.py` to see the portfolio metrics comparisons. 

Note: Step 1 to 2 is optional because we have already finished downloading the data and completed running the models. 

