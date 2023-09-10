from yahoo_fin import stock_info as si
import yfinance
from complex_mathematics.ml import LinearRegression
import numpy as np
import pandas as pd

tickers = si.tickers_nasdaq()

rate = {}

n = 0

for ticker in tickers:

  tick = yfinance.Ticker(ticker)

  try:

    y = np.array(tick.history("6mo")["High"])

    step = 180/y.shape[0]
    
    X = np.arange(step, 180 + step, step).reshape(-1, 1)
    
    model = LinearRegression(optimization_method="NormalEquations")
  
    model.fit(X, y)

    rate[ticker] = model.params[0]
  
  except:
    None

  n += 1

  print("Percent Done: ", n*100/len(tickers))

  


rate = dict(sorted(rate.items(), key=lambda item: item[1], reverse=True))

df = pd.DataFrame(rate.items(), columns=["Ticker", "Rate"])

df.to_csv("Stock_Rate_Changes", index=False)