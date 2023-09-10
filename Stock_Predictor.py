import yfinance
from complex_mathematics.ml import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

ticker = yfinance.Ticker(input("Ticker:\n"))

y = np.array(ticker.history("6mo")["High"])

step = 180/y.shape[0]

X = np.arange(step, 180 + step, step).reshape(-1, 1)

model = LinearRegression(optimization_method="NormalEquations")

model.fit(X, y)

prediction = model.predict(180 + int(input("In how many days:\n")))

print("Predicted price:  ", prediction)

xpoints = np.arange(0, 180)

ypoints = model.params[0] * xpoints + model.bias

plt.plot(X, y, "o")

plt.plot(xpoints, ypoints)

plt.show()