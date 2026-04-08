import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
from scipy.stats import norm

style.use('seaborn-v0_8')

ticker = 'AAPL'  # Apple Inc.
data = pd.DataFrame()

print("Descargando datos...")
data = yf.download(ticker, start='2012-01-01')
print(data.columns)
data = data[['Close']]
print("Datos descargados")

log_returns = np.log(1 + data['Close'].pct_change())

u = log_returns.mean()
var = log_returns.var()
drift = u - (0.5 * var)
stdev = log_returns.std()

days = 100
trials = 10000
Z = norm.ppf(np.random.rand(days, trials))
retornos_diarios = np.exp(drift.values + stdev.values * Z)
camino_de_precios = np.zeros_like(retornos_diarios)
camino_de_precios[0] = data['Close'].iloc[-1]

for i in range(1, days):
    camino_de_precios[i] = camino_de_precios[i-1] * retornos_diarios[i-1]

plt.figure(figsize=(12,6))
plt.plot(camino_de_precios)
plt.xlabel('Número de días')
plt.ylabel('Precio de ' + ticker)

plt.figure(figsize=(10,5))
sns.histplot(camino_de_precios[-1], bins=50)
plt.xlabel('Precio a ' + str(days) + ' días')
plt.ylabel('Frecuencia')

plt.show()