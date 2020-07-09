import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd
import sys
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime

#--------------collect stock data-------------
stockSymbol=sys.argv[1]
ts = TimeSeries(key='0345Y49CIQ03XJK3', output_format='pandas')
data, meta_data = ts.get_daily(symbol=stockSymbol, outputsize='full')
StockDate=data.tail(50)

#---------------data processing---------------
stockPrice=np.array(StockDate['4. close'])
timeStamp=list(StockDate.index)

# print(stockPrice)
# print(timeStamp)

date = [datetime.strptime(x,'%Y-%m-%d') for x in timeStamp]
# print(date)

# format=[[(x-date[0]).days] for x in date]
format=[[i] for i in range(len(date))]
# print(format)

predict=[[i] for i in range(len(date)+1)]
# print(predict)

#-----------Support Vector Regression---------
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
# price_rbf = svr_rbf.fit(format, stockPrice).predict(format)
svr_rbf.fit(format,stockPrice)
svr_rbf_predict=svr_rbf.predict(predict)

# print("Predict result is :")
# print(stockSymbol," price on next market day will be: ",svr_rbf_predict[50])
print(svr_rbf_predict[50])


#--------------------draw---------------------
lw = 2
plt.scatter(format, stockPrice, color='darkorange', label='data')
plt.plot(predict, svr_rbf_predict, color='navy', lw=lw, label='RBF model')
plt.xlabel('date')
plt.ylabel('price')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
plt.show()