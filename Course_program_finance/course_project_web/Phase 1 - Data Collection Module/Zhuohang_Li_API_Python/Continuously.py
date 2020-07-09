from alpha_vantage.timeseries import TimeSeries
import time
import sys
import numpy as np

stockSymbol=sys.argv[1]

while(True):
    ts = TimeSeries(key='0345Y49CIQ03XJK3', output_format='pandas') #replace YOUR_API_KEY with your key
    data, meta_data = ts.get_intraday(symbol=stockSymbol, interval='1min', outputsize='full')

    # convert pandas dataframe to ndarray, then ndarray to list
    a=list(np.array(np.array(data.tail(1).index)))
    b=np.array(data.tail(1)).tolist()
    c=[b[0][0],b[0][1],b[0][2],b[0][3],b[0][4]]
    a.extend(c)

    # a is of type list, columns are: timestamp open high low close volume
    print(a)
    time.sleep(60)
