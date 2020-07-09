from alpha_vantage.timeseries import TimeSeries
from pprint import pprint
import sys

for stockSymbol in sys.argv[1:]:
    ts = TimeSeries(key='0345Y49CIQ03XJK3', output_format='json')  # replace YOUR_API_KEY with your key
    data, meta_data = ts.get_daily(symbol=stockSymbol, outputsize='full')
    print(data)