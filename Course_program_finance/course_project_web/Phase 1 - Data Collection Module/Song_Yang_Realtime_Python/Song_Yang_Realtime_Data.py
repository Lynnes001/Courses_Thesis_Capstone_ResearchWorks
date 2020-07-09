
import datetime
import requests
import json
import sys

def getRealtimeStock(symbol):
    url=getURL(symbol)
    resp = requests.get(url)

    #parse the section from the html document containing the raw json data that we need
    #you can write jsonstr to a file, then open the file in a web browser to browse the structure of the json data
    r=resp.text.encode('utf-8')
    i1=0
    i1=r.find(b'root.App.main', i1)
    i1=r.find(b'{', i1)
    i2=r.find(b"\n", i1)
    i2=r.rfind(b';', i1, i2)
    jsonstr=r[i1:i2]

    currentdatetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    #load the raw json data into a python data object
    data = json.loads(jsonstr)

    #pull the values that we are interested in
    name=data['context']['dispatcher']['stores']['QuoteSummaryStore']['price']['shortName']
    price=data['context']['dispatcher']['stores']['QuoteSummaryStore']['price']['regularMarketPrice']['raw']
    volume=data['context']['dispatcher']['stores']['QuoteSummaryStore']['price']['regularMarketVolume']['raw']
    change=data['context']['dispatcher']['stores']['QuoteSummaryStore']['price']['regularMarketChange']['raw']

    #print the values
    #print ('Symbol:', symbol)
    #print 'Name:', name
    #print 'Price:', price
    #print 'Volume:', volume
    #print 'Change:', change


    res = []
    res.append(symbol)
    res.append(float(price))
    res.append(currentdatetime)
    res.append(int(volume))
    #print res
    return res

def getURL(stockCode):
    return "https://finance.yahoo.com/quote/"+ stockCode

if __name__ == "__main__":
    stock_info = getRealtimeStock(sys.argv[1])
    print (stock_info[1])
