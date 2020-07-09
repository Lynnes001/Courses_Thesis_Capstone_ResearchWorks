
# coding: utf-8

# In[26]:


import sys 
sys.path.append('Xin_Yang_Historical_Python')
sys.path.append('Song_Yang_Realtime_Python')
sys.path.append('SQL')

import time
import thread
import os

from Xin_Yang_Historical_Data import getHistoricalData, getURL
from Song_Yang_Realtime_Data import getRealtimeStock
from mysql_functions import dbInsertRealTime, dbInsertHistoryTime


# In[27]:


def RealtimeStock(StockName):
    print 'Thread Starting'
    while 1:
        realtimeInfo = getRealtimeStock(StockName)
        print realtimeInfo
        dbInsertRealTime(realtimeInfo)
        time.sleep(60)
    
def main(argv1, argv2):
    runtime = argv2
    stockname = argv1
    
    thread.start_new_thread(RealtimeStock, (stockname,))

    time.sleep(float(runtime))
    thread.exit_thread()
    print 'Thread Stopped'

# realtime 60s
if __name__ == '__main__':
    main(str(sys.argv[1]), str(sys.argv[2]))