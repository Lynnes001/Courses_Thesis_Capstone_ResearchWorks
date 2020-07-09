# Realtime Stock Data Getter 
This program is in python, getting information on Yahoo Finance using Http Request.

### How to run in bash
1. Find RealtimeRun.py in the parent folder and enter the folder
2. Run is bash or cmd `python RealtimeRun.py <StockSymbol> <runtime(min)>`
    - Example: the following command will fetch AMZN realtime stock price for 10 minutes.
      > python RealtimeRun.py AMZN 10
      
    *RealtimeRun.py is just a service container, the core code is actually in Song_Yang_Realtime_Python/Song_Yang_Realtime_Data.py

### How to run in jupyter
1. Find and open Song_Yang_Historical.ipynb in this folder with jupyter
2. Simply run each steps.
