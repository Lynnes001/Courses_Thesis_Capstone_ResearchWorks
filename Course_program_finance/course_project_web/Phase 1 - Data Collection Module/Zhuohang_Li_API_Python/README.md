# Stock Data collector using alpha_vantage API

# Before you run:
  **PLEASE MAKE SURE YOU HAVE ALREADY INSTALLED 'pip' for Python!**
  * Environment:
    + Python 3.6

  * Dependencies:

    + request 
      > pip install alpha_vantage
      > pip install numpy
      > pip install alpha_vantage
      > pip install pprint
--------------------------------------------------------------------------------

# How to run:
  * RealTime.py: 
    *to collect data up to the time you run this program and save to .csv files
    *to run the code, pass multiply stock symbols by command-line arguments
     , for example:
	> python RealTime.py GOOG AABA CSCO

  * Historical.py: 
    *to collect historical data and save to .csv files
    *to run the code, pass multiply stock symbols by command-line arguments
     , for example:
	> python Historical.py GOOG AABA CSCO

  * Continuously.py: 
    *run continuously to collect realtime data of a single stock
    *to run the code, pass stock symbols by command-line arguments
     , for example:
	> python Continuously.py GOOG

# NOTICE THAT the codes are all well-tested, but there is still a small chance that the code will not perform as expected due to the fact that the API server may not be able to respond to frequent pull request. Please reduce the number of stock symbols in the argument and retry if this happens, thank you for your patience. 

    **For Details Please Read the Code**

_Created and tested on Mar.1.2018 by Zhuohang Li (zl299)
