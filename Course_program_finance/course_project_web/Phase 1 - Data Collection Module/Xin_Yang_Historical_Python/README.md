# Python Web Crawler for Historical Stock Data (1 year) on Yahoo Finance using BeautifulSoup and Selenium

# Before you run:
  **PLEASE MAKE SURE YOU HAVE ALREADY INSTALLED MySQL and configured properly!**
  * Environment:
    + Python 2.7
    + Google Chrome
    + Jupyter Notebook:
      > pip install jupyter notebook

    + MySQL
    + MacOS(not required)

  * Dependencies:
    + numpy:
      > pip install numpy

    + Beautiful Soup:
      > pip install beautifulsoup4

    + Selenium 3.9
      > pip install -U selenium
        or
      > pip install selenium

    + MySQLdb-Python:
      > pip install MySQLdb-Python

    + chromedriver:
      Download drivers for your browsers: https://pypi.python.org/pypi/selenium/

--------------------------------------------------------------------------------

# How to customize and run:
  * Set path to your browser driver:
    - See the definition of function driveChrome.
    - Change "webdriver.Chrome('./chromedriver')" to your browser name and driver path.

  * Configure MySQL:
    - See function "dbInsertHistoryTime(list)"
    - Change the "passwd" into your own root password.

  * Get data of your stock:
    - Read the 'Examples' section in the code.
    - Get the stock code of your stock, e.g., Google: GOOG, Yahoo: AABA, etc.
    - Say you want to get the company example's stock, whose stock code is 'CODE':
      > example = getHistoricalData("CODE")
      or
      > getHistoricalData("CODE")

    - Retrieve data:
      > example: data of all year
        example[x]: all data of that day
        example[x][0]: date (String)
        example[x][1]: open price (float)
        example[x][2]: high price (float)
        example[x][3]: low price (float)
        example[x][4]: close price* (float)
        example[x][5]: adjusted close price** (float)
        example[x][6]: volume (int)
    **For Details Please Read the Code**

_Created and tested on Feb.28.2018 by Xin Yang (xy213)
