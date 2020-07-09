
# coding: utf-8

# In[154]:


import sys
sys.path.append('../SQL')
from urllib import urlopen
from bs4 import BeautifulSoup
from selenium import webdriver
import numpy as np
import time
import datetime
import MySQLdb


# In[155]:


# call selenium and chromedriver to get driver
def driveChrome(url):
    driver = webdriver.Chrome('./chromedriver') # CHANGE THE DRIVER TYPE and PATH HERE!!

    # mimic mouse operation to scroll down and show the full page
    driver.get(url)
    driver.execute_script("window.scrollBy(0,3000)")
    time.sleep(0.1)
    driver.execute_script("window.scrollBy(0,5000)")
    time.sleep(0.1)
    driver.execute_script("window.scrollBy(0,5000)")
    time.sleep(0.1)
    driver.execute_script("window.scrollBy(0,5000)")
    time.sleep(0.1)
    driver.execute_script("window.scrollBy(0,5000)")
    time.sleep(0.1)
    
    return driver


# In[156]:


# resolve html to soup
def getSoup(url, driver):
    response = urlopen(url)
    html = response.read()
    soup = BeautifulSoup(driver.page_source, "lxml")
    return soup;


# In[157]:


# transfer month into number
def transMon(day):
    monf = day.text.split(" "); # monf[0] - Jan - Dec    
    #Jan 04, 2018
    mon = "00"
    if monf[0] == "Jan":
        mon = "01"
    elif monf[0] == "Feb":
        mon = "02"
    elif monf[0] == "Mar":
        mon = "03"            
    elif monf[0] == "Apr":
        mon = "04"            
    elif monf[0] == "May":
        mon = "05"            
    elif monf[0] == "Jun":
        mon = "06"            
    elif monf[0] == "Jul":
        mon = "07"            
    elif monf[0] == "Aug":
        mon = "08"
    elif monf[0] == "Sep":
        mon = "09"        
    elif monf[0] == "Oct":
        mon = "10"
    elif monf[0] == "Nov":
        mon = "11"            
    elif monf[0] == "Dec":
        mon = "12"     
        
    dayf = monf[1].strip(',')
    resdate = monf[2]+"-"+mon+"-"+dayf
    return resdate  


# In[158]:


def getHistoricalData(stockCode):
    url = getURL(stockCode)
    driver  = driveChrome(url)
    soup = getSoup(url,driver)
    # quit selenium driver
    driver.quit()
    
    # get dates
    days = soup.find_all("td", class_ = "Py(10px) Ta(start) Pend(10px)") # <td class="Py(10px) Ta(start) Pend(10px)"><span>Feb 28, 2018</span></td>
    dates = []        
    for day in days:
        day = transMon(day)
        #print(day)
        dates.append(day)
    
    
    # delete invalid date
    bad = []
    tr = soup.find_all("tr", class_ = "BdT Bdc($c-fuji-grey-c) Ta(end) Fz(s) Whs(nw)")

    for inner in tr:
        badi = inner.find("td", class_ = "Ta(c) Py(10px) Pstart(10px)")
        if badi:
            #print(bad.text)
            badparent = badi.find_parent()
            baddate = badparent.find("td", class_ = "Py(10px) Ta(start) Pend(10px)")
            bad.append(transMon(baddate))

    for baditem in bad:
        dates.remove(baditem)
    
    # get prices and volume
    elements = soup.find_all("td", class_ = "Py(10px) Pstart(10px)") # <td class="Py(10px) Pstart(10px)"><span>1,123.03</span></td>

    openp = [] # open price
    highp = [] # high price
    lowp = [] # low price
    closep = [] # close price
    adjclosep = [] # adjusted close price
    volume = [] # volume
    
    
    counter = 0
    for element in elements:
        #print(element.text)
        text = element.text.replace("," , "")
        counter+=1;
        if counter % 6 == 1:
            openp.append(float(text))
        elif counter % 6 == 2:
            highp.append(float(text))
        elif counter % 6 == 3:
            lowp.append(float(text))
        elif counter % 6 == 4:
            closep.append(float(text))
        elif counter % 6 == 5:
            adjclosep.append(float(text))
        elif counter % 6 == 0:
            volume.append(int(text))
     
    # generate symbol column
    num = len(volume)
    symbols = []
    for i in range(0,num):
        symbols.append(stockCode)        
    
    print(len(symbols), len(dates), len(openp), len(highp), len(lowp), len(closep), len(adjclosep), len(volume))
    tmp = [symbols,dates, openp, highp, lowp, closep, adjclosep, volume]
    res = np.stack(tmp,axis = 1)
      
    # insert into database
    for record in res:
        dbInsertHistoryTime(record)
    
    return res


# In[159]:


# generate Yahoo Finance url
def getURL(stockCode):
    return "https://finance.yahoo.com/quote/"+ stockCode + "/history?p=" + stockCode


# In[160]:


# insert data into database
def dbInsertHistoryTime(list):
    db = MySQLdb.connect("127.0.0.1", "root", "passwd", "stockDB") # Change your database info here!
    cursor = db.cursor()
    sql = "INSERT INTO History_Time_Data(symbol, stock_date, open_price, high_price, low_price, close_price, adj_close, volume) VALUES(" + "'"+list[0] + "'," + "'" + list[1] + "'" + "," + list[2] + "," + list[3] + "," + list[4] + "," + list[5] + "," + list[6] + "," + list[7] + ")" + ";"
    try:
        cursor.execute(sql)
        db.commit()
    except:
        db.rollback()
        print("Insert historical data Failed!")
    db.close()


# ------------------------------- ** Examples ** --------------------------------------

# In[161]:


"""
Example: example = getHistoricalData("yourStockCode")


example: records of all year
example[x]: record of day x from today

example[x][0]: date (String)
example[x][1]: open price (float)
example[x][2]: high price (float)
example[x][3]: low price (float)
example[x][4]: close price* (float)
example[x][5]: adjusted close price** (float)
example[x][6]: volume (int)

Index from 0 to 251(252 total).
0   is the most recent day(Feb.28.2018)
251 is the least recent day(Mar.01.2017)
Tested Feb.28.2018 by Xin Yang.

*Close price adjusted for splits. **Adjusted close price adjusted for both dividends and splits.
"""


# In[162]:


getHistoricalData("GOOG")    


# In[163]:


getHistoricalData("AABA")


# In[164]:


getHistoricalData("CSCO")


# In[165]:


getHistoricalData("T")


# In[166]:


getHistoricalData("WMT")


# In[167]:


getHistoricalData("NOK")


# In[168]:


getHistoricalData("NFLX")


# In[169]:


getHistoricalData("APA")


# In[170]:


getHistoricalData("NKE")


# In[171]:


getHistoricalData("GE")

