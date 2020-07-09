# README for Stock Price Prediction with LSMT RNN
*Xin Yang
*xy213

*Please keep the csv file in the ./data directory naming in this way: [STOCK_CODE]_history.csv.*
*Please change the value of variable 'close_column'
  into the index of the column number of 'close' in the csv file AFTER stripping the date column.*
*To predict the price of a stock, call the command:*
> python RNN.py [STOCK_CODE]
e.g.
> python RNN.py GOOG

* Dependencies
  - Python 2.7
  - pandas
  - numpy
  - matplotlib
  - tensorflow with GPU for Linux with Python 2.7

 *Tested on Ubuntu 17.10 with Python 2.7 and TensorFlow with GPU support.
