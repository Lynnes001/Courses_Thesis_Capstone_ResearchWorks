#YOU SHOULD LOGI TO MYSQL AS root USER FIRST

#create a database
CREATE DATABASE stockDB;

#change database to stockDB
USE stockDB

#create tables...

CREATE TABLE Real_Time_Data(
	rid INT UNSIGNED AUTO_INCREMENT NOT NULL,
	symbol VARCHAR(4) NOT NULL,
	stock_time DATETIME,
	price FLOAT,
	volume INT,
	PRIMARY KEY(rid)
	);

CREATE TABLE History_Time_Data(
	hid INT UNSIGNED AUTO_INCREMENT NOT NULL,
	symbol VARCHAR(4) NOT NULL,
	stock_date DATE,
	open_price FLOAT,
	high_price FLOAT,
	low_price FLOAT,
	close_price FLOAT,
	adj_close FLOAT,
	volume INT,
	PRIMARY KEY(hid)
	);
