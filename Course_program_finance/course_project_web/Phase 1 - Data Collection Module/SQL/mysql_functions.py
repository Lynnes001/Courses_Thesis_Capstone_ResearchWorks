import MySQLdb
def dbInsertStock(sy):
	#database connection
	db = MySQLdb.connect("127.0.0.1", "kora", "1234", "stockDB")
	cursor = db.cursor()
	#define sql
	sql = """INSERT INTO stock(symbol) VALUES('"""+sy+"""')"""
	try:
		cursor.execute(sql)
		db.commit()
	except:
		db.rollback()
	db.close()

def dbInsertRealTime(list):
	'''
	list = [symbol, price, stock_time, volume]
	'''
	#database connection
	db = MySQLdb.connect("127.0.0.1", "root", "ys0101", "stockDB")
	cursor = db.cursor()
	#define sql
	sql = """INSERT INTO Real_Time_Data(symbol, price, stock_time, volume) VALUES('"""+list[0]+"""', """+ str(list[1]) + """, '"""+list[2]+ """', """ + str(list[3])+ """)"""
	try:
		cursor.execute(sql)
		db.commit()
	except:
		db.rollback()
	db.close()

def dbInsertHistoryTime(list):
	db = MySQLdb.connect("127.0.0.1", "root", "passwd", "stockDB")
	cursor = db.cursor()
	sql = "INSERT INTO History_Time_Data(symbol, stock_date, open_price, high_price, low_price, close_price, adj_close, volume) VALUES(" + "'"+list[0] + "'," + "'" + list[1] + "'" + "," + list[2] + "," + list[3] + "," + list[4] + "," + list[5] + "," + list[6] + "," + list[7] + ")" + ";"
	try:
		cursor.execute(sql)
		db.commit()
	except:
		db.rollback()
		print("Insert historical data: " + list + " Failed!")
	db.close()
