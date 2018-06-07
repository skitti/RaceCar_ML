import sqlite3
import pandas as pd

def dataset_df():
	con = sqlite3.connect("databazz.db")
	c = con.cursor()

	df = pd.read_csv('DriveKTH_2.csv', parse_dates=True)

	df.to_sql("carrides", con, if_exists='replace', index=False) # can use append instead of replace
	
	variables = """IMU_GPSLongetude, IMU_GPSLatetude, Time,
	IMU_speedSpeed_IMU, Battery_Status_2HV_Power, IMU_AccelerationAcceleration_X,
	IMU_AccelerationAcceleration_Y"""
	
	df = pd.read_sql_query("SELECT {} FROM carrides".format(variables),con)
	con.close()
	return df

def write_db (df_seg):
	#print(df_seg.columns)
	con = sqlite3.connect("databazz.db")
	df_seg.to_sql("carrides_filtered", con, if_exists='replace', index=False)
	#print(df_seg.columns)
	variables = """IMU_GPSLongetude, IMU_GPSLatetude, Time,
	IMU_speedSpeed_IMU, Battery_Status_2HV_Power, IMU_AccelerationAcceleration_X,
	IMU_AccelerationAcceleration_Y"""
	
	df_seg = pd.read_sql_query("SELECT {} FROM carrides".format(variables),con)
	
	#print(df_seg)
	con.close()

def create_table():
    c.execute("CREATE TABLE IF NOT EXISTS stuffToPlot(unix REAL, datestamp TEXT, keyword TEXT, value REAL)")
	
def write_realtime():
	con = sqlite3.connect("databazz.db")
	c = con.cursor()
	c.execute("INSERT INTO carrides VALUES (1231234,'MHAMMAD')")
	con.commit()
	c.close()
	con.close()
	
