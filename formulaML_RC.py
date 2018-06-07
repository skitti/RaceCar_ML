#flowers ML "hello world"
# Load libraries
import time
import pandas as pd
import numpy as np
import sqlite3
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from scipy.interpolate import UnivariateSpline
from math import sin, cos, radians, asin, sqrt 
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
#from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Imputer

from latlon_to_meter import *
#from sqlite import dataset_df, write_db
from optimering import main_optimize
from subdb import *

from pandas.util.testing import assert_frame_equal

start_time = time.clock()
#print(list(range(0,244)))
#exit()
class Datamining:
	
	def __init__(self, file, variables, att_speed):
		#self.columns = variables
		#self.datasheet = pd.read_csv(file)
		#self.df_all = pd.DataFrame(self.datasheet)
		#self.df = pd.DataFrame(self.datasheet,columns=variables)
		#self.df = dataset_df() #om du ska hämta från lokal sql databas
		self.df = main_skapadf()
		#print(self.df.dtypes.index)
		
		#self.df['timestamp'] = self.df['timestamp']-1527000000
	
		#Datamining.scaling(self)
		
		self.df['Battery_Status_2HV_Power'] = (self.df['current_motor'])*self.df['voltage_input']
		self.df['speed'] = self.df['speed']/2000
		self.df = self.df.drop(["duty_cycle","distance_traveled","current_input","current_motor","RCcarNumber","charge_drawn","charge_regen","sequence", "energy_drawn", "energy_regen","voltage_input","temperature_pcb", "displacement"], axis=1)
		
		self.df_power=[]
			
		#Datamining.printallcolumns(self)
				
		Datamining.moving_average(self)
		#Datamining.sample(self)
		#Datamining.scaling(self)
		plt.figure(2)
		plt.scatter(self.df['MA_GPSLAT'], self.df['MA_GPSLONG'], linewidth=0.5)
		
		Datamining.filtering(self)
		#scatter_matrix(self.df[['MA_GPSLAT','MA_GPSLONG']])
		#print(self.df['timestamp'])
		Datamining.segmentation(self)	
		
		pd.options.display.max_columns = 999
		print(self.df_seg)
		Datamining.write_db(self.df_seg)
		
		#Datamining.plots(self)
		
		Datamining.regression(self)
		Datamining.cross_val(self)
		Datamining.lasso(self)
		Datamining.ridge(self)

		Datamining.elastic_net(self)
	
	def printallcolumns(self):
		#Print out all the columns (features)
		#print(self.df.shape[1]) #[0] gives nr of rows [1] gives nr of columns
		for i in range(self.df.shape[1]):
			print(self.df.columns[i])
		
	def sample(self):
		self.df = self.df.iloc[::4, :]

		
	def scaling(self):
		scaler = MinMaxScaler(feature_range=(0,1))
		print(self.df)
		imp=Imputer(missing_values="NaN", strategy="mean",axis=0) #specify axis
		q = imp.fit_transform(self.df)#perform a transpose operation
		scaled_values = scaler.fit_transform(q)
		scaled_values = pandas.DataFrame(scaled_values, columns = self.df.columns)
		self.df = scaled_values
		print(self.df)
		exit()
		
	def moving_average(self):
		#plt.figure(1)
		#plt.plot(self.df['position_x'],self.df['position_y'], color='k')
		#Calculates the moving average of the GPS, velocity, accelerates.
	
		#Appending the short way
		self.df['MA_GPSLAT'] = self.df['position_x'].rolling(window=4).mean()
		self.df['MA_GPSLONG'] = self.df['position_y'].rolling(window=4).mean() #tagit bort [deg]
		#plt.figure(1)
		#plt.plot(self.df['MA_GPSLAT'],self.df['MA_GPSLONG'], color='r')
		#plt.show()
		#exit()
		self.df['MA_GPSLAT_SHIFT'] = self.df.MA_GPSLAT.shift(10)
		self.df['MA_GPSLONG_SHIFT'] = self.df.MA_GPSLONG.shift(10)
		
		#self.df_all['MA_GPSLONG_LAP'] = self.df['position_x'].rolling(window=4).mean()
		#self.df_all['MA_GPSLAT_LAP'] = self.df['position_y[deg]'].rolling(window=4).mean()
		#r=500
		#plt.plot(self.df_all['MA_GPSLONG_LAP'].iloc[1:r],self.df_all['MA_GPSLAT_LAP'].iloc[1:r],'r')
		
		self.df['MA_ACCX'] = self.df['acceleration'].rolling(window=4).mean()
		self.df['MA_VELX'] = self.df['speed'].rolling(window=4).mean()

		self.df['MA_GPSLAT_CHANGE'] = ((self.df.MA_GPSLAT - self.df.MA_GPSLAT_SHIFT))
		self.df['MA_GPSLONG_CHANGE'] = ((self.df.MA_GPSLONG	- self.df.MA_GPSLONG_SHIFT))
		

		self.df['GPS_k'] = self.df['MA_GPSLAT_CHANGE']/self.df['MA_GPSLONG_CHANGE']
		self.df['GPS_k_CHANGE'] = ((self.df.GPS_k - self.df.GPS_k.shift(2)))
		#Calculate distance travel with another script
		self.df['distance_traveled'] = self.df.apply(lambda row: distance_traveled_lol(row), axis=1)
		#pd.options.display.max_rows = 2999
		#print(self.df[['MA_VELX','GPS_k']])
		#self.df['distance_traveled'] = self.df.apply(lambda row: distance_traveled_lol(row), axis=1)		

	def filtering(self):
	#FILTER out all points where the speed is below 2*(SHIFT*0.1)m/s sampletime:0.1s
		#print(self.df[['GPS_k','MA_VELX']])
		#print(self.df['MA_VELX'].mean())
		#print(self.df['GPS_k'].mean())
		#print(0 in self.df.timestamp)
		#print(self.df[['timestamp','speed']])
		self.df = self.df.loc[self.df['MA_VELX'] > 0.01] #200 om ej skalar om speed
		self.df = self.df.loc[(self.df['GPS_k'] > -10) & (self.df['GPS_k'] <10)]
		#self.df = self.df.loc[(self.df['timestamp']==0)]
		#print(self.df[['timestamp','MA_VELX']])
		#print(0 in self.df.timestamp)
		#print("AMANAKAAKSAD", self.df)
		#exit()
		#self.df = self.df.loc[(abs(self.df['GPS_k_CHANGE']) < 0.0000000005)]
		#self.df = self.df.loc[abs(self.df['turning_gyro']) > 0.1]
		#print(abs(self.df['turning_gyro']) > 0.001)

	def segmentation(self):
		def f(x):
			d = {}
			d['Time_seg'] = abs(float((x['timestamp'].max() - x['timestamp'].min()))) ##TAGIT BORT [s]
			#print(self.df['Time_seg'])
			# Distance and velocity calculated from the GPS
			d['Dist_Calc'] = ((x['distance_traveled']).mean() * d['Time_seg'])
			d['Vel_Calc'] = (x['distance_traveled'].mean())
			
			#Start end straight section
			d['x_start']=(x['MA_GPSLAT'].min())
			d['x_end']=(x['MA_GPSLAT'].max())
			d['y_start']=(x['MA_GPSLONG'].min())
			d['y_end']=(x['MA_GPSLONG'].max())
			
			#Distance calculated with first and last GPS coordinate including moving average 
			lat1m, lon1m, lat2m, lon2m = map(radians, [x['MA_GPSLONG'].min(), x['MA_GPSLAT'].min(), x['MA_GPSLONG'].max(), x['MA_GPSLAT'].max()])
			lat1, lon1, lat2, lon2 = map(radians, [x['position_x'].min(), x['position_y'].min(), x['position_x'].max(), x['position_y'].max()])
			#TAGIT BORT [deg]
			d['Dist_GPSMA'] = distance_traveled_lol2(lat1m, lon1m, lat2m, lon2m)
			d['Dist_GPS'] = distance_traveled_lol2(lat1, lon1, lat2, lon2)
			d['speed'] = d['Dist_GPSMA']/d['Time_seg']
			
			#Distance from sensors
			d['Dist_Sensor'] = (x['speed'].mean() * d['Time_seg'])
			d['Dist_SensorMA'] = (x['MA_VELX'].mean() * d['Time_seg'])
			
			#Velocity from sensor, actual value and moving average 
			d['Vel_Sensor'] = (x['speed'].mean())
			d['Vel_SensorMA'] = (x['MA_VELX'].mean())
			
			
			#Velocity from the GPS distance (Shows faster speeds, 
			d['Vel_GPSMA'] = d['Dist_GPSMA']/d['Time_seg']
			d['Vel_GPS'] = d['Dist_GPS']/d['Time_seg']
			
			#Battery spec from sensors
			d['Battery_Status(kW)'] = (x['Battery_Status_2HV_Power']).mean()/1000 #kW
			d['Battery(kJ)'] = (x['Battery_Status_2HV_Power']).mean()*(d['Time_seg'])/1000
			#Tagit bort [W]
			d['Battery(W)'] = (d['Battery(kJ)']/d['Time_seg']) #KW
			
			#Acceleration from sensor with and without MA 
			IMU_Acc_X = x.ix[:, 'acceleration']
			d['Acc_Sensor>0'] = (IMU_Acc_X.loc[IMU_Acc_X>0]).mean()
			IMU_Acc_XMA = x.ix[:, 'MA_ACCX']
			d['Acc_SensorMA>0'] = (IMU_Acc_XMA.loc[IMU_Acc_XMA>0]).mean()
			

			
			#Velocity Difference between max and min
			d['Vel_diff_Sensor'] =(x['speed'].iloc[-1]-(x['speed'].iloc[0]))
			d['Vel_diff_SensorMA'] =(x['MA_VELX'].iloc[-1]-(x['MA_VELX'].iloc[0]))
			d['Vel_diff_Cal'] =(x['distance_traveled'].iloc[-1]-(x['distance_traveled'].iloc[0]))
			
			d['t0_Vel_Sensor'] = x['speed'].iloc[0]    
			d['t0_Vel_SensorMA'] = x['MA_VELX'].iloc[0]
			d['tn_Vel_SensorMA'] = x['MA_VELX'].iloc[-1]  
			d['t0_Vel_Cal'] = x['distance_traveled'].iloc[0]	
			
			
			#return pd.Series(d, index=['Time_seg', 'Dist_GPS','Dist_GPSMA', 'Dist_Calc','Vel_GPS','Vel_GPSMA','Vel_Calc','Vel_Sensor','Battery(W)','Battery(kJ)','Acc_Sensor','Acc_Sensor>0','Vel_diff_Sensor','Vel_diff_Cal','t0_Vel_Sensor','t0_Vel_Cal'])
			#return pd.Series(d, index=['Time_seg','Dist_Sensor','Dist_SensorMA',
			#'Vel_Sensor','Vel_SensorMA','Vel_diff_Sensor','Vel_diff_SensorMA',
			#'t0_Vel_Sensor','t0_Vel_SensorMA','tn_Vel_SensorMA','Battery(W)','Battery(kJ)',
			#'Acc_Sensor>0','Acc_SensorMA>0','Battery_Status(kW)'])

			return pd.Series(d, index=['Time_seg','Dist_SensorMA',
			'Vel_SensorMA','t0_Vel_SensorMA','tn_Vel_SensorMA','Battery(kJ)',
			'Battery_Status(kW)','x_start','x_end','y_start','y_end'])			
			
		#Create a new feature that saves sequential data 
		self.df['splitter_boolean'] = ~self.df['timestamp'].diff().le(1) #EYOOOO [s]
		#print(self.df['splitter_boolean'])
		self.df['segmentation'] = self.df['splitter_boolean'].cumsum()
		#print(self.df['segmentation'])
		
		self.df_seg = self.df.groupby('segmentation')
		self.df_seg = self.df_seg.apply(f)
		
		
		#print(self.df_seg[['t0_Vel_SensorMA','tn_Vel_SensorMA', 'Vel_SensorMA']])
		#self.df_seg[['t0_Vel_SensorMA','tn_Vel_SensorMA', 'Vel_SensorMA']] = self.df_seg[['t0_Vel_SensorMA','tn_Vel_SensorMA', 'Vel_SensorMA']] *0.04* 0.10472
		#print(self.df_seg)
		self.df_seg = self.df_seg.loc[(self.df_seg['Dist_SensorMA'] > 5) & (self.df_seg['Dist_SensorMA'] < 25)]   #Filter out specific lenghts 
		
	def write_db (df_seg):
		con = sqlite3.connect("filtered.db")
		df_seg.to_sql("carrides_filtered", con, if_exists='replace', index=False)
		
		variables = """IMU_GPSLongetude, IMU_GPSLatetude, Time,
		IMU_speedSpeed_IMU, Battery_Status_2HV_Power, IMU_AccelerationAcceleration_X,
		IMU_AccelerationAcceleration_Y"""
		
		#df_seg = pd.read_sql_query("SELECT {} FROM carrides".format(variables),con)
		
		#print(df_seg)
		con.close()
		
	
	def plots(self):
	
		plt.figure(2)

		X=[]
		Y=[]
		for x in range(len(self.df_seg)):
			x1 = self.df_seg.loc[self.df_seg.index[x],'x_start']
			x2 = self.df_seg.loc[self.df_seg.index[x],'x_end']
			y1 = self.df_seg.loc[self.df_seg.index[x],'y_start']
			y2 = self.df_seg.loc[self.df_seg.index[x],'y_end']
			plt.plot([x1,x2], [y1,y2], color='k')
			X.append([x1,x2])
			Y.append([y1,y2])
			
		plt.scatter(X,Y)
		plt.show()
		#exit()

		#plt.plot(self.df_all['MA_GPSLONG_LAP'].iloc[1:r],self.df_all['MA_GPSLAT_LAP'].iloc[1:r],'r')
		#plt.scatter(self.df['MA_GPSLONG'].iloc[1:r],self.df['MA_GPSLAT'].iloc[1:r], s=20, color="k")
		#scatter_matrix(self.df[['Time[s]','GPS_k','MA_GPSLAT','MA_GPSLONG']])
		#plt.figure(2)
		#scatter_matrix(self.df[['MA_GPSLAT','MA_GPSLONG']])
		#scatter_matrix(self.df[['timestamp','speed','MA_GPSLAT','MA_GPSLONG']])		
		#scatter_matrix(self.df[['GPS_k','MA_GPSLAT','MA_GPSLONG','speed']])
		#scatter_matrix(self.df[['position_x','position_y','timestamp','speed']])
		#scatter_matrix(self.df_seg[['Battery(kJ)','Battery(W)','Dist_SensorMA','Vel_SensorMA','Acc_SensorMA>0','Vel_diff_SensorMA']])

		#plt.show()		
		
	def regression(self):
		#print(self.df_seg['tn_Vel_SensorMA'])
		datasheet = self.df_seg ## loads Boston dataset from datasets library
		# define the data/predictors as the pre-set feature names  
		self.indep_var = ['t0_Vel_SensorMA','tn_Vel_SensorMA','Dist_SensorMA','Vel_SensorMA']
		self.predictors=self.indep_var
		#Datamining.add_powers(self)
		#self.indep_var.extend(self.df_power)
		#print(self.indep_var)
		
		#self.indep_var = self.indep_var.extend(self.df_power)
		
		self.pred_var = 'Battery(kJ)'
		self.df_reg = pd.DataFrame(self.df_seg, columns=self.indep_var)
		# Put the target (housing value -- MEDV) in another DataFrame
		target = pd.DataFrame(self.df_seg, columns=[self.pred_var])
		#print(self.df_reg)
		self.X_reg = self.df_reg
		self.y_reg = target[self.pred_var]
		
		X_train, X_test, y_train, y_test = train_test_split(self.df_reg, self.y_reg, test_size=0.25, shuffle=True) #random splits
		#print (X_train.shape, y_train.shape)
		#print (X_test.shape, y_test.shape)
		
		lm = linear_model.LinearRegression()
		model = lm.fit(X_train,y_train)
		predictions = lm.predict(X_test)
		#print(lm.score(X,y)) #accuracy
		rss = sum((y_test-predictions)**2)
		self.regular_regression = [rss, lm.intercept_, *lm.coef_]
		#print("\nPredictor Coeff after LR: ", lm.coef_)			# coefficients of the predictors
		#print("b0 LR: ", lm.intercept_) #b i modellen Y=mX+b
		#print(predictions[0:10])
		score = model.score(X_test, y_test)
		
		 #mindre än CV, Ridge and Lasso tsm amk. För lite data? Nej, vikten
		#print("{} {}".format("RSS LR: ", rss))
		
		print("\nOptimal data using Simple Regression: \n", main_optimize(self.regular_regression,"Simple Regression"))
		print("\nCoefficients used with Simple Regression: \n", self.regular_regression[1:6])
		print("RSS Simple Regression:", rss)
		print ('Score LR:', score)
		
		
		'''
		fig, ax = plt.subplots()
		fit = np.polyfit(y_test, predictions, deg=1)
		ax.plot(y_test, fit[0] * y_test + fit[1], color='red')
		
		plt.scatter(y_test, predictions)

		plt.xlabel('True Values')
		plt.ylabel('Predictions')
		plt.grid()
		plt.show()
		
		return score
		'''
	def cross_val(self):
		kf = KFold(n_splits=3, shuffle = False, random_state = None)
		kf.get_n_splits(self.X_reg)
		#print(kf)
		for train_index, test_index in kf.split(self.X_reg):
			#print('TRAIN:', train_index, '\nTEST:', test_index)		
			
			X_train, X_test = (self.X_reg.values[train_index], self.X_reg.values[test_index])
			y_train, y_test = (self.y_reg.values[train_index], self.y_reg.values[test_index])
			#print(X_train, X_test)
			#print(y_train, y_test)
			
		lm = linear_model.LinearRegression()
		model = lm.fit(X_train,y_train)
		#print(self.indep_var)
		#print("\nPredictor Coeff after CV: ", lm.coef_)
		#print("b0 CV: ", lm.intercept_)
		scores = cross_val_score(model, self.df_reg, self.y_reg, cv=3)
		
		predictions = cross_val_predict(model, self.df_reg, self.y_reg, cv=3)
		accuracy = metrics.r2_score(self.y_reg, predictions)
		#print ('Cross-Predicted Accuracy:', accuracy)
		rss = sum((self.y_reg-predictions)**2)
		#print("{} {}".format("RSS CV: ", rss))
		
		self.cross_fold = [rss, lm.intercept_, *lm.coef_]
		
		print("\nOptimal data using CrossFold: \n", main_optimize(self.cross_fold, "CrossFold"))
		print("\n\nCoefficients used with CrossFold: \n", self.cross_fold[1:6])
		print("RSS CrossFold:", rss)
		print ('Cross-validated score:', scores.mean())
		
		fig, ax = plt.subplots()
		fit = np.polyfit(self.y_reg, predictions, deg=1)
		ax.plot(self.y_reg, fit[0] * self.y_reg + fit[1], color='red')
		
		plt.scatter(self.y_reg, predictions)
		#plt.show()

	def ridge_regression(self, data, predictors, alpha, models_to_plot={}):
		#Fit the model
		ridgereg = Ridge(alpha=alpha,normalize=True)
		ridgereg.fit(data[self.predictors],data['Battery(kJ)'])
		y_pred = ridgereg.predict(data[self.predictors])
		
		'''
		#Check if a plot is to be made for the entered alpha
		if alpha in models_to_plot:
			plt.subplot(models_to_plot[alpha])
			plt.tight_layout()
			plt.plot(self.df_seg[predictors],y_pred)
			plt.plot(self.df_seg[predictors],self.df_seg['Battery(kJ)'],'.')
			plt.title('Plot for alpha: %.3g'%alpha)
		'''
		#Return the result in pre-defined format
		rss = sum((y_pred-data['Battery(kJ)'])**2)
		ret = [rss]
		ret.extend([ridgereg.intercept_])
		ret.extend(ridgereg.coef_)
		#print(ret)
		return ret
	
	def ridge (self):
						
		#Initialize predictors to be set of 15 powers of x
		
		#print(len(self.predictors))
		#predictors.extend(['x_%d'%i for i in range(len(predictors))]
		
		#Set the different values of alpha to be tested
		alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

		#Initialize the dataframe for storing coefficients.
		col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,len(self.predictors)+1)]
		ind = ['alpha_%.2g'%alpha_ridge[i] for i in range(0,10)]
		coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)

		models_to_plot = {1e-15:231, 1e-10:232, 1e-4:233, 1e-3:234, 1e-2:235, 5:236}
		#print(self.df_seg)
		data=self.df_seg
		for i in range(10):
			coef_matrix_ridge.iloc[i,] = Datamining.ridge_regression(self, data, self.predictors, alpha_ridge[i], models_to_plot)

		ret = coef_matrix_ridge.iloc[5,]
		pd.options.display.float_format = '{:,.2g}'.format
		pd.options.display.max_columns = 999
		#print("{} {}".format("\nRIDGE\n", coef_matrix_ridge))
		print("\nOptimal data using Ridge: \n", main_optimize(ret,"Ridge"))
		print("\nCoefficients used with Ridge: \n", *ret[1:6])
		print("RSS Ridge:", ret[0])
		
	def lasso_regression (self, data, predictors, alpha, models_to_plot={}):
		#Fit the model
		lassoreg = Lasso(alpha=alpha,normalize=True, max_iter=1e6)
		lassoreg.fit(data[self.predictors],data['Battery(kJ)'])
		y_pred = lassoreg.predict(data[self.predictors])
		
		
		#Return the result in pre-defined format
		rss = sum((y_pred-data['Battery(kJ)'])**2)
		ret = [rss]
		ret.extend([lassoreg.intercept_])
		ret.extend(lassoreg.coef_)
		
		return ret
		
	def lasso (self):
		#Initialize predictors to all 15 powers of x
		#predictors=self.indep_var
		#self.predictors = predictors
		
		#predictors.extend(['x_%d'%i for i in range(2,16)])

		#Define the alpha values to test
		alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]

		#Initialize the dataframe to store coefficients
		col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,len(self.predictors)+1)]
		ind = ['alpha_%.2g'%alpha_lasso[i] for i in range(0,10)]
		coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)

		#Define the models to plot
		models_to_plot = {1e-10:231, 1e-5:232,1e-4:233, 1e-3:234, 1e-2:235, 1:236}
		data=self.df_seg
		#Iterate over the 10 alpha values:
		for i in range(10):
			coef_matrix_lasso.iloc[i,] = Datamining.lasso_regression(self, data, self.predictors, alpha_lasso[i], models_to_plot)
			
		ret = coef_matrix_lasso.iloc[6,]
		pd.options.display.float_format = '{:,.2g}'.format
		pd.options.display.max_columns = 999
		#print("{} {}".format("\nLASSO\n", coef_matrix_lasso))
		#print(coef_matrix_lasso.apply(lambda x: sum(x.values==0),axis=1))
		
		print("\nOptimal data using Lasso: \n", main_optimize(ret,"Lasso"))
		print("\nCoefficients used with Lasso: \n", *ret[1:6])
		print("RSS Lasso:", ret[0])
		
		
	#def coefficients(self):
	
	def elastic_net(self):
		alpha=0.01
		l1_ratio=0.7
		enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
		
		#self.predictors = ['t0_Vel_SensorMA','tn_Vel_SensorMA','Dist_SensorMA','Vel_SensorMA']
		self.predictors = self.indep_var
		enet.fit(self.df_seg[self.predictors],self.df_seg['Battery(kJ)'])

		y_pred_enet = enet.predict(self.df_seg[self.predictors])
		#print(y_pred_enet)
		
		rss = sum((y_pred_enet-self.df_seg['Battery(kJ)'])**2)
		ret = [rss]
		ret.extend([enet.intercept_])
		ret.extend(enet.coef_)
		#print(ret)
		print("Optimal data using ElasticNet: \n", main_optimize(ret,"ElasticNet"))
		print("\nCoefficients used with ElasticNet: \n", ret[1:6])		
		print("RSS ElasticNet:", ret[0])
	
	
	def add_powers(self):
		for i in range(2,5):  #power of 1 is already there	
			for j in self.indep_var:
				#print(j)
				colname = j + "_" + str(i)
				self.df_seg[colname]=self.df_seg[j]**i				
				self.df_power.append(colname)		
				


		
def main():
	attributes = ['position_x', 'position_y[deg]','Time[s]',
	'speed','Battery_StatusHV_Battery_Current[A]','Battery_Status_2HV_Power[W]',
	'IMU_GyroYawrate','IMU_GyroRollrate','IMU_GyroPitchrate','IMU_AccelerationAcceleration_X',
	'IMU_AccelerationAcceleration_Y']
	attributes_speed = ['speed','Front_Vehicle_StatusWheel_Speed_Front_Right',
	'Rear_Vehicle_StatusWheel_Speed_Rear_Left']
	'Front_Vehicle_StatusWheel_Speed_Front_Left','Rear_Vehicle_StatusWheel_Speed_Rear_Right',
	#variables2 = ['Time[s]', 'IMU_GPS', 'IMU_GPS', 'Speed_IMU']
	
	attributes = [""]
	file = ("DriveKTH.csv")
	batch_object = Datamining(file, attributes, attributes_speed)
	

if __name__ == '__main__':
    main()

end_time = time.clock()
print("\n\nTid: ", end_time-start_time)









