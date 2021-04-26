import pandas as pd
import numpy as np
import requests
import torch
from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time

#APIKEY = '04NF3SJ1QFXC82ER'  #polito 
APIKEY = '80H8XTPEDCUKU37T' #eurecom

def stockTS(function = None, symbol = None, interval = None, adjusted = True, outputsize = 'full', datatype = 'csv', apikey = APIKEY):
	"""
	Input:
	- function: specify the time frequency of the time series:
		- TIME_SERIES_INTRADAY, TIME_SERIES_DAILY, TIME_SERIES_WEEKLY, TIME_SERIES_MONTHLY
	- symbol: the ticker of the symbol of which you want to retrieve data
	- interval: required to be specified if TIME_SERIES_INTRADAY
	- outputsize: 'full' over 20 years of data, 'compact' last 100 data points
	- datatype: 'csv' or 'json'
	- apikey: personal key to retrieve data
	
	Returns:
	A pandas dataframe of historical prices and volume
	"""
	if function == 'TIME_SERIES_INTRADAY' and interval is None:
		raise ValueError("Missing interval specification")
	
	if adjusted:
		function += "_ADJUSTED"
	
	query = "https://www.alphavantage.co/query?function={}&symbol={}&interval={}&outputsize={}&datatype={}&apikey={}".format(function,symbol,interval,outputsize, datatype, apikey)
	df = pd.read_csv(query, header = 0, index_col='timestamp')
	df.index = pd.to_datetime(df.index)
	df = df.sort_index(ascending = True)
	return df

def portfolioTS(components, interval = None):
	"""
	Input:
	- ind_components: list of independent stocks
	- dep_components: list of dependent components
	- interval: required to be specified if TIME_SERIES_INTRADAY
	
	Returns:
	A merged pandas dataframe with the historical close prices
	"""
	dfs = []
	for comp in components:
		try:
			dfs.append(stockTS(function = 'TIME_SERIES_DAILY', symbol = comp, interval = interval)['adjusted_close'])
		except:
			time.sleep(60)
			dfs.append(stockTS(function = 'TIME_SERIES_DAILY', symbol = comp, interval = interval)['adjusted_close'])
			
	merged_df = pd.concat(dfs, axis = 1)
	merged_df = merged_df.dropna(how='any',axis=0)
	merged_df.columns = components
	return merged_df

class StockTimeSeries(Dataset):
	

	def __init__(self, symbol, window = 30, transform=None):
		
		self.df = stockTS(function = 'TIME_SERIES_DAILY',symbol = symbol)['close']
		self.df = self.df.reset_index(drop = True)
		self.df = self.df.values.tolist()
		self.X = np.array([[self.df[i:i+window]] for i in range(len(self.df)-window)])
		self.X = np.squeeze(self.X)
		self.Y = self.X[:,-1]
		self.X = np.delete(self.X, -1, 1)
		print(self.X, self.Y)
		self.X = MinMaxScaler().fit(self.X).transform(self.X)
		self.X = np.expand_dims(self.X, axis=1)
		self.transform = transform
		print(self.X.shape, self.Y.shape)
		
	
	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		x,y = self.X[idx,:], self.Y[idx]
		x,y = self.transform(x,y)
		return (x,y)
	
	
class PortfolioTimeSeries(Dataset):
	def __init__(self, components, window = 30, transform=None):
		
		self.df = portfolioTS(components)
		self.df = self.df.reset_index(drop = True)
		trains, labels = [], []
		for comp in components:
			comp_df = self.df[comp].values.tolist()
			comp_X = np.array([[comp_df[i:i+window-1]] for i in range(len(comp_df)-window)])
			comp_X = np.squeeze(comp_X)
			
			comp_Y = self.df.values[window:,:]
			
			trains.append(comp_X)
			
		self.X = np.concatenate(trains, axis = 1)
		self.Y = self.df[window:]
		self.X = MinMaxScaler().fit(self.X).transform(self.X)
		self.X = np.expand_dims(self.X, axis=1)
		self.transform = transform
		print(self.X.shape, self.Y.shape)
		
	
	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		x,y = self.X[idx,:], self.Y[idx]
		x,y = self.transform(x,y)
		return (x,y)

class TechnicalPortfolioTimeSeries(Dataset):
	def __init__(self, components, window = 30, pred_window = 7,  transform=None):
		dfs = []
		columns = []
		for symb in components:
			df = pd.read_csv("Tech/{}.csv".format(symb), header = 0, index_col = 0)
			df.index = pd.to_datetime(df.index)
			df = df.sort_index(ascending = True)
			df = df.sort_index(axis=1)
			df = df.drop(['close','open','high','low','dividend_amount','split_coefficient'], axis=1)
			dfs.append(df)

			columns += [col for col in df.columns]
		self.merged_df = pd.concat(dfs, axis = 1)
		self.merged_df = self.merged_df.dropna(how='any',axis=0)
		self.merged_df.columns = columns
		self.Y = self.merged_df['adjusted_close'].values[window+pred_window:]
		self.index = np.array([self.merged_df.index.day.values[window+pred_window:],
							  self.merged_df.index.month.values[window+pred_window:],
							  self.merged_df.index.year.values[window+pred_window:]]).T
		print(self.index.shape)
							   
		self.merged_df = self.merged_df.drop(['adjusted_close'], axis=1)
	
		self.norm_df = MinMaxScaler().fit(self.merged_df).transform(self.merged_df)
		
		num_f = int(len(self.merged_df.columns) / len(components))
		trains = []

		for i in range(len(self.merged_df)-window-pred_window):
			stack = []
			for j in range(0,len(self.merged_df.columns), num_f):
				stack.append(np.transpose(self.norm_df[i:i+window, j:j+num_f]))

			stack = np.concatenate(stack, axis = 1)
			trains.append(stack)

		self.X = np.array(trains)
		print(self.X.shape, self.Y.shape)
		self.transform = transform


	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		x,y = self.X[idx,:], self.Y[idx]
		index = self.index[idx,:]
		x,y = self.transform(x,y)
		return (x,y,index)
	
	

class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""
	def __call__(self, x,y):
		return torch.from_numpy(np.array(x)).type(torch.FloatTensor), torch.Tensor([y]).type(torch.FloatTensor).squeeze()
		

def returns(df, **kwargs):
	"""
	Input:
	- df: pandas dataframe with historical prices
	
	Returns:
	A pandas dataframe with added the column return
	"""
	df['return'] = (df['adjusted_close'] - df['adjusted_close'].shift(1))/df['adjusted_close'].shift(1)
	return df

def annualize_vol(returns, n_periods):
	"""
	Input:
	- returns: pandas series or array of returns
	- n_periods: number of periods composing an year in the df, to annualize the volatility
	
	Returns:
	A pandas series or array of volatilities
	"""
	return returns.rolling(window = len(returns), min_periods = 1).std()*(n_periods**0.5)

def annualize_rets(returns, n_periods):
	"""
	Input:
	- returns: pandas series or array of returns
	- n_periods: number of periods composing an year in the df, to annualize the returns
	
	Returns:
	A pandas series or array fo returns
	"""
	compounded_growth = (1+returns).prod()
	periods = returns.shape[0]
	return compounded_growth**(n_periods/periods)-1


def sharpe_ratio(returns, riskfree_rate, n_periods):
	"""
	Input:
	- returns: pandas series or array of returns
	- riskfree_rate: interest rate of risk-free assets
	- n_periods: number of periods composing an year in the df, to annualize the sharpe_ratio
	
	Returns:
	A pandas series or array of sharpe ratio
	"""
	# convert the annual riskfree rate to per period
	rf_per_period = (1+riskfree_rate)**(1/n_periods)-1
	excess_ret = returns - rf_per_period
	ann_ex_ret = annualize_rets(excess_ret, n_periods)
	ann_vol = annualize_vol(returns, n_periods)
	return ann_ex_ret/ann_vol



def getIndicators(symbol = None, interval = 'daily'):
	dfs = []
	indicators = {
		"EMA":[10,20,50,100],
		"SMA":[10,20,50,100],
		"WMA":[10,20,50,100],
		"DEMA":[10,20,50,100],
		"TEMA":[10,20,50,100],
		"TRIMA":[10,20,50,100],
		"KAMA":[10,20,50,100],
		"MACD":[None],
		"APO":[None],
		"PPO":[None],
		"MAMA":[None],
		"STOCH":[None],
		"OBV":[None],
		#"VWAP":[None],
		"RSI":[14],
		"MOM":[14],
		"CMO":[14],
		"ROC":[14],
		"BBANDS":[10,20,50,100],
		"WILLR":[14],
		"ADX":[14],
		"CCI":[20],
		"AROON":[25]
	}
	columns = []
	i = 0
	for ind in indicators:
		for period in indicators[ind]:
			print(ind+"_{}".format(period))
			df = getIndicator(function = ind, symbol = symbol, interval = interval, timeperiod = period)
			if period:
				columns += [col+"_{}".format(period) for col in df.columns]
			else:
				columns += [col for col in df.columns]
			dfs.append(df)
			i +=1
			if i == 5:
				time.sleep(60)
				i = 0
	
	merged_df = pd.concat(dfs, axis = 1)
	merged_df = merged_df.dropna(how='any',axis=0)
	merged_df.columns = columns
	return merged_df
	
	

def getIndicator(function = None, symbol = None, interval = None, timeperiod = None, series_type = 'close', datatype = 'json', apikey = APIKEY):
	"""
	Input:
	- function: specifying the Indicator
	- symbol: the ticker of the symbol of which you want to retrieve data
	- interval: required to be specified 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly
	- timeperiod: eventual window of the indicator
	- series_type: 'close','open','high','low' prices
	- apikey: personal key to retrieve data
	
	Returns:
	A pandas dataframe of historical oscillator
	"""
	
	if function in ['SMA','EMA','WMA','DEMA','TEMA','TRIMA','KAMA']:
		params = "function={}&symbol={}&interval={}&time_period={}&series_type={}&datatype={}&apikey={}".format(function, symbol,interval,timeperiod, series_type, datatype, apikey)
	elif function in ['MACD','APO','PPO','MAMA']:
		params = "function={}&symbol={}&interval={}&series_type={}&datatype={}&apikey={}".format(function, symbol,interval, series_type, datatype, apikey)
	elif function in ['STOCH','OBV','VWAP']:
		params = "function={}&symbol={}&interval={}&datatype={}&apikey={}".format(function, symbol,interval, datatype, apikey)
	elif function in ['RSI','MOM','CMO','ROC','BBANDS']:
		params = "function={}&symbol={}&interval={}&time_period={}&series_type={}&datatype={}&apikey={}".format(function, symbol,interval,timeperiod, series_type, datatype, apikey)
	elif function in ['WILLR','ADX','CCI','AROON']:
		params = "function={}&symbol={}&interval={}&time_period={}&datatype={}&apikey={}".format(function, symbol,interval,timeperiod, datatype, apikey)
	
	
	query = "https://www.alphavantage.co/query?" 
	resp = requests.get(query+params).json()
	data = resp["Technical Analysis: {}".format(function)]
	df = pd.DataFrame.from_dict(data, orient = 'index')
	df.index = pd.to_datetime(df.index)
	df = df.sort_index(ascending = True)
	return df

def download_csv(stocks,*args):
	for s in stocks:
		print(s)
		ind = ut.getIndicators(symbol = s)
		ind = ind.sort_index(axis=1)
		time.sleep(60)
		price = ut.stockTS(function='TIME_SERIES_DAILY',symbol = s)
		price = price.sort_index(axis=1)
		merged_df = pd.concat([ind,price], axis = 1)
		#merged_df = pd.merge(ind,price, left_index = True, right_index = True)
		merged_df = merged_df.dropna(how='any',axis=0)
		merged_df.columns = ind.columns + price.columns
		merged_df.to_csv('Tech/{}.csv'.format(s))
		time.sleep(60)