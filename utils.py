import pandas as pd
import numpy as np
import requests
import torch
from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from mcmc_samplers import SGHMCSampler,LossModule
from ResForkNet import *
import time
import datetime
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import os
import shutil
import threading
from queue import Queue, PriorityQueue
from scipy.signal import argrelextrema

plt.style.use('ggplot')
rc('animation', html='jshtml')

device = "cuda:0" if torch.cuda.is_available() else "cpu"

#APIKEY = '04NF3SJ1QFXC82ER'  #polito 
APIKEY = '80H8XTPEDCUKU37T' #eurecom

def stockTS(function = None, symbol = None, interval = None, adjusted = True, outputsize = 'full', datatype = 'csv', apikey = APIKEY):
	"""
	Method to obtain a series of historical prices and volumes for a particular stock
	Args:
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
	Method to obtain a series of historical prices and volumes for a particular portfolio of stocks
	Args:
	- ind_components: list of independent stocks
	- dep_components: list of dependent components
	- interval: required to be specified if TIME_SERIES_INTRADAY
	
	Returns:
	A merged pandas dataframe with the historical adjusted close prices
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

class ToTensor(object):
	"""
	Class to convert ndarrays to Torch Tensors
	"""
	def __call__(self, x,y):
		return torch.from_numpy(np.array(x)).type(torch.FloatTensor), torch.Tensor([y]).type(torch.FloatTensor).squeeze()
		

class StockTimeSeries(Dataset):
	"""
	Class to build a torch.utils.data.Dataset for a singular stock, already fragmented in windows, having as feature only the adjusted close price
	Args:
	- symbol: the stock of interest
	- window: the number of past samples the model will consider
	- transform: the torchvision.transform to be applied
	"""

	def __init__(self, symbol, window = 30, transform=None):
		
		self.df = stockTS(function = 'TIME_SERIES_DAILY',symbol = symbol)['adjusted_close']
		self.df = self.df.reset_index(drop = True)
		self.df = self.df.values.tolist()
		self.X = np.array([[self.df[i:i+window]] for i in range(len(self.df)-window)])
		self.X = np.squeeze(self.X)
		self.Y = self.X[:,-1]
		self.X = np.delete(self.X, -1, 1)
		self.X = MinMaxScaler().fit(self.X).transform(self.X)
		self.X = np.expand_dims(self.X, axis=1)
		self.transform = transform
		print(self.X.shape, self.Y.shape, end="")
		
	
	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		x,y = self.X[idx,:], self.Y[idx]
		x,y = self.transform(x,y)
		return (x,y)
	
	
class PortfolioTimeSeries(Dataset):
	"""
	Class to build a torch.utils.data.Dataset for a portfolio of stocks, already fragmented in windows, having as feature only the adjusted close price
	Args:
	- symbol: the stock of interest
	- window: the number of past samples the model will consider
	- transform: the torchvision.transform to be applied
	"""
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
		print(self.X.shape, self.Y.shape, end="")
		
	
	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		x,y = self.X[idx,:], self.Y[idx]
		x,y = self.transform(x,y)
		return (x,y)

class TechnicalPortfolioTimeSeries(Dataset):
	"""
	Class to build a torch.utils.data.Dataset for a portfolio of stocks, already fragmented in windows, having as feature the technical indicators
	Args:
	- symbol: the stock of interest
	- window: the number of past samples the model will consider
	- transform: the torchvision.transform to be applied
	"""
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
		print(self.X.shape, self.Y.shape, end="")
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



def getIndicators(symbol = None, interval = 'daily'):
	"""
	Method to obtain a pre-specified list of technical indicators for a particular stock
	Args:
	- symbol: the stock symbol for which we want to retrieve the indicators
	- interval: the frequency of data to be retrieved
	
	Returns:
	- a pandas DataFrame containing all the technical indicators for the specified stock
	"""
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
	Args:
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
	"""
	Method to download DataFrames with technical indicators, given a list of stocks of interest. It automatically tune the frequency of the requests
	Args:
	- stocks: list of the stocks to be retrieved
	"""
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

def build_test(test_loader, dataset=None, window = 30, pred_window = 30, num_f = 60, more_days = True):
	"""
	Method to build test_data to be predicted
	Args:
	- test_loader: DataLoader containing test_data
	- dataset: original dataset
	- window: window used as the "memory" of the model
	- pred_windw: window used for the predictions
	- num_f: dimensionality of the feature space
	- more_days: flag to add those days not covered by test labels

	Returns:
	- test_data, test_labels, index of test_labels 
	"""
	test_data, test_labels, index = next(iter(test_loader))
	date_index = [datetime.datetime(index[i,2],index[i,1],index[i,0]) for i in range(index.shape[0])]
	test_data = test_data.to(device)

	if more_days:
		trains = []
		for i in range(len(dataset.merged_df)-window-pred_window, len(dataset.merged_df)-window):
			stack = []
			for j in range(0,len(dataset.merged_df.columns), num_f):
				stack.append(np.transpose(dataset.norm_df[i:window+i, j:j+num_f]))

			stack = np.concatenate(stack, axis = 1)
			trains.append(stack)
		new_data = np.array(trains)
		test_data = torch.cat([test_data, torch.Tensor(new_data).type(torch.FloatTensor).to(device)], axis = 0)

	return test_data, test_labels, date_index

def plot_predictions(stocks, test_labels, preds, index_true, index_pred, show_days = 100, savefig = False):
	"""
	Method to plot predictions
	Args:
	- stocks: components to be plotted
	- test_labels: ground truth to be plotted
	- preds: predictions to be plotted
	- index_true: temporal index of the ground truth
	- index_pred: temporal index of the predictions
	- show_days: number of days to show
	- show_days: number of most recent days to plot
	- savefig: whether to save or not the plot
	"""
	
	fig, axs = plt.subplots(len(stocks),1, figsize=(30,len(stocks)*15))
	for i,ax in enumerate(axs):
		avg = torch.mean(preds[:,:,i], dim = 0)
		q95 = np.percentile(preds[:,:,i].cpu().numpy(), 95, axis = 0)
		q05 = np.percentile(preds[:,:,i].cpu().numpy(), 5, axis = 0)
		q0 = np.min(preds[:,:,i].cpu().numpy(), axis = 0)
		q100 = np.max(preds[:,:,i].cpu().numpy(), axis = 0)

		ax.plot(index_pred[-show_days:],avg[-show_days:].detach().cpu().numpy(),
				label = stocks[i]+" predicted")
		ax.fill_between(index_pred[-show_days:], q95[-show_days:], q05[-show_days:], alpha = 0.3, label = stocks[i]+" uncertainty")
		ax.scatter(index_pred[-show_days:], q0[-show_days:], marker='x', s = 1.4, c='r', label='Max deviation')
		ax.scatter(index_pred[-show_days:], q100[-show_days:], marker='x', s = 1.4, c='r')
		ax.plot(index_true[-show_days:],test_labels[-show_days:,i].detach().cpu().numpy(),
				label = stocks[i]+" true")
		ax.set_xlabel('Date',fontsize=14)
		ax.set_ylabel('Price',fontsize=14)
		ax.legend(fontsize=14)
		
	if savefig:
		plt.savefig("Stocks.png")
	
	plt.show()


def plot_predictions_from_samples(sampled_weights, model, stocks, test_loader, dataset, window = 30, pred_window = 30, num_f = 60, show_days = 100, savefig = False):
	"""
	Method to plot predictions from sampled weights
	Args:
	- sampled_weights: collection of sampled models
	- model: model architecture
	- stocks: components to be plotted
	- test_loader: DataLoader containing test_data
	- dataset: original dataset
	- window: window used as the "memory" of the model
	- pred_windw: window used for the predictions
	- num_f: dimensionality of the feature space
	- show_days: number of most recent days to plot
	- savefig: whether to save or not the plot
	"""
	"""
	Pick test data up to the last available day
	"""
	print("\r", end="")
	print("Building test data", end="")
	test_data, test_labels, index = build_test(test_loader, dataset, window, pred_window, num_f)
	
	"""
	Compute the predictions
	"""
	print("\r", end="")
	print("Computing the predictions", end="")
	outputs = []
	for i, set_params in enumerate(sampled_weights):       
		state_dict = {}
		for k,(name, param) in enumerate(model.named_parameters()):
			state_dict[name] = torch.from_numpy(set_params[k])
		state_dict_it = OrderedDict(state_dict)
		model.load_state_dict(state_dict_it, strict=False)
		with torch.no_grad():
			output = model(test_data)
		outputs.append(torch.unsqueeze(output,0))
	preds = torch.cat(outputs, dim=0)

	index_true = index
	#compute the index up to the last predicted day
	index_pred = index_true + [index_true[-1] + datetime.timedelta(days=i) for i in range(1,pred_window+1)]
	
	"""
	Plot predictions vs. true trend
	"""
	print("\r", end="")
	print("Plotting data", end="")
	plot_predictions(stocks, test_labels, preds, index_true, index_pred, show_days, savefig)
	print("\r", end="")
	


def plot_predictions_from_files(model_folder, model, stocks, test_loader, dataset, window = 30, pred_window = 30, num_f = 60, show_days = 100, savefig = False):
	"""
	Method to plot predictions from saved models
	Args:
	- sampled_weights: collection of sampled models
	- model: model architecture
	- stocks: components to be plotted
	- test_loader: DataLoader containing test_data
	- dataset: original dataset
	- window: window used as the "memory" of the model
	- pred_windw: window used for the predictions
	- num_f: dimensionality of the feature space
	- show_days: number of most recent days to plot
	- savefig: whether to save or not the plot
	"""
	"""
	Pick test data up to the last available day
	"""
	print("\r", end="")
	print("Building test data", end="")
	test_data, test_labels, index = build_test(test_loader, dataset, window, pred_window, num_f)
	
	"""
	Compute the predictions
	"""
	print("\r", end="")
	print("Computing the predictions", end="")
	outputs = []

	for f, file_model in enumerate(os.listdir(model_folder)):
		model.load_state_dict(torch.load(model_folder + "/" + file_model, map_location = torch.device(device)))
		model = model.to(device)

		with torch.no_grad():
			output = model(test_data)
		outputs.append(torch.unsqueeze(output,0))

		print("",end="\r")
		print("Processed File {} of {}".format(f+1, len(os.listdir(model_folder))),end="")
	
	preds = torch.cat(outputs, dim=0)

	index_true = index
	#compute the index up to the last predicted day
	index_pred = index_true + [index_true[-1] + datetime.timedelta(days=i) for i in range(1,pred_window+1)]

	"""
	Plot predictions vs. true trend
	"""
	print("\r", end="")
	print("Plotting data", end="")
	plot_predictions(stocks, test_labels, preds, index_true, index_pred, show_days, savefig)
	print("\r", end="")

def get_predictions_from_files(model_folder, model, stocks, test_loader):
	"""
	Method to plot predictions from saved models
	Args:
	- sampled_weights: collection of sampled models
	- model: model architecture
	- stocks: components to be plotted
	- test_loader: DataLoader containing test_data
	- dataset: original dataset
	
	Returns:
	- predictions
	- true_labels
	- index
	"""
	print("\r", end="")
	print("Building test data", end="")
	test_data, test_labels, index = build_test(test_loader, more_days = False)

	"""
	Compute the predictions
	"""
	print("\r", end="")
	print("Computing the predictions", end="")
	outputs = []

	for f, file_model in enumerate(os.listdir(model_folder)):
		model.load_state_dict(torch.load(model_folder + "/" + file_model, map_location = torch.device(device)))
		model = model.to(device)

		with torch.no_grad():
			output = model(test_data)
		outputs.append(torch.unsqueeze(output,0))

		print("",end="\r")
		print("Processed File {} of {}".format(f+1, len(os.listdir(model_folder))),end="")
	
	preds = torch.cat(outputs, dim=0)

	return preds, test_labels, index


"""
Performance Assessment
"""
def args_as_tensors(*index):
	"""A simple decorator to convert numpy arrays to torch tensors"""
	def decorator(method):
		def wrapper(*args, **kwargs):
			converted_args = [torch.tensor(a).float() 
							  if i in index and type(a) is np.ndarray else a 
							  for i, a in enumerate(args)]
			return method(*converted_args, **kwargs)
		return wrapper  
	return decorator

@args_as_tensors(0, 1)
def PearsonCorr(true, target, reduction = True):
	"""
	Method to compute the Pearson Correlation coefficient between two sequences
	Args:
	- true: a tensor num_prices x num_stocks
	- target: a tensor batch x num_prices x num_stocks
	
	Returns:
	- the Pearson Correlation coefficient
	"""
	v1 = true - true.mean(0)
	v2 = target - target.mean(1).unsqueeze(1)
	pearson_coeff = 1-(v1*v2).sum(1) / (torch.sqrt((v1**2).sum(0)) * torch.sqrt(v2 ** 2).sum(1))

	if reduction:
		return pearson_coeff.mean(0), pearson_coeff.std(0)
	else:
		return pearson_coeff

@args_as_tensors(0, 1)
def MSE(true, target, reduction = True):
	"""
	Method to compute the Mean Squared Error between two sequences
	Args:
	- true: a tensor num_prices x num_stocks
	- target: a tensor batch x num_prices x num_stocks
	
	
	Returns:
	- the Mean Squared Error
	"""
	mse = ((true - target)**2).mean(1)

	if reduction:
		return mse.mean(0), mse.std(0)
	else:
		return mse

@args_as_tensors(0, 1)
def RMSE(true, target, reduction = True):
	"""
	Method to compute the Root Mean Squared Error between two sequences
	Args:
	- true: a tensor num_prices x num_stocks
	- target: a tensor batch x num_prices x num_stocks
	
	
	Returns:
	- the Root Mean Squared Error
	"""
	rmse = torch.sqrt(((true - target)**2).mean(1))

	if reduction:
		return rmse.mean(0), rmse.std(0)
	else:
		return rmse

@args_as_tensors(0, 1)
def MAE(true, target, reduction = True):
	"""
	Method to compute the Mean Absolute Error between two sequences
	Args:
	- y1, y2: the two sequences for whcih we want to compute the Mean Absolute Error
	
	Returns:
	- the Mean Absolute Error
	"""
	mae = torch.abs(true - target).mean(1)

	if reduction:
		return mae.mean(0), mae.std(0)
	else:
		return mae

@args_as_tensors(0, 1)
def MAPE(true, target, reduction = True):
	"""
	Method to compute the Mean Absolute Percentage Error between two sequences
	Args:
	- true: a tensor num_prices x num_stocks
	- target: a tensor batch x num_prices x num_stocks
	
	Returns:
	- the Mean Absolute Percentage Error
	"""
	mape = (torch.abs(true-target) / true).mean(1)

	if reduction:
		return mape.mean(0), mape.std(0)
	else:
		return mape

@args_as_tensors(0, 1)
def aggregate_statistics(preds, test_labels, stocks):
	"""
	Method to compute the aggregated statistics between a set of predictions and the groud truth
	Args:
	- test_labels: a tensor num_prices x num_stocks
	- preds: a tensor batch x num_prices x num_stocks
	- stocks: the names of the components
	
	Returns:
	- a pandas DataFrame with the aggregated statistics
	"""
	preds, test_labels = preds.to(device), test_labels.to(device)
	
	return pd.DataFrame(
		{
			("MSE","mean") : MSE(test_labels, preds)[0].cpu().numpy(), ("MSE", "std"): MSE(test_labels, preds)[1].cpu().numpy(),
			("MAE","mean") : MAE(test_labels, preds)[0].cpu().numpy(), ("MAE", "std"): MAE(test_labels, preds)[1].cpu().numpy(),
			("MAPE","mean") : MAPE(test_labels, preds)[0].cpu().numpy(), ("MAPE", "std"): MAPE(test_labels, preds)[1].cpu().numpy(),
			("RMSE","mean") : RMSE(test_labels, preds)[0].cpu().numpy(), ("RMSE", "std"): RMSE(test_labels, preds)[1].cpu().numpy(),
			("PearsonCorr","mean") : PearsonCorr(test_labels, preds)[0].cpu().numpy(), ("PearsonCorr", "std"): PearsonCorr(test_labels, preds)[1].cpu().numpy()
			
		}, index = stocks
	)

@args_as_tensors(0, 1)
def plot_hist_errors(preds, test_labels, stocks, axs, label):
	"""
	Method to plot the aggregated statistics between a set of predictions and the groud truth
	Args:
	- test_labels: a tensor num_prices x num_stocks
	- preds: a tensor batch x num_prices x num_stocks
	- stocks: the names of the components
	- savefig (optional): whether to save or not the plot
	"""
	preds, test_labels = preds.to(device), test_labels.to(device)

	aggregate_fns = [("MSE",MSE), ("MAE",MAE), ("MAPE",MAPE), ("RMSE",RMSE), ("PearsonCorr",PearsonCorr)]

	stats = OrderedDict({fn[0] : fn[1](test_labels, preds, reduction = False) for fn in aggregate_fns})

	for j,stat in enumerate(stats.keys()):
		for i, stock in enumerate(stocks):
			metric = pd.Series(stats[stat][:, i].cpu().numpy())
			axs[i,j].grid()
			metric.plot.kde(bw_method = 0.3, ax = axs[i,j], label = label)
			plt.setp(axs[i,j].get_xticklabels(), rotation=45)
			axs[i,j].set_title("{}-{}".format(stock, stat))
	

'''
def plot_hist_errors(stocks, model_dict, window = 30, pred_window = 30, **kwargs):
	"""
	Method to plot distributions of the errors for a model
	Args:
	- stocks: the model referred to some stocks
	- model_dict: a tuple containing the instance of the model and the initialization of the model
	- window: the specification of the model's window on past samples
	- pred_window: the specification of the model's prediction window parameter
	- **kwargs: additional parameters as the specifications of the batch_size and the command to save the plots
	"""

	batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 32

	"""
	Creating sequential iterators with a training window of 3*batch_size days, with a stride = 1
	"""
	print("\r")
	print("Building the loaders", end="")

	dataset = TechnicalPortfolioTimeSeries(components = stocks, window = window, pred_window = pred_window, transform = ToTensor())
	loader = DataLoader(dataset, batch_size = batch_size, drop_last = True, shuffle = False, num_workers = 0)

	train_loaders, test_loaders = [], []
	iterator = []

	for i, (batch, labels, _) in enumerate(loader, 0):
		if len(iterator) != 3:
			iterator.append((batch, labels, _))
		else:
			train_loaders.append(iterator)
			test_loaders.append([(batch, labels, _)])
			iterator.pop(0)
			iterator.append((batch, labels, _))

	"""
	Defining collections of errors
	"""
	
	fig, axs = plt.subplots(len(stocks), 5, figsize = (30,len(stocks)*15))

	model, path_to_model = model_dict

	stats = OrderedDict({"mae":[], "mape":[], "rmse":[], "mse":[], "corr":[]})
	
	#Sampler Parameters
	lr = kwargs['lr'] if 'lr' in kwargs else 0.0007
	num_burn_in_steps = kwargs['num_burn_in_steps'] if 'num_burn_in_steps' in kwargs else 300
	keep_every = kwargs['keep_every'] if 'keep_every' in kwargs else 100
	nsamples = kwargs['nsamples'] if 'nsamples' in kwargs else 7


	for i, train_loader in enumerate(train_loaders,0):
		print("\r")
		print("Training and Plotting {}/{}".format(i+1, len(train_loaders)), end="")

		model = model.to(device)
		model.load_state_dict(torch.load(path_to_model, map_location = torch.device(device)))
		loss_fn = torch.nn.MSELoss(reduction='sum')
		lm = LossModule(model,train_loader, loss_fn, N = batch_size*3)
		SGHMC = SGHMCSampler(lm, num_burn_in_steps=num_burn_in_steps, lr = lr, keep_every=keep_every)
		SGHMC.sample(nsamples=nsamples)

		os.makedirs('temp_models')
		for i, set_params in enumerate(SGHMC.sampled_weights):       
			state_dict = {}
			for k,(name, param) in enumerate(model.named_parameters()):
				state_dict[name] = torch.from_numpy(set_params[k])
			state_dict_it = OrderedDict(state_dict)
			model.load_state_dict(state_dict_it, strict=False)
			torch.save(model.state_dict(),"temp_models/model_{}".format(i))

		mse, rmse, mae, mape, corr = compute_statistics(stocks, model, 'temp_models', test_loaders[i], verbose = False)

		shutil.rmtree('temp_models', ignore_errors=True)

		stats["mse"].append(mse)
		stats["mae"].append(mae)
		stats["rmse"].append(rmse)
		stats["mape"].append(mape)
		stats["corr"].append(corr)

	for s, stock in enumerate(stocks,0):
		for p, key in enumerate(stats.keys(),0):
			hist = torch.stack(stats[key], dim=0)

			axs[s,p].grid()
			axs[s,p].hist(hist[:,s].cpu().numpy(), bins = min(70, len(train_loaders)), edgecolor='k', linewidth=1.2)
			axs[s,p].text(0.68, 0.04, 'Mean = {}\n Std = {}'.format(round(torch.mean(hist[:,s]).item(),3), round(torch.std(hist[:,s]).item(),3)),
			style='italic', transform=axs[s,p].transAxes, fontsize = 12, family='fantasy',
			bbox={'facecolor':'tomato', 'alpha':0.9, 'pad':5})

			axs[s,p].set_title("{}'s {}".format(stock, key))


	if 'savefig' in kwargs:
		plt.savefig("Errors.png")

	plt.show()
'''
