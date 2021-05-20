import pandas as pd
import numpy as np
from numpy.linalg import inv
from scipy.optimize import minimize
import scipy.stats
from scipy.stats import norm


def returns(df, **kwargs):
	"""
	Input:
	- df: pandas dataframe with historical prices
	
	Returns:
	A pandas dataframe with added the column return
	"""
	return_df = (df - df.shift(1))/df.shift(1)
	return df

def annualize_vol(returns, n_periods=250):
	"""
	Input:
	- returns: pandas series or array of returns
	- n_periods: number of periods composing an year in the df, to annualize the volatility
	
	Returns:
	A pandas series or array of volatilities
	"""
	return returns.rolling(window = len(returns), min_periods = 1).std()*(n_periods**0.5)

def annualize_rets(returns, n_periods=250):
	"""
	Method to evaluate the annualized returns given a dataframe of returns
	Args:
	- returns: pandas series or array of returns
	- n_periods: number of periods composing an year in the df, to annualize the returns
	
	Returns:
	A pandas series or array of returns
	"""
	compounded_growth = (1+returns).prod()
	periods = returns.shape[0]
	return compounded_growth**(n_periods/periods)-1


def sharpe_ratio(returns, riskfree_rate, n_periods=250):
	"""
	Args:
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

def is_normal(r, level=0.01):
	"""
	Applies the Jarque-Bera test to determine if a Series is normal or not
	Test is applied at the 1% level by default
	Returns True if the hypothesis of normality is accepted, False otherwise
	"""
	if isinstance(r, pd.DataFrame):
		return r.aggregate(is_normal)
	else:
		statistic, p_value = scipy.stats.jarque_bera(r)
		return p_value > level


def drawdown(return_series: pd.Series):
	"""Takes a time series of asset returns.
	   returns a DataFrame with columns for
	   the wealth index, 
	   the previous peaks, and 
	   the percentage drawdown
	"""
	wealth_index = 1000*(1+return_series).cumprod()
	previous_peaks = wealth_index.cummax()
	drawdowns = (wealth_index - previous_peaks)/previous_peaks
	return pd.DataFrame({"Wealth": wealth_index, 
						 "Previous Peak": previous_peaks, 
						 "Drawdown": drawdowns})


def semideviation(r):
	"""
	Returns the semideviation aka negative semideviation of r
	r must be a Series or a DataFrame, else raises a TypeError
	"""
	if isinstance(r, pd.Series):
		is_negative = r < 0
		return r[is_negative].std(ddof=0)
	elif isinstance(r, pd.DataFrame):
		return r.aggregate(semideviation)
	else:
		raise TypeError("Expected r to be a Series or DataFrame")


def var_historic(r, level=5):
	"""
	Returns the historic Value at Risk at a specified level
	i.e. returns the number such that "level" percent of the returns
	fall below that number, and the (100-level) percent are above
	"""
	if isinstance(r, pd.DataFrame):
		return r.aggregate(var_historic, level=level)
	elif isinstance(r, pd.Series):
		return -np.percentile(r, level)
	else:
		raise TypeError("Expected r to be a Series or DataFrame")


def cvar_historic(r, level=5):
	"""
	Computes the Conditional VaR of Series or DataFrame
	"""
	if isinstance(r, pd.Series):
		is_beyond = r <= -var_historic(r, level=level)
		return -r[is_beyond].mean()
	elif isinstance(r, pd.DataFrame):
		return r.aggregate(cvar_historic, level=level)
	else:
		raise TypeError("Expected r to be a Series or DataFrame")


def skewness(r):
	"""
	Alternative to scipy.stats.skew()
	Computes the skewness of the supplied Series or DataFrame
	Returns a float or a Series
	"""
	demeaned_r = r - r.mean()
	# use the population standard deviation, so set dof=0
	sigma_r = r.std(ddof=0)
	exp = (demeaned_r**3).mean()
	return exp/sigma_r**3


def kurtosis(r):
	"""
	Alternative to scipy.stats.kurtosis()
	Computes the kurtosis of the supplied Series or DataFrame
	Returns a float or a Series
	"""
	demeaned_r = r - r.mean()
	# use the population standard deviation, so set dof=0
	sigma_r = r.std(ddof=0)
	exp = (demeaned_r**4).mean()
	return exp/sigma_r**4


def compound(r):
	"""
	returns the result of compounding the set of returns in r
	"""
	return np.expm1(np.log1p(r).sum())


def var_gaussian(r, level=5, modified=False):
	"""
	Returns the Parametric Gauusian VaR of a Series or DataFrame
	If "modified" is True, then the modified VaR is returned,
	using the Cornish-Fisher modification
	"""
	# compute the Z score assuming it was Gaussian
	z = norm.ppf(level/100)
	if modified:
		# modify the Z score based on observed skewness and kurtosis
		s = skewness(r)
		k = kurtosis(r)
		z = (z +
				(z**2 - 1)*s/6 +
				(z**3 -3*z)*(k-3)/24 -
				(2*z**3 - 5*z)*(s**2)/36
			)
	return -(r.mean() + z*r.std(ddof=0))


def summary_stats(r, riskfree_rate=0.03):
	"""
	Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
	"""
	ann_r = r.aggregate(annualize_rets, periods_per_year=12)
	ann_vol = r.aggregate(annualize_vol, periods_per_year=12)
	ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=12)
	dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
	skew = r.aggregate(skewness)
	kurt = r.aggregate(kurtosis)
	cf_var5 = r.aggregate(var_gaussian, modified=True)
	hist_cvar5 = r.aggregate(cvar_historic)
	return pd.DataFrame({
		"Annualized Return": ann_r,
		"Annualized Vol": ann_vol,
		"Skewness": skew,
		"Kurtosis": kurt,
		"Cornish-Fisher VaR (5%)": cf_var5,
		"Historic CVaR (5%)": hist_cvar5,
		"Sharpe Ratio": ann_sr,
		"Max Drawdown": dd
	})

def portfolio_return(weights, returns):
	"""
	Computes the return on a portfolio from constituent returns and weights
	weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
	"""
	return weights.T @ returns


def portfolio_vol(weights, covmat):
	"""
	Computes the vol of a portfolio from a covariance matrix and constituent weights
	weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
	"""
	vol = (weights.T @ covmat @ weights)**0.5
	return vol

def minimize_vol(target_return, er, cov):
	"""
	Returns the optimal weights that achieve the target return
	given a set of expected returns and a covariance matrix
	"""
	n = er.shape[0]
	init_guess = np.repeat(1/n, n)
	bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
	# construct the constraints
	weights_sum_to_1 = {'type': 'eq',
						'fun': lambda weights: np.sum(weights) - 1
	}
	return_is_target = {'type': 'eq',
						'args': (er,),
						'fun': lambda weights, er: target_return - portfolio_return(weights,er)
	}
	weights = minimize(portfolio_vol, init_guess,
					   args=(cov,), method='SLSQP',
					   options={'disp': False},
					   constraints=(weights_sum_to_1,return_is_target),
					   bounds=bounds)
	return weights.x

def msr(riskfree_rate, er, cov):
	"""
	Returns the weights of the portfolio that gives you the maximum sharpe ratio
	given the riskfree rate and expected returns and a covariance matrix
	"""
	n = er.shape[0]
	init_guess = np.repeat(1/n, n)
	bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
	# construct the constraints
	weights_sum_to_1 = {'type': 'eq',
						'fun': lambda weights: np.sum(weights) - 1
	}
	def neg_sharpe(weights, riskfree_rate, er, cov):
		"""
		Returns the negative of the sharpe ratio
		of the given portfolio
		"""
		r = portfolio_return(weights, er)
		vol = portfolio_vol(weights, cov)
		return -(r - riskfree_rate)/vol
	
	weights = minimize(neg_sharpe, init_guess,
					   args=(riskfree_rate, er, cov), method='SLSQP',
					   options={'disp': False},
					   constraints=(weights_sum_to_1,),
					   bounds=bounds)
	return weights.x

def gmv(cov):
	"""
	Returns the weights of the Global Minimum Volatility portfolio
	given a covariance matrix
	"""
	n = cov.shape[0]
	return msr(0, np.repeat(1, n), cov)

def sample_cov(r, **kwargs):
	"""
	Returns the sample covariance of the supplied returns
	"""
	return r.cov()

def weight_gmv(r, cov_estimator=sample_cov, **kwargs):
	"""
	Produces the weights of the GMV portfolio given a covariance matrix of the returns 
	"""
	est_cov = cov_estimator(r, **kwargs)
	return gmv(est_cov)

def cc_cov(r, **kwargs):
	"""
	Estimates a covariance matrix by using the Elton/Gruber Constant Correlation model
	"""
	rhos = r.corr()
	n = rhos.shape[0]
	# this is a symmetric matrix with diagonals all 1 - so the mean correlation is ...
	rho_bar = (rhos.values.sum()-n)/(n*(n-1))
	ccor = np.full_like(rhos, rho_bar)
	np.fill_diagonal(ccor, 1.)
	sd = r.std()
	return pd.DataFrame(ccor * np.outer(sd, sd), index=r.columns, columns=r.columns)

def shrinkage_cov(r, delta=0.5, **kwargs):
	"""
	Covariance estimator that shrinks between the Sample Covariance and the Constant Correlation Estimators
	"""
	prior = cc_cov(r, **kwargs)
	sample = sample_cov(r, **kwargs)
	return delta*prior + (1-delta)*sample

def style_analysis(dependent_variable, explanatory_variables):
	"""
	Returns the optimal weights that minimizes the Tracking error between
	a portfolio of the explanatory variables and the dependent variable
	"""
	n = explanatory_variables.shape[1]
	init_guess = np.repeat(1/n, n)
	bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
	# construct the constraints
	weights_sum_to_1 = {'type': 'eq',
						'fun': lambda weights: np.sum(weights) - 1
	}
	solution = minimize(portfolio_tracking_error, init_guess,
					   args=(dependent_variable, explanatory_variables,), method='SLSQP',
					   options={'disp': False},
					   constraints=(weights_sum_to_1,),
					   bounds=bounds)
	weights = pd.Series(solution.x, index=explanatory_variables.columns)
	return weights

def weight_ew(r, cap_weights=None, max_cw_mult=None, microcap_threshold=None, **kwargs):
	"""
	Returns the weights of the EW portfolio based on the asset returns "r" as a DataFrame
	If supplied a set of capweights and a capweight tether, it is applied and reweighted 
	"""
	n = len(r.columns)
	ew = pd.Series(1/n, index=r.columns)
	if cap_weights is not None:
		cw = cap_weights.loc[r.index[0]] # starting cap weight
		## exclude microcaps
		if microcap_threshold is not None and microcap_threshold > 0:
			microcap = cw < microcap_threshold
			ew[microcap] = 0
			ew = ew/ew.sum()
		#limit weight to a multiple of capweight
		if max_cw_mult is not None and max_cw_mult > 0:
			ew = np.minimum(ew, cw*max_cw_mult)
			ew = ew/ew.sum() #reweight
	return ew

def risk_contribution(w,cov):
	"""
	Compute the contributions to risk of the constituents of a portfolio, given a set of portfolio weights and a covariance matrix
	"""
	total_portfolio_var = portfolio_vol(w,cov)**2
	# Marginal contribution of each constituent
	marginal_contrib = cov@w
	risk_contrib = np.multiply(marginal_contrib,w.T)/total_portfolio_var
	return risk_contrib

def target_risk_contributions(target_risk, cov):
	"""
	Returns the weights of the portfolio that gives you the weights such
	that the contributions to portfolio risk are as close as possible to
	the target_risk, given the covariance matrix
	"""
	n = cov.shape[0]
	init_guess = np.repeat(1/n, n)
	bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
	# construct the constraints
	weights_sum_to_1 = {'type': 'eq',
						'fun': lambda weights: np.sum(weights) - 1
	}
	def msd_risk(weights, target_risk, cov):
		"""
		Returns the Mean Squared Difference in risk contributions
		between weights and target_risk
		"""
		w_contribs = risk_contribution(weights, cov)
		return ((w_contribs-target_risk)**2).sum()
	
	weights = minimize(msd_risk, init_guess,
					   args=(target_risk, cov), method='SLSQP',
					   options={'disp': False},
					   constraints=(weights_sum_to_1,),
					   bounds=bounds)
	return weights.x

def equal_risk_contributions(cov):
	"""
	Returns the weights of the portfolio that equalizes the contributions
	of the constituents based on the given covariance matrix
	"""
	n = cov.shape[0]
	return target_risk_contributions(target_risk=np.repeat(1/n,n), cov=cov)

def weight_erc(r, cov_estimator=sample_cov, **kwargs):
	"""
	Produces the weights of the ERC portfolio given a covariance matrix of the returns 
	"""
	est_cov = cov_estimator(r, **kwargs)
	return equal_risk_contributions(est_cov)


def implied_returns(delta, sigma, w):
	"""
	Obtain the implied expected returns by reverse engineering the weights
	Inputs:
	delta: Risk Aversion Coefficient (scalar)
	sigma: Variance-Covariance Matrix (N x N) as DataFrame
		w: Portfolio weights (N x 1) as Series
	Returns an N x 1 vector of Returns as Series
	"""
	ir = delta * sigma.dot(w).squeeze() # to get a series from a 1-column dataframe
	ir.name = 'Implied Returns'
	return ir

# Assumes that Omega is proportional to the variance of the prior
def proportional_prior(sigma, tau, p):
	"""
	Returns the He-Litterman simplified Omega
	Inputs:
	sigma: N x N Covariance Matrix as DataFrame
	tau: a scalar
	p: a K x N DataFrame linking Q and Assets
	returns a P x P DataFrame, a Matrix representing Prior Uncertainties
	"""
	helit_omega = p.dot(tau * sigma).dot(p.T)
	# Make a diag matrix from the diag elements of Omega
	return pd.DataFrame(np.diag(np.diag(helit_omega.values)),index=p.index, columns=p.index)

from numpy.linalg import inv

def bl(w_prior, sigma_prior, p, q,
				omega=None,
				delta=2.5, tau=.02):
	"""
	Computes the posterior expected returns based on 
	the original black litterman reference model
	
	W.prior must be an N x 1 vector of weights, a Series
	Sigma.prior is an N x N covariance matrix, a DataFrame
	P must be a K x N matrix linking Q and the Assets, a DataFrame
	Q must be an K x 1 vector of views, a Series
	Omega must be a K x K matrix a DataFrame, or None
	if Omega is None, we assume it is
	proportional to variance of the prior
	delta and tau are scalars
	"""
	if omega is None:
		omega = proportional_prior(sigma_prior, tau, p)
	# Force w.prior and Q to be column vectors
	# How many assets do we have?
	N = w_prior.shape[0]
	# And how many views?
	K = q.shape[0]
	# First, reverse-engineer the weights to get pi
	pi = implied_returns(delta, sigma_prior,  w_prior)
	# Adjust (scale) Sigma by the uncertainty scaling factor
	sigma_prior_scaled = tau * sigma_prior  
	# posterior estimate of the mean, use the "Master Formula"
	# we use the versions that do not require
	# Omega to be inverted (see previous section)
	# this is easier to read if we use '@' for matrixmult instead of .dot()
	#     mu_bl = pi + sigma_prior_scaled @ p.T @ inv(p @ sigma_prior_scaled @ p.T + omega) @ (q - p @ pi)
	mu_bl = pi + sigma_prior_scaled.dot(p.T).dot(inv(p.dot(sigma_prior_scaled).dot(p.T) + omega).dot(q - p.dot(pi).values))
	# posterior estimate of uncertainty of mu.bl
	sigma_bl = sigma_prior + sigma_prior_scaled - sigma_prior_scaled @ p.T @ inv(p @ sigma_prior_scaled @ p.T + omega) @ p @ sigma_prior_scaled
	#sigma_bl = sigma_prior + sigma_prior_scaled - sigma_prior_scaled.dot(p.T).dot(inv(p.dot(sigma_prior_scaled).dot(p.T) + omega)).dot(p).dot(sigma_prior_scaled)
	return (mu_bl, sigma_bl)

def inverse(d):
	"""
	Invert the dataframe by inverting the underlying matrix
	"""
	return pd.DataFrame(inv(d.values), index=d.columns, columns=d.index)

def w_star(delta, sigma, mu):
	return (inverse(sigma).dot(mu))/delta

def w_msr(sigma, mu, scale=True):
	"""
	Optimal (Tangent/Max Sharpe Ratio) Portfolio weights
	by using the Markowitz Optimization Procedure
	Mu is the vector of Excess expected Returns
	Sigma must be an N x N matrix as a DataFrame and Mu a column vector as a Series
	This implements page 188 Equation 5.2.28 of
	"The econometrics of financial markets" Campbell, Lo and Mackinlay.
	"""
	w = inverse(sigma).dot(mu)
	if scale:
		w = w/sum(w) # fix: this assumes all w is +ve
	return w

"""
Plots
"""
def backtest_ws(r_true, r_pred, estimation_window=30, weighting=weight_ew, verbose=False, **kwargs):
	"""
	Backtests a given weighting scheme, given some parameters:
	r : asset returns to use to build the portfolio
	estimation_window: the window to use to estimate parameters
	weighting: the weighting scheme to use, must be a function that takes "r", and a variable number of keyword-value arguments
	"""
	n_periods = r_true.shape[0]
	# return windows
	windows = [(start, start+estimation_window) for start in range(n_periods-estimation_window)]
	weights_true = [weighting(r_true.iloc[win[0]:win[1]], **kwargs) for win in windows]
	weights_pred = [weighting(r_pred.iloc[win[0]:win[1]], **kwargs) for win in windows]
	# convert List of weights to DataFrame
	weights_true = pd.DataFrame(weights_true, index=r_true.iloc[estimation_window:].index, columns=r_true.columns)
	weights_pred = pd.DataFrame(weights_pred, index=r_pred.iloc[estimation_window:].index, columns=r_pred.columns)
	returns_true = (weights_true * r_true).sum(axis="columns",  min_count=1) #mincount is to generate NAs if all inputs are NAs
	returns_pred = (weights_pred * r_pred).sum(axis="columns",  min_count=1)
	return returns_true, returns_pred
	

def get_portfolio(true_returns, pred_returns, pred_window = 30):

	#EW porfolio
	ew_true, ew_pred = backtest_ws(
		true_returns, pred_returns, estimation_window = pred_window, weighting = weight_ew)

	#GMV Sample Covariance portfolio
	gmv_sample_true, gmv_sample_pred = backtest_ws(
		true_returns, pred_returns, estimation_window = pred_window, weighting=weight_gmv, cov_estimator = sample_cov)

	#GMV Constant Correlation portfolio
	gmv_cc_true, gmv_cc_pred = backtest_ws(
		true_returns, pred_returns, estimation_window = pred_window, weighting=weight_gmv, cov_estimator = cc_cov)

	#GMV Shrinkage Correlation portfolio
	gmv_shrink_true, gmv_shrink_pred = backtest_ws(
		true_returns, pred_returns, estimation_window = pred_window, weighting=weight_gmv, cov_estimator = shrinkage_cov)

	return pd.DataFrame(
		{
		"EW_true": ew_true, "EW_pred": ew_pred,
		"GMV_sample_true": gmv_sample_true, "GMV_sample_pred": gmv_sample_pred,
		"GMV_cc_true": gmv_cc_true, "GMV_cc_pred": gmv_cc_pred,
		"GMV_shrinkage_true": gmv_shrink_true, "GMV_shrinkage_pred": gmv_shrink_pred
		}
		)

def get_summary_stats(true_returns, pred_returns, pred_window = 30):
	return summary_stats(get_portfolio(true_returns, pred_returns, pred_window).dropna())

def plot_portfolio(true_returns, pred_returns, pred_window = 30, savefig=False):

	rets = get_portfolio(true_returns, pred_returns, pred_window = 30)
	cum_rets = (1+rets).cumprod()

	fig, axs = plt.subplots(rets.columns // 2, 1, figsize = (30, (rets.columns// 2)*15))

	axs[0].plot(true_returns.index, cum_rets["EW_true"], label="True")
	axs[0].plot(true_returns.index, cum_rets["Ew_pred"], label="Pred")

	axs[1].plot(true_returns.index, cum_rets["GMV_sample_true"], label="True")
	axs[1].plot(true_returns.index, cum_rets["GMV_sample_pred"], label="Pred")

	axs[2].plot(true_returns.index, cum_rets["GMV_cc_true"], label="True")
	axs[2].plot(true_returns.index, cum_rets["GMV_cc_pred"], label="Pred")

	axs[3].plot(true_returns.index, cum_rets["GMV_shrinkage_true"], label="True")
	axs[3].plot(true_returns.index, cum_rets["GMV_shrinkage_pred"], label="Pred")

	plt.show()

