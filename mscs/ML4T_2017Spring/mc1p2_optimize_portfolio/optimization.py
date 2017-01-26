"""MC1-P2: Optimize a portfolio."""


import pandas as pd
import matplotlib.pyplot as plt
#import scipy.optimize as spo
import numpy as np
import datetime as dt
import scipy.optimize as spo
from util import get_data, plot_data


def normalize(prices):
    '''
    Normalize values from stock prices.
    '''
    prices.fillna(axis=1, method='ffill', inplace=True)
    prices.fillna(axis=1, method='bfill', inplace=True)
    value = prices / prices.iloc[0]
    return value

def get_stats(values):
    '''
    Return mean and covariance matrix of ret.
    '''
    ret = (values / values.shift(1) - 1)[1:]
    cov = ret.cov().values
    mean = ret.mean().values
    return mean, cov

def get_perf(values, allocs, rfr, n):
    '''
    Get performance metrics for a portfolio.
    '''
    value = (values * allocs).sum(axis=1)
    ret = (value / value.shift(1) - 1)[1:]
    adr = ret.mean()
    sddr = ret.std()
    sr = (adr - rfr) * np.sqrt(n) / sddr
    cr = value.iloc[-1] / value.iloc[0] - 1
    return sr, adr, sddr, cr, value

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)[syms+['SPY']]  # automatically adds SPY
    values = normalize(prices_all)
    value_SPY = values['SPY']  # only SPY, for comparison later
    values = values[syms]  # only portfolio symbols
    # Optimize portfolio allocations.
    mean, cov = get_stats(values)
    stock_num = len(syms)
    x = np.ones((stock_num,1)) / stock_num
    bnds = tuple([(0,None)] * stock_num)
    #obj = lambda x: np.matmul(x.T, np.matmul(cov, x))
    #cons = ({'type': 'eq', 'fun': lambda x:  1 - np.matmul(x.T,mean)})
    def obj(x):
        value = (values * x).sum(axis=1)
        ret = (value / value.shift(1) - 1)[1:]
        return - ret.mean() * np.sqrt(252) / ret.std()
    cons = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})
    result = spo.minimize(obj, x, method='SLSQP',bounds=bnds, constraints=cons)
    x = result['x']
    allocs = x / sum(x)
    #allocs = [  5.38105153e-16,   3.96661695e-01,   6.03338305e-01,  -5.42000166e-17]
    # Get the performance of the portfolio.
    sr, adr, sddr, cr, value = get_perf(values, allocs, 0, 252)
    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        compare = pd.concat([value, value_SPY], keys=['Portfolio', 'SPY'], axis=1)
        ax = compare.plot(title="Daily portfolio value and SPY", label='Portfolio')
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized price")
        plt.legend(loc='upper left')
        plt.grid()
        plt.savefig('plot.png')
        pass
    return allocs, cr, adr, sddr, sr

def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are available to your function/code
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    #start_date = dt.datetime(2010,1,1)
    #end_date = dt.datetime(2010,12,31)
    #symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']

    #start_date = dt.datetime(2004,1,1)
    #end_date = dt.datetime(2006,1,1)
    #symbols = ['AXP', 'HPQ', 'IBM', 'HNZ']

    #start_date = dt.datetime(2004,12,1)
    #end_date = dt.datetime(2006,05,31)
    #symbols = ['YHOO', 'XOM', 'GLD', 'HNZ']

    start_date = dt.datetime(2005,12,1)
    end_date = dt.datetime(2006,5,31)
    symbols = ['YHOO', 'HPQ', 'GLD', 'HNZ']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = True)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
