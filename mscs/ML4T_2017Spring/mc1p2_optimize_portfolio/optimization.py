"""MC1-P2: Optimize a portfolio."""


import pandas as pd
import matplotlib.pyplot as plt
#import scipy.optimize as spo
import numpy as np
import datetime as dt
from util import get_data, plot_data


def obj(allocs, mean, corr, rfr):
    tmp = np.dot(allocs, mean) - rfr
    return - tmp * tmp / (allocs.transpose() * corr * allocs)

def normalize(prices):
    '''
    Normalize values from stock prices.
    '''
    prices.fillna(axis=1, method='bfill', inplace=True)
    prices.fillna(axis=1, method='ffill', inplace=True)
    value = prices / prices.iloc[0]
    return value

def get_stats(values):
    '''
    Return mean and correlation matrix of ret.
    '''
    values = (values / values.shift(1) - 1)[1:]
    corr = values.corr().values
    mean = values.mean().values
    return mean, corr

def get_perf(values, allocs, rfr, n):
    '''
    Get performance metrics for a portfolio.
    '''
    value = values * allocs
    adr = value.mean()
    sddr = value.std()
    sr = (mean - rfr) * sqrt(n) / std
    cr = value.iloc[-1]
    return sr, adr, sddr, cr, value

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)[syms+['SPY']]  # automatically adds SPY
    values = normalize(prices_all)
    values_SPY = values_SPY['SPY']  # only SPY, for comparison later
    values = values[syms]  # only portfolio symbols
    # Optimize portfolio allocations.
    mean, corr = get_stats(values)
    stock_num = len(syms)
    allocs = np.ones((stock_num,1)) / stock_num
    cons = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})
    bnds = tuple([(0,1)] * stock_num)
    result = spo.minimize(obj, allocs, method=‘SLSQP’, bounds=bnds, constraints=cons)
    allocs = result['x']
    # Get the performance of the portfolio.
    sr, adr, sddr, cr, value = get_perf(values, allocs, rfr, n)
    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        compare = pd.concat([value, value_SPY], keys=['Portfolio', 'SPY'], axis=1)
        ax = compare.plot(title="Daily portfolio value and SPY", label='Portfolio')
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized price")
        plt.legend(loc='upper left')
        plt.grid()
        plt.show()
        pass
    return allocs, cr, adr, sddr, sr

def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are available to your function/code
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    start_date = dt.datetime(2009,1,1)
    end_date = dt.datetime(2010,1,1)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM', 'IBM']

sd=dt.datetime(2008,1,1)
ed=dt.datetime(2009,1,1)
syms=['GOOG','AAPL','GLD','XOM']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = False)

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
