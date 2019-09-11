# -*- coding: utf-8 -*-


from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pykalman import KalmanFilter


# Function to illustrate visually the cointegration between both assets based on a simple linear regression: hedge ratio
def visual_coint(precio1, precio2):
    (precio1/precio2).plot(figsize=(14,7)) 
    plt.axhline((precio1/precio2).mean(), color='red', linestyle='--') 
    plt.xlabel('Time')
    plt.legend(['Price Ratio', 'Mean'])
    plt.show()


# Construction of the zscore using the spread previously calculated
def zscore(series):
    puntuacion = (series - series.mean()) / np.std(series)
    puntuacion.plot()
    plt.axhline(puntuacion.mean())
    plt.axhline(1.0, color='red')
    plt.axhline(-1.0, color='green')
    plt.show()


def draw_date_coloured_scatterplot(etfs, prices):
    # Create a yellow-to-red colourmap where yellow indicates
    # early dates and red indicates later dates
    plen = len(prices)
    colour_map = plt.cm.get_cmap('YlOrRd')
    colours = np.linspace(0.1, 1, plen)
    # Create the scatterplot object
    scatterplot = plt.scatter(
        prices[etfs[0]], prices[etfs[1]],
        s=30, c=colours, cmap=colour_map,
        edgecolor='k', alpha=0.8
    )

    # Add a colour bar for the date colouring and set the
    # corresponding axis tick labels to equal string-formatted dates
    colourbar = plt.colorbar(scatterplot)
    colourbar.ax.set_yticklabels(
        [str(p.date()) for p in prices[::plen//9].index]
    )
    plt.xlabel(prices.columns[0])
    plt.ylabel(prices.columns[1])
    plt.show()




# Get the cointegration
#NOTE CRITICAL LEVEL HAS BEEN SET TO 5% FOR COINTEGRATION TEST
def find_cointegrated_pairs(dataframe, critial_level = 0.05):
    n = dataframe.shape[1] # the length of dateframe
    pvalue_matrix = np.ones((n, n)) # initialize the matrix of p
    keys = dataframe.columns # get the column names
    pairs = [] # initilize the list for cointegration
    for i in range(n):
        for j in range(i+1, n): # for j bigger than i
            stock1 = dataframe[keys[i]] # obtain the price of "stock1"
            stock2 = dataframe[keys[j]]# obtain the price of "stock2"
            result = sm.tsa.stattools.coint(stock1, stock2) # get conintegration
            pvalue = result[1] # get the pvalue
            pvalue_matrix[i, j] = pvalue
            if pvalue < critial_level: # if p-value less than the critical level
                pairs.append((keys[i], keys[j], pvalue)) # record the contract with that p-value
    return pvalue_matrix, 

