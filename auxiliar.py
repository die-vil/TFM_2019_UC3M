# -*- coding: utf-8 -*-



"""

Created on Wed Jun 12 15:02:50 2019

@author: x282066



Warning 1: 

Ahora la funcion DataReader:

from pandas.io.data import DataReader

parece ser que no funciona porque yahoo a dejado de suportear la API. En este hilo de stackoverflow 

se dan workarounds que deberían ser suficiente:

https://stackoverflow.com/questions/44045158/python-pandas-datareader-no-longer-works-for-yahoo-finance-changed-url

https://stackoverflow.com/questions/47972667/importing-pandas-io-data



La idea para hacer este programa se extrajo de la página: 

https://www.quantstart.com/articles/Dynamic-Hedge-Ratio-Between-ETF-Pairs-Using-the-Kalman-Filter#ref-kinlay



Dynamic Hedge Ratio Between ETF Pairs Using the Kalman Filter

"""







from __future__ import print_function

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from pykalman import KalmanFilter



def visual_coint(precio1, precio2):

    (precio1/precio2).plot(figsize=(14,7)) 

    plt.axhline((precio1/precio2).mean(), color='red', linestyle='--') 

    plt.xlabel('Time')

    plt.legend(['Price Ratio', 'Mean'])

    plt.show()



def zscore(series):

    puntuacion = (series - series.mean()) / np.std(series)

    puntuacion.plot()

    plt.axhline(puntuacion.mean())

    plt.axhline(1.0, color='red')

    plt.axhline(-1.0, color='green')

    plt.show()



def draw_date_coloured_scatterplot(etfs, prices):

    """

    Create a scatterplot of the two ETF prices, which is

    coloured by the date of the price to indicate the

    changing relationship between the sets of prices

    """

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



def calc_slope_intercept_kalman(etfs, prices, original):

    """

    Utilise the Kalman Filter from the pyKalman package

    to calculate the slope and intercept of the regressed

    ETF prices.

    """

    delta = 1e-5

    trans_cov = delta / (1 - delta) * np.eye(2)

    obs_mat = np.vstack(

        [prices[etfs[0]], np.ones(prices[etfs[0]].shape)]

    ).T[:, np.newaxis]



    #Reintento redifiniendo constantes para cuadrar



    if original == "estrategia":



        kf = KalmanFilter(



            n_dim_obs=1,



            n_dim_state=2,



            initial_state_mean=np.zeros(2),



            initial_state_covariance=np.zeros((2, 2)),



            transition_matrices=np.eye(2),



            observation_matrices=obs_mat,



            observation_covariance=1e-3,



            transition_covariance=trans_cov



        )



    else:     



        kf = KalmanFilter(



            n_dim_obs=1,



            n_dim_state=2,



            initial_state_mean=np.zeros(2),



            initial_state_covariance=np.ones((2, 2)),



            transition_matrices=np.eye(2),



            observation_matrices=obs_mat,



            observation_covariance=1.0,



            transition_covariance=trans_cov



        )



    # Using kf.filter()



    state_means, state_covs = kf.filter(prices[etfs[1]].values)



    



    """



    Otra alternativa a kf.filter() es usar kf.filter_update() e ir aplicando el algoritmo 



    una vez cada vez que llega una nueva medida (online filter) en vez de aplicarlo a todo de golpe



    esto seria



    



    filtered_state_means = kf.initial_state_mean



    filtered_state_covariances = kf.initial_state_covariance



    prices.shape[0]



    



    filtered_state_means = []



    filtered_state_covs = []



    auxiliar = []



    #creamos los vectores a rellenar luego: dinero, posiciones



    for i in range(0,prices.shape[0]):



        auxiliar.append(i)



    filtered_state_means.append(auxiliar[:])



    filtered_state_covs.append(auxiliar[:])



    



    filtered_state_means[0] = np.zeros(2)



    filtered_state_covs[0] = np.ones((2, 2))



    



    for i in range(0,prices.shape[0]):



        filtered_state_means[i], filtered_state_covariances[i] = (



        kf.filter_update(filtered_state_means[i], filtered_state_covariances[i], observation = prices[i]))



    """



    return state_means, state_covs











def draw_slope_intercept_changes(prices, state_means):



    """



    Plot the slope and intercept changes from the



    Kalman Filte calculated values.



    """



    pd.DataFrame(



        dict(



            slope=state_means[:, 0],



            intercept=state_means[:, 1]



        ), index=prices.index



    ).plot(subplots=True)



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











#viene al final una idea del grafico que quiero hacer

# https://stackoverflow.com/questions/32474434/trying-to-plot-a-line-plot-on-a-bar-plot-using-matplotlib