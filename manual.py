# -*- coding: utf-8 -*-







"""

Created on Tue Jul 30 08:55:15 2019

@author: x282066

"""





from __future__ import print_function

import numpy as np





def estrategia_manual(prices, delta=1e-4, vt=1e-3):

    # Preparamos los vectores a rellenar con la estrategia
    money, tlt, iei, movimientos, posicion, sqrs_Qt, e, state_mean, state_covariance = [], [], [], [], [], [], [], [], []
    #valores iniciales
    money.append(100000); tlt.append(0); iei.append(0)
    movimientos.append("start")
    posicion.append("cerrado")
    etfs = [prices.columns[0], prices.columns[1]]
    # Constantes y valores iniciales
    # latest_prices = np.array([-1.0, -1.0])
    #delta = 1e-4
    wt = delta / (1 - delta) * np.eye(2)
    #vt = 1e-3
    #state_mean.append( np.asarray([prices[etfs[0]][0]/prices[etfs[0]][1],0]))
    state_mean.append( np.asarray([0,20]))
    #state_covariance.append(np.zeros((2,2)))
    state_covariance.append(np.eye(2)*3)
    N=2000
    """
    Aquí voy haciendo en cada paso la actualizacion y calculo del estado y la covarianza. Sin embargo, según entiendo,
    esto ya está calculado en state_means y state_covariances. Asi q podría usar esas cantidades, o comprobar que van saliedno resultados
    muy similares (comparar state_mean con state_means y state_covariance con state_covariances)
    """
    for i in range(0, prices.shape[0]):

    #for i in range(0, 2):

        # Create the observation matrix of the latest prices

        # of TLT and the intercept value (1.0) as well as the

        # scalar value of the latest price from IEI

        F = np.asarray([prices[etfs[1]][i], 1.0]).reshape((1, 2))

        y = prices[etfs[0]][i]

        # The prior value of the states \theta_t is

        # distributed as a multivariate Gaussian with

        # mean a_t = I * theta_t-1 and variance-covariance R_t

        if i > 0:

            state_mean.append(state_mean[i-1])

            state_covariance.append(state_covariance[i-1] + wt)

        # Calculate the Kalman Filter update

        # ----------------------------------

        # Calculate prediction of new observation

        # as well as forecast error of that prediction

        yhat = np.dot(F, state_mean[i].transpose())

        #e[i] = y - yhat

        e.append(float(y-yhat))

        

        # Q_t is the variance of the prediction of

        # observations and hence \sqrt{Q_t} is the

        # standard deviation of the predictions

        Qt = np.dot(F, state_covariance[i]).dot(F.transpose()) + vt

        sqrt_Qt = np.sqrt(Qt)

        sqrs_Qt.append(float(sqrt_Qt))

        

        # The posterior value of the states \theta_t is

        # distributed as a multivariate Gaussian with mean

        # m_t and variance-covariance C_t

        # Kalman Gain

        Kt = np.dot(state_covariance[i], F.transpose()) / Qt

        

        # State update

        state_mean[i] = state_mean[i] + Kt.transpose()*e[i]

        state_covariance[i] = state_covariance[i] - Kt * np.dot(F, state_covariance[i])



        # Se entra long (si partimos de no posicion)

        # se define un burn in period

        if i>=1:

            # Se entra long (si partimos de no posicion)

            if ( (e[i] < -sqrt_Qt) and (posicion[i-1] == "cerrado") ):

                #actualizamos el numero de posiciones

                movimientos.append("EL")

                posicion.append("long")

                iei.append(iei[i-1] + N)             

                tlt.append(tlt[i-1] - round(N * state_mean[i][0][0]) )

                #actualizamos el dinero acumulado con los cambios de posición hechos

                money.append(money[i-1] -N*prices.loc[i][0] + round(N*state_mean[i][0][0])*prices.loc[i][1])

            # Se entra short (si partimos de no posicion)

            elif (e[i] > sqrt_Qt and posicion[i-1] == "cerrado"):

                movimientos.append("ES")

                posicion.append("short")

                iei.append(iei[i-1] - N)

                tlt.append(tlt[i-1] + round(N * state_mean[i][0][0]) )

                money.append( money[i-1] + N*prices.loc[i][0] - round(N*state_mean[i][0][0])*prices.loc[i][1] ) 



            # Se sale long (si estabamos long)

            elif (e[i] > -sqrt_Qt and posicion[i-1] == "long"):

                movimientos.append("SL")

                posicion.append("cerrado")

                iei.append(iei[i-1] - N)

                tlt.append(tlt[i-1] - tlt[i-1])

                money.append(money[i-1] +N*prices.loc[i][0] - abs(tlt[i-1])*prices.loc[i][1])



           # Se sale short (si estabamos short)

            elif (e[i] < sqrt_Qt and posicion[i-1] == "short"):

                movimientos.append("SS")

                posicion.append("cerrado")

                iei.append(iei[i-1] + N)

                tlt.append(tlt[i-1] - tlt[i-1])

                money.append(money[i-1] -N*prices.loc[i][0] + abs(tlt[i-1])*prices.loc[i][1])



            # No se opera

            else:

                movimientos.append("Nada")

                #Posicion  

                posicion.append(posicion[i-1])

                tlt.append( tlt[i-1])

                iei.append( iei[i-1])

                money.append(money[i-1])







    return money, tlt, iei, e, movimientos, posicion, sqrs_Qt, state_mean, state_covariance