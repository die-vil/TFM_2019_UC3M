
from __future__ import print_function
import numpy as np
from sklearn import linear_model
import pandas as pd

def estrategia_basica_2(prices, prices_train):
    # Preparamos los vectores a rellenar con la estrategia
    money, tlt, iei, movimientos, posicion, std_list, e, state_mean, diferencias = [], [], [], [], [], [], [], [], []
    # Valores iniciales
    money.append(100000); tlt.append(0); iei.append(0)
    movimientos.append("start")
    posicion.append("cerrado")

    N=2000
    period = 100

    etfs = [prices.columns[0], prices.columns[1]]
    datos = pd.concat([prices_train, prices]).reset_index(drop = True)
    """
    En est método se utiliza una rolling regression
    """

    """
    Primero se calcula la desviacion estandar con datos históricos que es lo que se va a usar como señal de trading
    """
    for i in range(0, prices_train.shape[0]):
        datos_entrenamiento = prices_train.loc[i -min(i,period)  : i -1]
        rolling_ratio = (datos_entrenamiento[etfs[0]] / datos_entrenamiento[etfs[1]]).mean()
        y = prices_train[etfs[0]][i]
        # Calculate prediction of new observation
        yhat = rolling_ratio*prices_train[etfs[1]].values[i]
        diferencias.append(float(y-yhat))        

    """
    Estrategia
    """
    e = diferencias + e
    for i in range(0, prices.shape[0]):
        # Se crea el dataset sobre el que se entrena (rolling 60 step window back)
        datos_entrenamiento = datos.loc[i + len(prices_train) -period -1 : len(prices_train) + i -1]
        # Se ajusta la regresión lineal a esos datos
        rolling_ratio = (datos_entrenamiento[etfs[0]] / datos_entrenamiento[etfs[1]]).mean()
        state_mean.append(rolling_ratio)
        y = prices[etfs[0]][i]
        # Calculate prediction of new observation
        yhat = rolling_ratio*prices[etfs[1]].values[i]
        e.append(float(y-yhat))

        # Se calcula la desviacion con el 60-window como señal de trading
        std = np.std(np.asarray( e[ i - period + len(diferencias) : i - 1 + len(diferencias) ]))
        std_list.append(std)
        # Se entra long (si partimos de no posicion)
        # se define un burn in period
        if i>=1:
            # Se entra long (si partimos de no posicion)
            if ( (e[i] < - std) and (posicion[i-1] == "cerrado") ):
                #actualizamos el numero de posiciones
                movimientos.append("EL")
                posicion.append("long")
                iei.append(iei[i-1] + N)             
                tlt.append(tlt[i-1] - round(N * rolling_ratio) )
                #actualizamos el dinero acumulado con los cambios de posición hechos
                money.append(money[i-1] -N*prices.loc[i][0] + round(N*rolling_ratio)*prices.loc[i][1])
            # Se entra short (si partimos de no posicion)
            elif (e[i] > std and posicion[i-1] == "cerrado"):
                movimientos.append("ES")
                posicion.append("short")
                iei.append(iei[i-1] - N)
                tlt.append(tlt[i-1] + round(N * rolling_ratio) )
                money.append( money[i-1] + N*prices.loc[i][0] - round(N*rolling_ratio)*prices.loc[i][1] )

            # Se sale long (si estabamos long)
            elif (e[i] > -std and posicion[i-1] == "long"):
                movimientos.append("SL")
                posicion.append("cerrado")
                iei.append(iei[i-1] - N)
                tlt.append(tlt[i-1] - tlt[i-1])
                money.append(money[i-1] +N*prices.loc[i][0] - abs(tlt[i-1])*prices.loc[i][1])
           # Se sale short (si estabamos short)
            elif (e[i] < std and posicion[i-1] == "short"):
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

    e_2 = e[len(diferencias):]
    return money, tlt, iei, e, movimientos, posicion, std_list, state_mean, e_2
