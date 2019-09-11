# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:11:42 2019
@author: x282066
"""

from __future__ import print_function
from pandas_datareader import data as wb
from auxiliar import zscore, visual_coint, draw_date_coloured_scatterplot, calc_slope_intercept_kalman, draw_slope_intercept_changes
from metricas import metricas, metrica_comparada
#from alternativa import estrategia_alternativa
from manual import estrategia_manual
#from base_strat import estrategia_basica
from base_strat2 import estrategia_basica_2
import pandas as pd
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm


def ejecucion(casa, prices, start_train_date, start_date, end_date, ticker='GOOGL', data_source='yahoo', etfs = ['IEI', 'TLT']):
    if type(prices) == int:
        # Se descargan los datos de precio que se van a usar para el training y para el backtest de la estrategia
        ticker_data = wb.DataReader(etfs, data_source=data_source, start=start_train_date, end=end_date) #portatil personal
        prices = ticker_data['Adj Close']
    # Se divide y preparan los datos
    prices_train = prices.loc[start_train_date:start_date]
    prices_train = prices_train.drop(prices_train.index[-1], axis = 0)
    prices_test = prices.loc[start_date:end_date]
    # Un par de plots del train set para ver que aspecto tienen los datos
    draw_date_coloured_scatterplot(etfs, prices_train)
    zscore(prices_train[etfs[0]]/prices_train[etfs[1]])
    prices_test.index = range(len(prices_test.index))
    prices_train.index = range(len(prices_train.index))
    # Se implementa el trading: primero la estrategia benchmark y luego la estrategia con el filtro de kalman
    money_b2, tlt_b2, iei_b2, e_b2, movimientos_b2, posicion_b2, sqrs_Qt_b2, state_mean_b2, e_2  = estrategia_basica_2(prices_test, prices_train)
    money_m, tlt_m, iei_m, e_m, movimientos_m, posicion_m, sqrs_Qt_m, state_mean_m, e_m  = estrategia_manual(prices_test, delta=1e-2, vt=1e-5)
    performance, daily_returns = metricas(money_b2, prices_test, movimientos_b2,'si', 'si' )    
    spread(e_2, sqrs_Qt_b2)
    money_m, tlt_m, iei_m, e_m, movimientos_m, posicion_m, sqrs_Qt_m, state_mean_m, state_cov = estrategia_manual(prices_test, delta=1e-4, vt=1e-3)
    spread(e_m, sqrs_Qt_m)
    performance, daily_returns = metricas(money_m, prices_test, movimientos_m,'si', 'si' )
    # Se evalua la performance de ambas técnicas
    metrica_comparada(money_m, prices_test, movimientos_m, money_b2, movimientos_b2, 'si')
    if type(prices) == int:
        return prices, prices_train, prices_test


"""
=========================================================================================
"""
# Choose the ETF symbols to work with along with
# start and end dates for the price histories
start_date = "2010-8-01"
start_train_date = "2007-8-01"
end_date = "2016-08-01"
ticker='GOOGL'
data_source='yahoo'

# Code to scrap the companies of the SP500 tickers
data = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
table = data[0]
table.head()
sliced_table = table[1:]
#header = table.iloc[0]
#corrected_table = sliced_table.rename(columns=header)
#corrected_table
#tickers = corrected_table['Ticker symbol'].tolist()
tickers = sliced_table['Symbol'].tolist()
sectors = sliced_table['GICS Sector'].unique().tolist()

# Code to scrap companies by sector
sector_number = 
tickers_sector1 = sliced_table[sliced_table['GICS Sector']==sectors[sector_number]]['Symbol'].tolist()
#set empty list o hold the stock price DataFrames that we can later concatenate into a master frame
df_list = []
#not all stocks will return data so set up an empty list to store the stock tickers that actually successfully returns data
used_stocks = []
for stock in tickers_sector1:
    try:
        data = pd.DataFrame(wb.DataReader(stock, data_source=data_source, start=start_train_date, end=end_date)['Adj Close'])
        print(stock)
        data.columns = [stock]
        df_list.append(data)
        used_stocks.append(stock)
    except:
        print(stock, ' falló')
        pass
#concatenate list of individual tciker price DataFrames into one master DataFrame
df = pd.concat(df_list,axis=1)
df.isnull().sum()
for column in df.columns:
    if df[column].isnull().sum() > 20:
        df.drop(column, axis=1, inplace=True)
    else:
        df[column].fillna(method='ffill')
df.isnull().sum()
df.to_pickle('consumer_staples_9.pkl') 
df = pd.read_pickle('information_technology_1.pkl')

#df = pd.read_pickle('health_care_0.pkl')
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
    return pvalue_matrix, pairs



archivos = ['health_care_0.pkl', 'information_technology_1.pkl','communication_services_2.pkl',
            'consumer_discretionary_3.pkl','utilities_4.pkl','financials_5.pkl',
            'materials_6.pkl','industrials_7.pkl','real_estate_8.pkl','consumer_staples_9.pkl',
            'energy_10.pkl']
i=0
numeritos = []
for archivo in archivos:
    df = pd.read_pickle(archivo)
    if archivo == 'energy_10.pkl':
        df.drop('CXO', axis=1, inplace=True)
    pvalue_matrix,pairs = find_cointegrated_pairs(df[:start_date])
    #convert our matrix of stored results into a DataFrame
    pvalue_matrix_df = pd.DataFrame(pvalue_matrix)
    #use Seaborn to plot a heatmap of our results matrix
    #fig, ax = plt.subplots(figsize=(15,10))
    #sns.heatmap(pvalue_matrix_df,xticklabels=used_stocks,yticklabels=used_stocks,ax=ax)
    for pair in pairs:
        print("Stock {} and stock {} has a co-integration score of {}".format(pair[0],pair[1],round(pair[2],4)))
    exitos = []
    n_exitos = 0
    total_return = 0
    fracasos = []
    n_fracasos = 0
    performance_total_exitos = pd.DataFrame(index = range(0,10), columns=['Resultado'])
    performance_total_exitos = performance_total_exitos.fillna(0)
    performance_total_fracasos = pd.DataFrame(index = range(0,10), columns=['Resultado'])
    performance_total_fracasos = performance_total_fracasos.fillna(0)
    for pair in pairs:
        etfs = [pair[0], pair[1]]
        prices = df[etfs]
        prices_train = prices.loc[start_train_date:start_date]
        prices_train = prices_train.drop(prices_train.index[-1], axis = 0)
        prices_test = prices.loc[start_date:end_date]
        prices_test.index = range(len(prices_test.index))
        prices_train.index = range(len(prices_train.index))
        money_m, tlt_m, iei_m, e_m, movimientos_m, posicion_m, sqrs_Qt_m, state_mean_m, state_cov = estrategia_manual(prices_test, delta=1e-4, vt=1e-1)
        #money_m, tlt_m, iei_m, e_m, movimientos_m, posicion_m, sqrs_Qt_m, state_mean_m, state_cov = estrategia_basica_2(prices_test, prices_train)
        print(etfs)
        try:
            performance, daily_returns = metricas(money_m, prices_test, movimientos_m,'no', 'si' )
            performance.drop('Métrica', axis=1, inplace=True)
            Total_Return = (money_m[-1]) /money_m[0]
            if Total_Return > 1:
                #print(Total_Return)
                exitos.append([pair[0],pair[1],Total_Return ])
                performance_total_exitos = performance_total_exitos.add(performance)
                total_return += Total_Return
                n_exitos += 1
            else:
                fracasos.append([pair[0],pair[1],Total_Return ])
                performance_total_fracasos = performance_total_fracasos.add(performance)
                n_fracasos += 1 
        except:
            pass
    print('Total pares: ', len(pairs))
    print('N éxitos: ' ,n_exitos)
    print('N fracasos: ' ,n_fracasos)
    performance_total = performance_total_exitos.add(performance_total_fracasos)
    performance_total_promedio = performance_total/len(pairs)
    performance_total['Métrica'] = métricas
    performance_total_promedio['Métrica'] = métricas
    performance_total_exitos_promedio = performance_total_exitos/n_exitos
    performance_total_fracasos_promedio = performance_total_fracasos/n_fracasos
    performance_total_exitos['Métrica'] = métricas
    performance_total_fracasos['Métrica'] = métricas
    performance_total_exitos_promedio['Métrica'] = métricas
    performance_total_fracasos_promedio['Métrica'] = métricas
    numeritos.append([len(pairs), n_exitos,n_fracasos])
    print('performance total')
    print(performance_total)
    print('performance total promedio')
    print(performance_total_promedio)
    print('performance exito promedio')
    print(performance_total_exitos_promedio)
    print('performance fracaso promedio')
    print(performance_total_fracasos_promedio)
    
    
    with pd.ExcelWriter('resultados_v1/output_'+str(i)+'.xlsx') as writer:  # doctest: +SKIP
        performance_total.to_excel(writer, sheet_name='total')
        performance_total_promedio.to_excel(writer, sheet_name='total_promedio')
        performance_total_exitos_promedio.to_excel(writer, sheet_name='total_exitos_promedio')
        performance_total_fracasos_promedio.to_excel(writer, sheet_name='total_fracasos_promedio')
    i+=1


"""
=========================================================================================
"""

ticker_data = wb.DataReader(etfs, data_source=data_source, start=start_train_date, end=end_date) #portatil personal
prices = ticker_data['Adj Close']
prices = df[etfs]

plt.plot(prices[etfs[0]], 'r-', label = etfs[0])
plt.plot(prices[etfs[1]], 'b-',  label = etfs[1])
plt.title('Price evolution of two cointegrated pairs')
plt.xlabel('Trading Day')
plt.ylabel('Price')
plt.legend(loc= "best")
plt.show()

plt.plot(prices[etfs[1]] - prices[etfs[0]], 'b-',  label = 'Spread')
plt.axhline((prices[etfs[1]] - prices[etfs[0]]).mean(), color='red', linestyle='--', label = 'Mean') 
plt.title('Price evolution of the spread')
plt.xlabel('Trading Day')
plt.ylabel('Price')
plt.legend(loc = 'best')
plt.show()


puntuacion = ((prices[etfs[1]] - prices[etfs[0]]) - (prices[etfs[1]] - prices[etfs[0]]).mean()) / np.std((prices[etfs[1]] - prices[etfs[0]]))
puntuacion.plot()
plt.axhline(puntuacion.mean())
plt.axhline(1.0, color='red')
plt.axhline(-1.0, color='green')
plt.title('Zscore of the spread')
plt.show()

prices_train = prices.loc[start_train_date:start_date]
prices_train = prices_train.drop(prices_train.index[-1], axis = 0)
prices_test = prices.loc[start_date:end_date]

visual_coint(prices_train[etfs[0]],prices_train[etfs[1]])
visual_coint(prices_test[etfs[0]],prices_test[etfs[1]])

result = ts.coint(prices_train[etfs[0]],prices_train[etfs[1]]) # get conintegration
print(result)
pvalue = result[1] # get the pvalue
print(pvalue)


ganado = 0   
for exito in exitos:
    ganado += (exito[2]-1)
print('Dinero ganado: ', 100000*ganado)
perdido = 0   
for pifia in fracasos:
    perdido += (pifia[2]-1)
print('Dinero perdido: ', 100000*perdido)
print('Profit: ' , 100000*ganado + 100000*perdido)

prices = df[etfs]
prices_train = prices.loc[start_train_date:start_date]
prices_train = prices_train.drop(prices_train.index[-1], axis = 0)
prices_test = prices.loc[start_date:end_date]
prices_test.index = range(len(prices_test.index))
prices_train.index = range(len(prices_train.index))
# Se implementa el trading: primero la estrategia benchmark y luego la estrategia con el filtro de kalman
money_b2, tlt_b2, iei_b2, e_b2, movimientos_b2, posicion_b2, sqrs_Qt_b2, state_mean_b2, e_2  = estrategia_basica_2(prices_test, prices_train)
spread(e_2, sqrs_Qt_b2)
money_m, tlt_m, iei_m, e_m, movimientos_m, posicion_m, sqrs_Qt_m, state_mean_m, state_cov = estrategia_manual(prices_test)
spread(e_m, sqrs_Qt_m)
# Se evalua la performance de ambas técnicas
metrica_comparada(money_m, prices_test, movimientos_m, money_b2, movimientos_b2, 'si')




money_m, tlt_m, iei_m, e_m, movimientos_m, posicion_m, sqrs_Qt_m, state_mean_m, state_cov = estrategia_manual(prices_test)
daily_returns, Total_Return = returns(money_m, 1, 'si')
Total_Return > 0.5
ejecucion('si',prices, start_train_date, start_date, end_date, ticker='GOOGL', data_source='yahoo', etfs = etfs)
