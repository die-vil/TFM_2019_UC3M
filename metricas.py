# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:25:57 2019
@author: x282066
"""

from __future__ import print_function
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np 
import pandas as pd

'''
============================================================================================================================
Cálculo métricas de perfromance
============================================================================================================================
'''

def metricas(money_m, prices, movimientos_m, plot, casa): 
    # initialize list of lists 
    daily_returns, Total_Return = returns(money_m,plot,1,casa)
    mean_return, volatility = annual_volatility(daily_returns)
    mean_returns, volatilities = rolling_volatility(daily_returns,plot,1, casa)
    #SR = sharpe_ratio(mean_return, volatility)
    SR = sharpe_ratio_true(money_m)
    sharpe_ratios = rolling_sharpe_ratio(mean_returns, volatilities,plot,1,1,casa)
    SOR_RAT = sortino_ratio(daily_returns,1)
    Max_Drawdown_Abs, Max_Drawdown, Duration, position_min, position_max, time_to_recover = max_drawdown2(money_m,plot, 1,casa)
    window, Daily_drawdown = daily_drawdown(money_m, position_min, prices, plot, casa)
    #MAX_DRAW, position_min, position_max = max_drawdown(money_m)
    profit_long_trades, profit_short_trades, WL_RATIO  = win_loss_ratio(movimientos_m, money_m,1,1)
    profitabilitY, n_profitable_trades, n_nonprofit_trades, number_trades = profitability(profit_long_trades,profit_short_trades,1,1)
    data = [['Total Return', Total_Return], ['Annual Volatility', np.asarray(money_m).std()], 
            ['Sharpe Ratio', SR], ['Sortino Ratio', SOR_RAT], ['Max Drawdown', Max_Drawdown], ['Max Drawdown Duration', Duration ], 
            ['Max Drawdown recover time', time_to_recover], ['Win/Loss Ratio', WL_RATIO],['Number trades', number_trades], ['Profitability', profitabilitY]] 
    # Create the pandas DataFrame 
    df = pd.DataFrame(data, columns = ['Métrica', 'Resultado']) 
    if plot == 'si':
        print(df)
    return df, daily_returns


def metrica_comparada(money_m, prices, movimientos_m, money_b2, movimientos_b2, casa): 
    # Metricas de la primera estrategia
    daily_returns_m, Total_Return_m, daily_returns_b2, Total_Return_b2 = returns(money_m,'si', money_b2,casa)
    mean_return_m, volatility_m, mean_return_b2, volatility_b2 = annual_volatility(daily_returns_m, daily_returns_b2)
    mean_returns_m, volatilities_m, mean_returns_b2, volatilities_b2  = rolling_volatility(daily_returns_m, 'si',daily_returns_b2, casa)
    SR_m, SR_b2 = sharpe_ratio(mean_return_m, volatility_m, mean_return_b2, volatility_b2)
    sharpe_ratios_m, sharpe_ratios_b2 = rolling_sharpe_ratio(mean_returns_m, volatilities_m, 'si',mean_returns_b2, volatilities_b2,casa)
    SOR_RAT_m, SOR_RAT_b2 = sortino_ratio(daily_returns_m, daily_returns_b2)
    Max_Drawdown_Abs_m, Max_Drawdown_m, Duration_m, position_min_m, position_max_m, time_to_recover_m, Max_Drawdown_Abs_b2, Max_Drawdown_b2, Duration_b2, position_min_b2, position_max_b2, time_to_recover_b2 = max_drawdown2(money_m, 'si', money_b2,casa)
    window_m, daily_drawdown1 = daily_drawdown(money_m, position_min_m, prices,'si',casa)
    window_b2, daily_drawdown2 = daily_drawdown(money_b2, position_min_b2, prices,'si',casa)
    plt.plot(daily_drawdown1, label = 'Kalman Filter' )
    plt.plot(daily_drawdown2, label = 'Rolling regression' )
    plt.plot(position_min_m, daily_drawdown1.loc[position_min_m][0], 'o', color='Red', markersize=10)
    plt.plot(position_min_b2, daily_drawdown2.loc[position_min_b2][0], 'o', color='Red', markersize=10)
    plt.title('Daily Maximum Drawdown')
    plt.xlabel('Trading Day')
    plt.ylabel('Drawdown')
    plt.legend(loc= "best")
    if casa!='si':
        plt.savefig(r'C:\Users\x282066\OneDrive - Santander Office 365\Desktop\imagenes\daily_drawdown_mixed.png', bbox_inches='tight')
    else:
        plt.savefig(r'C:\Users\diego\Google Drive\UNIVERSIDAD\MASTER\uc3m\TFM\ultimo estado\ultimoestadoseguridad\imagenes\daily_drawdown_mixed.png', bbox_inches='tight')
    plt.show()
    profit_long_trades_m, profit_short_trades_m, WL_RATIO_m , profit_long_trades_b2, profit_short_trades_b2, WL_RATIO_b2 = win_loss_ratio(movimientos_m, money_m, movimientos_b2, money_b2)
    profitabilitY_m, n_profitable_trades_m, n_nonprofit_trades_m, number_trades_m, profitabilitY_b2, n_profitable_trades_b2, n_nonprofit_trades_b2, number_trades_b2 = profitability(profit_long_trades_m,profit_short_trades_m, profit_long_trades_b2, profit_short_trades_b2)
    data = [['Total Return', Total_Return_m -1, Total_Return_b2 -1], ['Annual Volatility', volatility_m, volatility_b2], 
            ['Sharpe Ratio', SR_m, SR_b2], ['Sortino Ratio', SOR_RAT_m, SOR_RAT_b2], ['Max Drawdown', str(Max_Drawdown_m)  + '%', str(Max_Drawdown_b2)  + '%'], ['Max Drawdown Duration', Duration_m , Duration_b2], 
            ['Max Drawdown recover time', time_to_recover_m, time_to_recover_b2], ['Win/Loss Ratio', WL_RATIO_m, WL_RATIO_b2],['Number trades', number_trades_m, number_trades_b2], ['Profitability', profitabilitY_m, profitabilitY_b2]] 
    # Create the pandas DataFrame 
    df = pd.DataFrame(data, columns = ['Métrica', 'Estrategia 1', 'Estrategia 2']) 
    print(df)

    
def spread(spread, variance):
    spread = np.asarray(spread[3:])
    variance = np.asarray(variance[3:])
    plt.plot(spread, 'b-', label = 'Spread')
    plt.axhline(spread.mean(), color='red', linestyle='--', label = 'Mean Spread') 
    plt.plot(variance, 'g-', label = 'Variance')
    plt.title('Spread/Variance Evolution')
    plt.xlabel('Trading Day')
    plt.legend(loc= "best")
    plt.savefig(r'C:\Users\diego\Google Drive\UNIVERSIDAD\MASTER\uc3m\TFM\ultimo estado\ultimoestadoseguridad\imagenes\spread.png', bbox_inches='tight')
    plt.show()
    
    plt.plot(spread/variance, 'b-', label = 'Spread')
    plt.axhline(1, color='red', linestyle='--') 
    plt.axhline(-1, color='red', linestyle='--') 
    plt.title('Spread Z-score Evolution')
    plt.xlabel('Trading Day')
    plt.savefig(r'C:\Users\diego\Google Drive\UNIVERSIDAD\MASTER\uc3m\TFM\ultimo estado\ultimoestadoseguridad\imagenes\zscore.png', bbox_inches='tight')
    plt.show()

def returns(money,plot, money2=1, casa='si'):
    daily_returns, accumulated_returns = [], []
    for i in range(1,len(money)-1):
        daily_returns.append((money[i+1]-money[i]) / money[i])
        accumulated_returns.append((money[i+1]-money[0]) / money[0])
    daily_returns = [x for x in daily_returns]
    accumulated_returns = [x for x in accumulated_returns]
    Total_Return = (money[-1]) /money[0]

    if money2 != 1:
        daily_returns2, accumulated_returns2 = [], []
        for i in range(1,len(money2)-1):
            daily_returns2.append((money2[i+1]-money2[i]) / money2[i])
            accumulated_returns2.append((money2[i+1]-money2[0]) / money2[0])
        daily_returns2 = [x for x in daily_returns2]
        accumulated_returns2 = [x for x in accumulated_returns2]
        Total_Return2 = (money2[-1]) /money2[0]
    if plot == 'si':
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax2.plot(money, 'g-', label='Kalman Filter')
        if money2 != 1:
            ax2.plot(money2, 'b-', label='Rolling regression')
        ax2.set_xlabel('Trading Day')
        #ax1.set_ylabel('Daily returns %', color='g')
        ax2.set_ylabel('Cumulative money €', color='b')
        plt.title('Performance of the strategy - Equity curve')
        plt.legend(loc= "best")
        if casa != 'si':
            plt.savefig(r'C:\Users\x282066\OneDrive - Santander Office 365\Desktop\imagenes\returns.png', bbox_inches='tight')
        else:
            plt.savefig(r'C:\Users\diego\Google Drive\UNIVERSIDAD\MASTER\uc3m\TFM\ultimo estado\ultimoestadoseguridad\imagenes\returns.png', bbox_inches='tight')
        plt.show()
        plt.plot(daily_returns, label = 'Kalman Filter')
        if money2 != 1:
            plt.plot(daily_returns2, label = 'Rolling Regression')
        plt.title('Daily Returns')
        plt.xlabel('Trading Day')
        plt.legend(loc= "best")
        if casa != 'si':
            plt.savefig(r'C:\Users\x282066\OneDrive - Santander Office 365\Desktop\imagenes\daily_returns.png', bbox_inches='tight')
        else:
            plt.savefig(r'C:\Users\diego\Google Drive\UNIVERSIDAD\MASTER\uc3m\TFM\ultimo estado\ultimoestadoseguridad\imagenes\daily_returns.png', bbox_inches='tight')
        plt.show()
    if money2 != 1:
        return daily_returns, Total_Return, daily_returns2, Total_Return2
    else: 
        return daily_returns, Total_Return

# Annualized volatility
# The standard deviation of daily returns of the model in a year. Volatility is used as a measure of risk, therefore higher vol implies riskier model.
def annual_volatility(returns, returns2=1):
    mean_return = sum(returns)/len(returns)
    volatility = sqrt(sum( (x - mean_return)**2 for x in returns) / len(returns))
    if returns2 !=1:
        mean_return2 = sum(returns2)/len(returns2)
        volatility2 = sqrt(sum( (x - mean_return2)**2 for x in returns2) / len(returns2))
        return mean_return, volatility, mean_return2, volatility2
    else: 
        return mean_return, volatility
    
def sharpe_ratio_true(money_m):
    return (money_m[-1] - money_m[0])/np.asarray(money_m).std()
# as a rolling period: 
def rolling_volatility(returns, plot, returns2=1, casa='si'):
    mean_returns, volatilities, mean_returns2, volatilities2 = [], [], [], []
    for i in range(0, len(returns)-30):
        mean_returns.append( sum( x for x in returns[i-min(i,30):i] ) / (i+1) )
        volatilities.append( sqrt(sum( (x - mean_returns[i])**2 for x in returns[i-min(i,30):i+1]) / (min(i+1,30))))
    if returns2 != 1:
        for i in range(0, len(returns)-30):
            mean_returns2.append( sum( x for x in returns2[i-min(i,30):i] ) / (i+1) )
            volatilities2.append( sqrt(sum( (x - mean_returns2[i])**2 for x in returns2[i-min(i,30):i+1]) / (min(i+1,30))))
    if plot == 'si': 
        fig, ax1 = plt.subplots()
        ax1.plot(volatilities, 'b-', label='Kalman Filter')
        if returns2 !=1:
            ax1.plot(volatilities2, 'g-', label='Rolling Regression')
        ax1.set_xlabel('Trading Day')
        ax1.set_ylabel('Volatility')
        plt.title('1-Month Rolling Volatility')
        plt.legend(loc= "best")
        if casa != 'si':
            plt.savefig(r'C:\Users\x282066\OneDrive - Santander Office 365\Desktop\imagenes\rolling_returns.png', bbox_inches='tight')
        else:
            plt.savefig(r'C:\Users\diego\Google Drive\UNIVERSIDAD\MASTER\uc3m\TFM\ultimo estado\ultimoestadoseguridad\imagenes\rolling_returns.png', bbox_inches='tight')
        plt.show() 
    if returns2 != 1:
        return mean_returns, volatilities, mean_returns2, volatilities2
    else: 
        return mean_returns, volatilities

# Sharpe Ratio
# The reward/risk ratio or risk adjusted returns of the strategy, calculated as Annualized Return/Annualized Volatility
def sharpe_ratio(mean_return, volatility, mean_return2 = 1, volatility2 = 1):
    sharpe_ratio = mean_return / volatility * sqrt(252)
    if mean_return2 != 1:
        sharpe_ratio2 = mean_return2 / volatility2 * sqrt(252)   
        return sharpe_ratio, sharpe_ratio2
    else:
        return sharpe_ratio

# as a rolling period: 
def rolling_sharpe_ratio(mean_returns, volatilities,plot, mean_returns2=1, volatilities2=1, casa='si'):
    sharpe_ratios = []
    for i in range(0,len(mean_returns)):
        if volatilities[i]>0:
            sharpe_ratios.append(mean_returns[i]/volatilities[i]*sqrt(252))
        if volatilities[i] == 0: 
            sharpe_ratios.append(0)
    if mean_returns2 != 1:
        sharpe_ratios2 = []
        for i in range(0,len(mean_returns2)):
            if volatilities2[i]>0:
                sharpe_ratios2.append(mean_returns2[i]/volatilities2[i]*sqrt(252))
            if volatilities2[i] == 0: 
                sharpe_ratios2.append(0)     
    if plot == 'si':
        fig, ax1 = plt.subplots()
        ax1.plot(sharpe_ratios, 'b-', label='Kalman Filter')
        if mean_returns2 !=1:
            ax1.plot(sharpe_ratios2, 'g-', label='Rolling Regression')
        ax1.set_xlabel('Trading Day')
        ax1.set_ylabel('Sharpe Ratio')
        plt.title('1-Month Rolling Sharpe Ratio')
        plt.legend(loc= "best")
        if casa!='si':
            plt.savefig(r'C:\Users\x282066\OneDrive - Santander Office 365\Desktop\imagenes\rolling_sharpe_ratio.png', bbox_inches='tight')
        else:
            plt.savefig(r'C:\Users\diego\Google Drive\UNIVERSIDAD\MASTER\uc3m\TFM\ultimo estado\ultimoestadoseguridad\imagenes\rolling_sharpe_ratio.png', bbox_inches='tight')
        plt.show()
    if mean_returns2 !=1:
        return sharpe_ratios, sharpe_ratios2
    else: 
        return sharpe_ratios

# Sortino ratio
# Returns adjusted for downside risk, calculated as Annualized Return/Annualized Volatility of Negative Returns
# Duda: en este ratio el mean return se tiene que calcular tambien tomando solo los returns negativos o ese se mantiene normal?
def sortino_ratio(returns, returns2 = 1):
    mean_return = sum(returns)/len(returns)
    sortino_ratio = sqrt(sum ( (x - mean_return)**2 for x in returns if x<0) / len(returns))
    if returns2 != 1:
        mean_return2 = sum(returns2)/len(returns2)
        sortino_ratio2 = sqrt(sum ( (x - mean_return2)**2 for x in returns2 if x<0) / len(returns2))   
        return sortino_ratio, sortino_ratio2
    else:
        return sortino_ratio

# Max drawdown
# Largest drop in Pnl or maximum negative difference in total portfolio value. 
# It is calculated as the maximum high to subsequent low difference before a new high is reached.
def max_drawdown2(money,plot, money2 = 1, casa='si'):
    i = np.argmax(np.maximum.accumulate(money) - money) # end of the period
    j = np.argmax(money[:i]) # start of period
    index=0
    for z in money[j+1:]:
        index += 1
        if z>money[j]:
            break
    if plot == 'si':
        plt.plot(money)
        plt.plot([i, j], [money[i], money[j]], 'o', color='Red', markersize=10, label ='Drawdown time')
        if (index+j) < len(money) +1:
            plt.plot( index + j, money[j+index], 'o', color='Green', markersize=10, label ='Recover time')
        plt.figtext(.9, .6, "Max Drawdown Abs = " + str( round(money[i]-money[j]) ) )
        plt.figtext(.9, .5, "Max Drawdown % = " + str( round((money[i]-money[j])/money[j]*100)) )
        plt.figtext(.9, .4, "Duration = " + str(i-j))
        plt.title('Maximum Drawdown position')
        plt.xlabel('Trading Day')
        plt.xlabel('Accumulated Money')
        plt.legend( loc = "best")
        plt.grid()
        if casa!='si':
            plt.savefig(r'C:\Users\x282066\OneDrive - Santander Office 365\Desktop\imagenes\drawdown.png', bbox_inches='tight')
        else:
            plt.savefig(r'C:\Users\diego\Google Drive\UNIVERSIDAD\MASTER\uc3m\TFM\ultimo estado\ultimoestadoseguridad\imagenes\drawdown.png', bbox_inches='tight')
        plt.show()
    Max_Drawdown_Abs = round(money[i]-money[j]) 
    Max_Drawdown = round((money[i]-money[j])/money[j]*100)
    Duration =  str(i-j)
    position_min = i
    position_max = j
    if money2 !=1:
        i = np.argmax(np.maximum.accumulate(money2) - money2) # end of the period
        j = np.argmax(money2[:i]) # start of period
        index2=0
        for z in money2[j+1:]:
            index2 += 1
            if z>money2[j]:
                break
        if plot=='si':
            plt.plot(money2)
            plt.plot([i, j], [money2[i], money2[j]], 'o', color='Red', markersize=10,  label ='Drawdown time')
            if (index2+j) +1 < len(money2):
                plt.plot( index2 + j, money2[j+index2], 'o', color='Green', markersize=10, label ='Recover time')
            plt.figtext(.9, .6, "Max Drawdown Abs = " + str( round(money2[i]-money2[j]) ) )
            plt.figtext(.9, .5, "Max Drawdown % = " + str( round((money2[i]-money2[j])/money2[j]*100)) )
            plt.figtext(.9, .4, "Duration = " + str(i-j))
            plt.title('Maximum Drawdown position')
            plt.xlabel('Trading Day')
            plt.xlabel('Accumulated Money')
            plt.legend( loc = "best")
            plt.grid()
            if casa!='si':
                plt.savefig(r'C:\Users\x282066\OneDrive - Santander Office 365\Desktop\imagenes\drawdown2.png', bbox_inches='tight')
            else:
                plt.savefig(r'C:\Users\diego\Google Drive\UNIVERSIDAD\MASTER\uc3m\TFM\ultimo estado\ultimoestadoseguridad\imagenes\drawdown2.png', bbox_inches='tight')
            plt.show()
        Max_Drawdown_Abs2 = round(money2[i]-money2[j]) 
        Max_Drawdown2 = round((money2[i]-money2[j])/money2[j]*100)
        Duration2 =  str(i-j)
        position_min2 = i
        position_max2 = j
        return Max_Drawdown_Abs, Max_Drawdown, int(Duration), position_min, position_max, index, Max_Drawdown_Abs2, Max_Drawdown2, Duration2, position_min2, position_max2, index2
    else: 
        return Max_Drawdown_Abs, Max_Drawdown, int(Duration), position_min, position_max, index

def daily_drawdown(money, position_min, prices, plot, casa='si'):
    prices2 = pd.DataFrame(money, columns=["Money"]).set_index(prices.index, drop=False)
    # We are going to use a trailing 252 trading day window
    window = 252
    # Calculate the max drawdown in the past window days for each day in the series.
    # Use min_periods=1 if you want to let the first 252 days data have an expanding window
    Roll_Max = prices2.rolling(window, min_periods=1).max()
    Daily_Drawdown = prices2/Roll_Max - 1.0
    # Next we calculate the minimum (negative) daily drawdown in that window.
    # Again, use min_periods=1 if you want to allow the expanding window
    Max_Daily_Drawdown = Daily_Drawdown.rolling(window, min_periods=1).min()
    # Plot the results
    if plot == 'si':
        plt.plot(Daily_Drawdown, label = 'Daily Drawdown' )
        plt.plot(Max_Daily_Drawdown, label = '252 day max Drawdown' )
        plt.plot(position_min, Daily_Drawdown.loc[position_min][0], 'o', color='Red', markersize=10)
        plt.title('Daily Maximum Drawdown')
        plt.xlabel('Trading Day')
        plt.ylabel('Drawdown')
        plt.legend(loc= "best")
    
        if casa!='si':
            plt.savefig(r'C:\Users\x282066\OneDrive - Santander Office 365\Desktop\imagenes\daily_drawdown.png', bbox_inches='tight')
        else:
            plt.savefig(r'C:\Users\diego\Google Drive\UNIVERSIDAD\MASTER\uc3m\TFM\ultimo estado\ultimoestadoseguridad\imagenes\daily_drawdown.png', bbox_inches='tight')
        plt.show()
    return window, Daily_Drawdown

# Average Profit/Loss: Sum(or Avergae) of Profits from trades that results in profits/Sum(or Average) of losses from trades that results in losses
def win_loss_ratio(movimientos, money, movimientos2=1, money2=1):
# Detectamos cierre posiciones
    profit_short_trades, profit_long_trades, cierres_index_short, cierres_index_long = [], [], [], []
    for i in range(0, len(movimientos)-1):
        if movimientos[i] == "SS":
            cierres_index_short.append(i)
        if movimientos[i] == "SL":
            cierres_index_long.append(i)
    for i in cierres_index_short: 
        for j in range(i, 0, -1):
            if movimientos[j] == "ES":
                profit_short_trades.append(money[i] - money[j-1])
                break
    for i in cierres_index_long: 
        for j in range(i, 0, -1):
            if movimientos[j] == "EL":
                profit_long_trades.append(money[i] - money[j-1])
                break
    win_profit = sum( x for x in profit_long_trades if x>0) + sum( x for x in profit_short_trades if x>0)
    loss_profit = sum( x for x in profit_long_trades if x<0) + sum( x for x in profit_short_trades if x<0)
    win_loss_ratio = win_profit / ((-1)*loss_profit )
    if money2 != 1:
        profit_short_trades2, profit_long_trades2, cierres_index_short2, cierres_index_long2 = [], [], [], []
        for i in range(0, len(movimientos2)-1):
            if movimientos2[i] == "SS":
                cierres_index_short2.append(i)
            if movimientos2[i] == "SL":
                cierres_index_long2.append(i)
        for i in cierres_index_short2: 
            for j in range(i, 0, -1):
                if movimientos2[j] == "ES":
                    profit_short_trades2.append(money2[i] - money2[j-1])
                    break
        for i in cierres_index_long2: 
            for j in range(i, 0, -1):
                if movimientos2[j] == "EL":
                    profit_long_trades2.append(money2[i] - money2[j-1])
                    break
        win_profit2 = sum( x for x in profit_long_trades2 if x>0) + sum( x for x in profit_short_trades2 if x>0)
        loss_profit2 = sum( x for x in profit_long_trades2 if x<0) + sum( x for x in profit_short_trades2 if x<0)
        win_loss_ratio2 = win_profit2 / ((-1)*loss_profit2 )   
        return profit_long_trades, profit_short_trades, win_loss_ratio , profit_long_trades2, profit_short_trades2, win_loss_ratio2
    else: 
        return profit_long_trades, profit_short_trades, win_loss_ratio 


def profitability(profit_long_trades,profit_short_trades, profit_long_trades2=1, profit_short_trades2=1):
    #profitable trades
    n_profitable_trades = sum(1 for x in profit_long_trades if x>0) + sum(1 for x in profit_short_trades if x>0)
    n_nonprofit_trades = sum(1 for x in profit_long_trades if x<0) + sum(1 for x in profit_short_trades if x<0)
    profitability = n_profitable_trades/n_nonprofit_trades
    number_trades = n_profitable_trades + n_nonprofit_trades
    if profit_long_trades2!=1:
        n_profitable_trades2 = sum(1 for x in profit_long_trades2 if x>0) + sum(1 for x in profit_short_trades2 if x>0)
        n_nonprofit_trades2 = sum(1 for x in profit_long_trades2 if x<0) + sum(1 for x in profit_short_trades2 if x<0)
        profitability2 = n_profitable_trades2/n_nonprofit_trades2
        number_trades2 = n_profitable_trades2 + n_nonprofit_trades2
        return profitability, n_profitable_trades, n_nonprofit_trades, number_trades, profitability2, n_profitable_trades2, n_nonprofit_trades2, number_trades2
    else: 
        return profitability, n_profitable_trades, n_nonprofit_trades, number_trades



    