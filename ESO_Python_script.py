# © 2020 Guglielmo Del Sarto & Marco Contadini
#
# The content hereby presented is fully coded by us with the exeption of
# the function "mean_confidence_interval" which was taken from:
# https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
#
# ESO option price, Assignment FINANCIAL RISK MANAGEMENT
# Built on Python version 3.8
# Libraries used:
import numpy as np
import pandas as pd
import time
import scipy
import scipy.stats as ss
from scipy.stats import ttest_1samp as ttest
import matplotlib
import math
from matplotlib import pyplot as pl
#

# # # SECTION 1: classes # # #

# 1. A plain vanilla option:
class Option:
    # First, the characteristichs:
    def __init__(self, Underlying_0, Strike_Price, Volatility, Risk_free, Maturity, Call_Put):
        self.U = Underlying_0
        self.K = Strike_Price
        self.Type = Call_Put
        self.vol = Volatility
        self.T = Maturity
        self.r = Risk_free
    # BS preliminaries:
    def d1d2(self):
        d1 =  (np.log(self.U/self.K) + (self.r + self.vol**2 / 2) * self.T)/(self.vol * np.sqrt(self.T))
        d2 =  (np.log(self.U/self.K) + (self.r - self.vol**2 / 2) * self.T)/(self.vol * np.sqrt(self.T))
        return d1, d2
    # BS:
    def Black_Scholes(self):
        d1, d2 = self.d1d2()
        if self.Type=='C' or self.Type=='c':
            return self.U * ss.norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * ss.norm.cdf(d2)
        else:
            return self.K * np.exp(-self.r * self.T) * ss.norm.cdf(-d2) - self.U * ss.norm.cdf(-d1)
# 2. Our EsoOption:
class Eso_option:
    def __init__(self, s_0, k, vol, rf, t, ns):
        self.szero = s_0
        self.k = k
        self.final = simulate_path(s_0, vol, rf, t)
        self.irr, self.irr_tx = find_annual_IRR(s_0,self.final,t)
        self.call, self.put = calculate_call_payoffs(self.final,self.k,vol,rf,t)
        self.nstock = find_number_of_stocks(self.irr,ns)
        self.nstock_tx = find_number_of_stocks(self.irr_tx,ns)
        self.ESOvalues = self.call*self.nstock
        self.ESOvalues_tx = self.call*self.nstock_tx
        self.ESOvalue, self.ESOvalue_min, self.ESOvalue_max, self.sd = confidence_interval(self.ESOvalues)
        self.ESOvalue_tx, self.ESOvalue_min_tx, self.ESOvalue_max_tx, self.sd_tx = confidence_interval(self.ESOvalues_tx)
#

# # # SECTION 2: auxiliary functions # # #

# 1. Discount Function:
def discount(Values,rf,t):
    df = math.exp(-rf*t)
    return Values*df
# 2. IRR and IRR_tax:
def find_annual_IRR(s_0,s_t,t):
    IRR = ((s_t/s_0)**(1/t)) - 1
    IRR_tax = (((s_t*(1-0.26))/s_0 + 0.26)**(1/t)) - 1
    return IRR, IRR_tax
# 3. Number of stocks given an IRR value:
def find_number_of_stocks(IRR,BA):
    NS = np.empty(len(IRR))
    for i in range(len(IRR)):
        if IRR[i] < 0.2:
            NS[i] = 0
        elif IRR[i] >= 0.2 and IRR[i] < 0.25:
            NS[i] = 0.025*BA + ((0.06*BA - 0.025*BA) * (IRR[i] - 0.2)/(0.25 - 0.2))
        elif IRR[i] >= 0.25 and IRR[i] < 0.3:
            NS[i] = 0.06*BA + ((0.09*BA - 0.06*BA) * (IRR[i] - 0.25)/(0.3 - 0.25))
        elif IRR[i] >= 0.3 and IRR[i] < 0.35:
            NS[i] = 0.09*BA + ((0.11*BA - 0.09*BA) * (IRR[i] - 0.3)/(0.35 - 0.3))
        elif IRR[i] >= 0.35 and IRR[i] < 0.4:
            NS[i] = 0.11*BA + ((0.125*BA - 0.11*BA) * (IRR[i] - 0.35)/(0.4 - 0.35))
        elif IRR[i] >= 0.4:
            NS[i] = 0.125*BA
    return NS
# 4. Monte Carlo Simulation function:
def simulate_path(s_0, vol, rf, t):
    number_paths = 30000000
    epsilon = np.random.randn(number_paths)
    s_T = s_0 * np.exp((rf - 0.5*(vol**2))*t + vol*math.sqrt(t)*epsilon)
    return s_T
# 5. Find a call option payoff from an array of stock value:
def calculate_call_payoffs(s_t, k, vol, rf, t):
    payoffs_t_c = np.maximum(s_t - k, 0)
    payoffs_0_c = discount(payoffs_t_c, rf, t)
    payoffs_t_p = np.maximum(k - s_t, 0)
    payoffs_0_p = discount(payoffs_t_p, rf, t)
    return payoffs_0_c, payoffs_0_p
# 6. Confide interval (not coded by us):
def confidence_interval(data,confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h, se
#

# # # SECTION 3: User interface # # #

print("Hello. Nice to meet you. I am ESO_pricer_bot.\nI am here to help you discover the price of the ESO option desribed in the assignment.")
test = int(input('Let us first check that everything works fine. Would you like make a test? (logic 1/0): '))
if test == 1:
    time.sleep(1)
    print("Let me price a plain vanilla option. \nI will compare my result with BS: let's see whether I am good or not.")
    time.sleep(1)
    print("Let's start!")
    time.sleep(2)
    #
    type = str(input('The option you want to price is call (c or C) or put (p or P)? '))
    s_0 = float(input('What is the current value (€) of the underlying? '))
    k = float(input('What is the strike price? '))
    vol = float(input('What is the volatility of the stock? Please, use decimal values (24% -> 0.24) '))
    t = float(input('What is the maturiry of the contract? (please, value in years) '))
    rf = float(input('What is the risk free rate between today and maturity? '))
    print('Perfect. I will be back in few seconds.')
    BS_option = Option(s_0, k, vol, rf, t, type)
    MY_option = Eso_option(s_0, k, vol, rf, t, 1)
    if type == 'c' or type == 'C':
        mine = MY_option.call
    else:
        mine = MY_option.put
    bs_price = BS_option.Black_Scholes()
    print(f'The price according to BS is: {round(bs_price,5)} €. \nThe price according to my own method is: {round(np.mean(mine),5)} €')
    print('...')
    time.sleep(1.8)
    t_test, p_value = ttest(mine, bs_price)
    print('Let me build a t-test. I will show my precision testing\nH_0: my price = BS prcie')
    print(f'The result of the test is: \nt-statistic: {round(t_test,5)}\npvalue of: {round(p_value,5)}.\nSeems that I was quite accurate.')
print('...\nNow, refresh my memory and give me details about ESO option!')
HoldCo_0 = float(input("What is HoldCo today's stock value (€)? "))
H_k = HoldCo_0
HoldCo_vol = float(input("What is HoldCo equity volatility? "))
number_stock = float(input("How many outstanding shares does HoldCo have? "))
exit_times = list(map(float, input('Which are the possible exit years? (value separated by space: 1 3...) ').split()))
print('Now, FOLLOWING THE ORDER in which you have insert THE YEARS, please:')
exit_proba = list(map(float, input('indicate the exit probability (decimal value separated by space: 0.4 0.6..): ').split()))
exit_rf = list(map(float, input('indicate the risk free rate for [0,T] (decimal values separated by space: 0.4 0.6..): ').split()))
print('Allow me few seconds please.')
if len(exit_times) != len(exit_proba) or len(exit_proba) != len(exit_rf):
    print ("Woooops. Seems vector length do not agree. Be more careful.")
    exit_times = list(map(float, input('What are possible the exit time? (value separated by space: 1 3 5 6...) ').split()))
    exit_proba = list(map(float, input('For each exit year, please indicate the exit probability (decimal value separated by space: 0.4 0.6...): ').split()))
    exit_rf = list(map(float, input('For each exit year, please indicate the risk free rate (decimal value separated by space: 0.4 0.6...): ').split()))
if len(exit_times) != len(exit_proba) or len(exit_proba) != len(exit_rf):
    print ("Woooops. Seems vector length do not agree. Try to restart the program.")
list_of_eso = list()
n = range(len(exit_times))
for i in n:
    list_of_eso.append(Eso_option(HoldCo_0, H_k, HoldCo_vol, exit_rf[i], exit_times[i], number_stock))
value_given_exit = np.empty(len(list_of_eso))
value_given_exit_min = np.empty(len(list_of_eso))
value_given_exit_max = np.empty(len(list_of_eso))
sd = np.empty(len(list_of_eso))
value_given_exit_tx = np.empty(len(list_of_eso))
value_given_exit_min_tx = np.empty(len(list_of_eso))
value_given_exit_max_tx = np.empty(len(list_of_eso))
sd_tx = np.empty(len(list_of_eso))
for i in n:
    value_given_exit[i], value_given_exit_min[i], value_given_exit_max[i], sd[i] = list_of_eso[i].ESOvalue, list_of_eso[i].ESOvalue_min, list_of_eso[i].ESOvalue_max, list_of_eso[i].sd
    value_given_exit_tx[i], value_given_exit_min_tx[i], value_given_exit_max_tx[i], sd_tx[i] = list_of_eso[i].ESOvalue_tx, list_of_eso[i].ESOvalue_min_tx, list_of_eso[i].ESOvalue_max_tx, list_of_eso[i].sd_tx
# Unconditional value:
unconditional_expected_value, unconditional_expected_value_min, unconditional_expected_value_max = 0, 0, 0
#
unconditional_expected_value_tx, unconditional_expected_value_min_tx, unconditional_expected_value_max_tx  = 0, 0, 0
for i in n:
    unconditional_expected_value += value_given_exit[i]*exit_proba[i]
    unconditional_expected_value_min += value_given_exit_min[i]*exit_proba[i]
    unconditional_expected_value_max += value_given_exit_max[i]*exit_proba[i]
    #
    unconditional_expected_value_tx += value_given_exit_tx[i]*exit_proba[i]
    unconditional_expected_value_min_tx += value_given_exit_min_tx[i]*exit_proba[i]
    unconditional_expected_value_max_tx += value_given_exit_max_tx[i]*exit_proba[i]
#

print('')
print('The ESO option price is in the interval (values in €) (',unconditional_expected_value_min,';',unconditional_expected_value_max,')')
print('with a confidence level of 95%')
print('The centered value is: ',unconditional_expected_value, '€')
#
print('')
print('The ESO option price in the case in which our investors are subject to Italian fiscal regime')
print('is in the interval (values in €) (',unconditional_expected_value_min_tx,';',unconditional_expected_value_max_tx,')')
print('with a confidence level of 95%')
print('The centered value is: ',unconditional_expected_value_tx, '€')
