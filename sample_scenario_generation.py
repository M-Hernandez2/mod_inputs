#July 24, 2024
#this is used to sample precipitation from historic values (15) and projected values (35)
#then a synthetic time series of 75 years will be created for each sample

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import chaospy as cp
from statsmodels.tsa.ar_model import AutoReg
import scipy.stats as stats


#NOT THIS, JUST USE AS REFERENCE FOR SAMPLING
#historic precipitation range, mm
'''p_hist = [542.80, 1707.1]
#projected precipitation range, mm
p_proj = [669.04, 2255.0]

#create 15 random historic values within range
distribution = cp.J(cp.Uniform(p_hist[0], p_hist[1]))
num_hist = 15
samples_hist = distribution.sample(size=num_hist, rule="L")
print(samples_hist)

#create 35 random projected values within range
distribution = cp.J(cp.Uniform(p_proj[0], p_proj[1]))
num_proj = 35
samples_proj = distribution.sample(size=num_proj, rule="L")
print(samples_proj)
'''

#LOOKING AT HISTROIRCAL PRECIP find mean and standard deviation of annual values
p_t_data = pd.read_excel('C:/Users\mjh7517\OneDrive - The Pennsylvania State University\Downloads\hist_noaa_data_new.xlsx', sheet_name='corr_data')

#historical time series visualization
fig,ax = plt.subplots(figsize=(12,4))
plt.plot(p_t_data['YR'], p_t_data['PRCP'], color='royalblue')
plt.title('NOAA Historic Precipitation Time Series')
plt.ylabel('precip (mm)')
plt.show()

#find mean and standard deviation of historic precipitation
mean_hist = np.mean(p_t_data['PRCP'])
std_hist = np.std(p_t_data['PRCP'])
#print(mean_hist, std_hist)

#create a gaussian distribution of historical precip
p_h = p_t_data['PRCP'].tolist()
print(p_h)
p_sort = sorted(p_h)
fit = stats.norm.pdf(p_sort, loc=mean_hist, scale=std_hist)
plt.plot(p_sort, fit, color='royalblue', label=('mean:', mean_hist, ' sd:', std_hist))
plt.title('Historic Precipitation Gaussian Distribution')
plt.legend()
plt.show()

plt.hist(p_h, bins=20, color='royalblue', edgecolor='navy')
plt.ylabel('frequency')
plt.xlabel('values')
plt.title('Historic Precipitation Historgram')
plt.show()



#NOW TAKE PROJECTED DATA, SAMPLE OUT MEAN AND STD FOR EACH MODEL OF EACH SSP
#using excel file with raw data from NOAA downscaled precip projections ALL_P_p
s26 = pd.read_excel('C:/Users\mjh7517\OneDrive - The Pennsylvania State University\Downloads\ALL_P_p.xlsx', sheet_name='ssp2.6')
s45 = pd.read_excel('C:/Users\mjh7517\OneDrive - The Pennsylvania State University\Downloads\ALL_P_p.xlsx', sheet_name='ssp4.5')
s70 = pd.read_excel('C:/Users\mjh7517\OneDrive - The Pennsylvania State University\Downloads\ALL_P_p.xlsx', sheet_name='ssp7.0')
s85 = pd.read_excel('C:/Users\mjh7517\OneDrive - The Pennsylvania State University\Downloads\ALL_P_p.xlsx', sheet_name='ssp8.5')

