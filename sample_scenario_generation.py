#July 24, 2024 - September 16, 2024
#this is used to sample precipitation from historic values (15) and projected values (35)
#then a synthetic time series of 75 years will be created for each sample

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import chaospy as cp
from statsmodels.tsa.ar_model import AutoReg
import scipy.stats as stats
from scipy.stats import norm
import seaborn as sns


#LOOKING AT HISTROIRCAL PRECIP find mean and standard deviation of annual values
p_data = pd.read_excel('C:/Users\mjh7517\OneDrive - The Pennsylvania State University\Downloads\hist_noaa_data_new.xlsx', sheet_name='compareP')
t_data = pd.read_excel('C:/Users\mjh7517\OneDrive - The Pennsylvania State University\Downloads\hist_noaa_data_new.xlsx', sheet_name='compareT')

p_data = pd.DataFrame(p_data)
t_data = pd.DataFrame(t_data)


#PRECIPITATION
#historical time series visualization
'''fig,ax = plt.subplots(figsize=(12,4))
plt.plot(p_data['YR'], p_data['HIST_P'], color='royalblue')
plt.title('NOAA Historic Precipitation Time Series')
plt.ylabel('precip (mm)')
plt.show()
'''
#find mean and standard deviation of historic precipitation
mean_hist = np.mean(p_data['HIST_P'])
std_hist = np.std(p_data['HIST_P'])
print(mean_hist, std_hist)

#create a gaussian distribution of historical precip
p_h = p_data['HIST_P'].tolist()
print(p_h)
p_sort = sorted(p_h)
'''fit = stats.norm.pdf(p_sort, loc=mean_hist, scale=std_hist)
plt.plot(p_sort, fit, color='royalblue', label=('mean:', mean_hist, ' sd:', std_hist))
plt.title('Historic Precipitation Gaussian Distribution')
plt.legend(loc='lower center')
plt.show()
'''

#BIAS CORRECTION FOR MODEL CMIP HISTORIC TO LINE UP WITH NOAA MEASUREMENTS
#save and remove the noaa historic measurements from the data set of means and stds
#precipitation
p_hist_means = []
p_hist_std = []
p_hist_stats = []
noaa_p_mean = mean_hist
noaa_p_std = std_hist

print(noaa_p_mean)
print(noaa_p_std)

#using excel file with raw data from NOAA downscaled precip projections ALL_P_p
s26 = pd.read_excel('C:/Users\mjh7517\OneDrive - The Pennsylvania State University\Downloads\SWI_Research\ALL_P_proj\ALL_P_p.xlsx', sheet_name='2.6_proj')
s45 = pd.read_excel('C:/Users\mjh7517\OneDrive - The Pennsylvania State University\Downloads\SWI_Research\ALL_P_proj\ALL_P_p.xlsx', sheet_name='4.5_proj')
s70 = pd.read_excel('C:/Users\mjh7517\OneDrive - The Pennsylvania State University\Downloads\SWI_Research\ALL_P_proj\ALL_P_p.xlsx', sheet_name='7.0_proj')
s85 = pd.read_excel('C:/Users\mjh7517\OneDrive - The Pennsylvania State University\Downloads\SWI_Research\ALL_P_proj\ALL_P_p.xlsx', sheet_name='8.5_proj')
#NOW TAKE PROJECTED DATA, SAMPLE OUT MEAN AND STD FOR EACH MODEL OF EACH SSP
#remove years columns for each
s26 = s26.drop('year', axis=1)
s45 = s45.drop('year', axis=1)
s70 = s70.drop('year', axis=1)
s85 = s85.drop('year', axis=1)
#remove the first row with the model names
s26.drop(0)
s45.drop(0)
s70.drop(0)
s85.drop(0)


p_proj = pd.concat([s26, s45, s70, s85], axis=1)
p_proj = pd.DataFrame(p_proj)


#create gaussian distribution for each historic model run to compare with historic data
plt.figure()
p_corr = pd.DataFrame()
p_corr['Year'] = p_data['YR']
for i, col in enumerate(p_data.columns[1:]):
    mean = p_data[col].mean()
    std = p_data[col].std()
    p_hist_means.append(mean)
    p_hist_std.append(std)

    #save means and stds to list for later use
    p_hist_stats.append((col, mean, std))

    x = np.linspace(p_data[col].min(), p_data[col].max(), 100)
    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

    if i == 0:
        plt.plot(x, y, color='gold', zorder=100, linewidth=2)
    else:
        plt.plot(x, y, color='royalblue')

    # BIAS CORRECTION NOW
    col_name = f'corrected_{col}'
    p_corr[col_name] = (p_data[col] - mean) * (noaa_p_std / std) + noaa_p_mean

print('bias corr precip', p_corr)
plt.title('Historic Precipitation Gaussian Distribution')
plt.legend()
plt.show()

#import mean and std for old models in simple dataframe same size as projected dataframe to make looping through easy
ph_mean = pd.read_excel('C:/Users\mjh7517\PycharmProjects\pythonProject\\venv\precip_historic_means.xlsx', sheet_name='Sheet2')
ph_std = pd.read_excel('C:/Users\mjh7517\PycharmProjects\pythonProject\\venv\precip_historic_std.xlsx', sheet_name='Sheet2')
ph_mean_df = pd.DataFrame(ph_mean)
ph_std_df = pd.DataFrame(ph_std)
ph_mean_df = ph_mean_df.astype(float)
ph_std_df = ph_std_df.astype(float)

p_bc = pd.DataFrame(index=p_proj.index, columns=p_proj.columns)

p_proj = p_proj.astype(float)

#apply bias correction to projected data
plt.figure()
x=0
for row in p_proj.index:
    for col in p_proj.columns:
        mean = ph_mean_df.loc[row, col]
        std = ph_std_df.loc[row, col]
        val = p_proj.loc[row, col]

        p_bc.loc[row, col] = (val - mean) * (noaa_p_std /std) + noaa_p_mean

        x = np.linspace(mean - 4 * std, mean + 4 * std, 1000)
        y = norm.pdf(x, mean, std)
        plt.plot(x, y, color='royalblue')


y1 = norm.pdf(x, noaa_p_mean, noaa_p_std)
plt.plot(x, y1, color='gold')
plt.title('Precipitation Bias Corrected Projected Distributions')
plt.xlabel('precipitation (mm/yr)')
plt.ylabel('probability density')
plt.show()
print('BIAS CORR FUTURE PROJECTED PRECIP')
print(p_bc)

'''plt.hist(p_h, bins=20, color='royalblue', edgecolor='navy')
plt.ylabel('frequency')
plt.xlabel('values')
plt.title('Historic Precipitation Historgram')
plt.show()

#TEMPERATURE
#historical time series visualization
fig,ax = plt.subplots(figsize=(12,4))
plt.plot(t_data['YR'], t_data['HIST_T'], color='indianred')
plt.title('NOAA Historic Temperature Time Series')
plt.ylabel('temp (°C)')
plt.show()
'''
#find mean and standard deviation of historic precipitation
Tmean_hist = np.mean(t_data['HIST_T'])
Tstd_hist = np.std(t_data['HIST_T'])
print(Tmean_hist, Tstd_hist)

#create a gaussian distribution of historical precip
t_h = t_data['HIST_T'].tolist()
print(t_h)
t_sort = sorted(t_h)
'''fit = stats.norm.pdf(t_sort, loc=Tmean_hist, scale=Tstd_hist)
plt.plot(t_sort, fit, color='indianred', label=('mean:', Tmean_hist, ' sd:', Tstd_hist))
plt.title('Historic Temperature Gaussian Distribution')
plt.legend()
plt.show()
'''
#NOW TAKE PROJECTED DATA, SAMPLE OUT MEAN AND STD FOR EACH MODEL OF EACH SSP
#using excel file with raw data from NOAA downscaled precip projections ALL_P_p
Ts26 = pd.read_excel('C:/Users\mjh7517\OneDrive - The Pennsylvania State University\Downloads\SWI_Research\ALL_T_proj\ALL_T_p.xlsx', sheet_name='2.6_proj')
Ts45 = pd.read_excel('C:/Users\mjh7517\OneDrive - The Pennsylvania State University\Downloads\SWI_Research\ALL_T_proj\ALL_T_p.xlsx', sheet_name='4.5_proj')
Ts70 = pd.read_excel('C:/Users\mjh7517\OneDrive - The Pennsylvania State University\Downloads\SWI_Research\ALL_T_proj\ALL_T_p.xlsx', sheet_name='7.0_proj')
Ts85 = pd.read_excel('C:/Users\mjh7517\OneDrive - The Pennsylvania State University\Downloads\SWI_Research\ALL_T_proj\ALL_T_p.xlsx', sheet_name='8.5_proj')

#remove years columns for each
Ts26 = Ts26.drop('year', axis=1)
Ts45 = Ts45.drop('year', axis=1)
Ts70 = Ts70.drop('year', axis=1)
Ts85 = Ts85.drop('year', axis=1)
#remove the first row with the model names
Ts26.drop(0)
Ts45.drop(0)
Ts70.drop(0)
Ts85.drop(0)

t_proj = pd.concat([Ts26, Ts45, Ts70, Ts85], axis=1)
t_proj = pd.DataFrame(t_proj)


#temperature
t_hist_means = []
t_hist_std = []
t_hist_stats = []
noaa_t_mean = Tmean_hist
noaa_t_std = Tstd_hist

#create gaussian distribution of measured histopric and models historic temperature and bias correct models to observed
t_corr = pd.DataFrame()
t_bc = pd.DataFrame()
t_corr['Year'] = t_data['YR']
plt.figure()
for i, col in enumerate(t_data.columns[1:]):
    mean = t_data[col].mean()
    std = t_data[col].std()
    t_hist_means.append(mean)
    t_hist_std.append(std)

    #save list of each mean and standard deviation
    t_hist_stats.append((col, mean, std))

    x = np.linspace(t_data[col].min(), t_data[col].max(), 100)
    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

    if i == 0:
        plt.plot(x, y, color='gold', zorder=100, linewidth=2)
    else:
        plt.plot(x, y, color='indianred')

    #BIAS CORRECTION NOW
    t_corr[col] = (t_data[col] - mean) * (noaa_t_std / std) + noaa_t_mean

print('bias corr temp', t_corr)
print(t_hist_stats)

plt.title('Historic Temperature Gaussian Distribution')
plt.legend()
plt.show()

#t_hist_means = pd.DataFrame(t_hist_means)
#t_hist_means.to_excel('temp_hist_means.xlsx', index=False)
#t_hist_std = pd.DataFrame(t_hist_std)
#t_hist_std.to_excel('temp_hist_std.xlsx', index=False)

#import mean and std for old models in simple dataframe same size as projected dataframe to make looping through easy
th_mean = pd.read_excel('C:/Users\mjh7517\PycharmProjects\pythonProject\\venv\\temp_hist_means.xlsx', sheet_name='Sheet2')
th_std = pd.read_excel('C:/Users\mjh7517\PycharmProjects\pythonProject\\venv\\temp_hist_std.xlsx', sheet_name='Sheet2')
th_mean_df = pd.DataFrame(th_mean)
th_std_df = pd.DataFrame(th_std)
th_mean_df = th_mean_df.astype(float)
th_std_df = th_std_df.astype(float)

t_bc = pd.DataFrame(index=t_proj.index, columns=t_proj.columns)

t_proj = t_proj.astype(float)

#apply bias correction to projected data
plt.figure()
x=0
for row in t_proj.index:
    for col in t_proj.columns:
        mean = th_mean_df.loc[row, col]
        std = th_std_df.loc[row, col]
        val = t_proj.loc[row, col]

        t_bc.loc[row, col] = (val - mean) * (noaa_t_std /std) + noaa_t_mean

        x = np.linspace(mean - 4 * std, mean + 4 * std, 1000)
        y = norm.pdf(x, mean, std)
        plt.plot(x, y, color='indianred')

y1 = norm.pdf(x, noaa_t_mean, noaa_t_std)
plt.plot(x, y1, color='gold')
plt.title('Temperature Bias Corrected Projected Distributions')
plt.xlabel('temperature (C)')
plt.ylabel('probability density')
plt.show()

print('BIAS CORR FUTURE PROJECTED TEMP')
print(t_bc)

#PRINT NEWPLOT OF BIAS CORRECTED CMIP6 HISTORIC DATA TO ALIGN WITH NOAA HISTORIC
#precipitation
pdata = []
pmean = []
pstd = []

plt.figure(figsize=(8,6))
for col in p_corr.columns[1:]:
    data = p_corr[col]
    Pmean = data.mean()
    Pstd = data.std()

    pdata.append(data)
    pmean.append(Pmean)
    pstd.append(Pstd)

    #make gaussian curves
    x = np.linspace(data.min(), data.max(), 100)
    y = norm.pdf(x, Pmean, Pstd)

    #plot distributions
    plt.plot(x, y, color='royalblue')

ph_stats = pd.DataFrame({'means': pmean,
                        'std': pstd})
#ph_stats.to_excel('precip_hist_stats.xlsx', index=False)
plt.title('Precipitation Historic Bias Corrected Distributions')
plt.xlabel('precipitaton')
plt.ylabel('density')
plt.show()

#temperature
tdata = []
tmean = []
tstd = []

plt.figure(figsize=(8,6))
for col in t_corr.columns[1:]:
    data = t_corr[col]
    Tmean = data.mean()
    Tstd = data.std()

    tdata.append(data)
    tmean.append(Tmean)
    tstd.append(Tstd)

    #make gaussian curves
    x = np.linspace(data.min(), data.max(), 100)
    y = norm.pdf(x, Tmean, Tstd)

    #plot distributions
    plt.plot(x, y, color='indianred')

th_stats = pd.DataFrame({'means': tmean,
                        'std': tstd})
plt.title('Temperature Historic Bias Corrected Distributions')
plt.xlabel('temperature')
plt.ylabel('density')
plt.show()


'''
plt.hist(t_h, bins=20, color='indianred', edgecolor='brown')
plt.ylabel('frequency')
plt.xlabel('values')
plt.title('Historic Temperature Historgram')
plt.show()
'''

#get list of means and std for each model run for projected to create sample space after
#precipitation
pmean = p_bc.mean()
pstd = p_bc.std()

#temperature
tmean = t_bc.mean()
tstd = t_bc.std()




#create space of all projected means and standard deviations
#precipitation
plt.scatter(pmean, pstd, color='royalblue', label='projected')
plt.scatter(mean_hist, std_hist, color='gold', label='historic')
plt.legend()
plt.ylabel('standard deviations')
plt.xlabel('precipitation (mm/yr)')
plt.title('Precipitation means and standard deviations')
plt.show()

#plot distribution and histogram for projected ssps
ssp_sort = sorted(pmean)

plt.hist(ssp_sort, bins=20, color='royalblue', edgecolor='navy')
plt.ylabel('frequency')
plt.xlabel('values')
plt.title('Projected Precipitation Historgram')
plt.show()


p_mean_min = min(pmean)
p_mean_max = max(pmean)
p_std_min = min(pstd)
p_std_max = max(pstd)

print('PRECIP MEAN MIN THEN STD MIN')
print(p_mean_min)
print(p_std_min)

#create space of all projected means and standard deviations
#temperature
plt.scatter(tmean, tstd, color='indianred', label='projected')
plt.scatter(Tmean_hist, Tstd_hist, color='gold', label='historic')
plt.ylabel('standard deviations')
plt.legend()
plt.xlabel('temperature (C)')
plt.title('Temperature means and standard deviations')
plt.show()


#plot distribution and histogram for projected ssps
means_list_T = sorted(tmean)

plt.hist(means_list_T, bins=20, color='indianred', edgecolor='brown')
plt.ylabel('frequency')
plt.xlabel('values')
plt.title('Projected Temperature Historgram')
plt.show()

t_mean_min = min(tmean)
t_mean_max = max(tmean)
t_std_min = min(tstd)
t_std_max = max(tstd)



#SAMPLING SPACE OUT OF MEANS AND STANDARD DEVIATIONS FOR P AND T TO USE FOR THEN MAKING TIME SERIES

#USE THIS
#four dimensional sampling space with 100 samples
t_mean_range = [t_mean_min, t_mean_max]
t_std_range = [t_std_min, t_std_max]
p_mean_range = [p_mean_min, p_mean_max]
p_std_range = [p_std_min, p_std_max]

distribution = cp.J(cp.Uniform(p_mean_range[0], p_mean_range[1]),
                    cp.Uniform(p_std_range[0], p_std_range[1]),
                    cp.Uniform(t_mean_range[0], t_mean_range[1]),
                    cp.Uniform(t_std_range[0], t_std_range[1]))
samples = distribution.sample(100, rule='L')
print(samples.shape)
print(samples)

#put it all into a dataframe and save it to excel file
sample_space_df = pd.DataFrame(samples.T, columns=['p mean', 'p standard deviation', 't mean', 't standard deviation'])

#sample_space_df.to_excel('precipTemp_samplingSpace.xlsx', index=False)

#create plot of 4d sampling space
sns.pairplot(sample_space_df, corner='True', palette='PuOr')
plt.show()


#CREATE SYNTHETIC TIME SERIES FOR PRECIPITATION AND TEMP USING SAMPLE SPACE BASED ON MEANS AND STDS

n = 75              #num of years to create time series for

dates = pd.date_range(start='2025', periods=n, freq='Y')

#precipitation
p_time_series = pd.DataFrame(index=dates)
for i, row in sample_space_df.iterrows():
    mean_val = row['p mean']
    std_val = row['p standard deviation']
    synthetic = np.random.normal(loc=mean_val, scale=std_val, size=n)
    while np.any(synthetic <= 0):
        synthetic[synthetic <= 0] = np.random.normal(loc=mean_val, scale=std_val, size=np.sum(synthetic <= 0))

    p_time_series[f'pseries_{i+1}'] = synthetic

#to make sure all the data is a positive number for creating the syntetic time series for p


#temperature
t_time_series = pd.DataFrame(index=dates)
for i, row in sample_space_df.iterrows():
    mean_val = row['t mean']
    std_val = row['t standard deviation']
    synthetic = np.random.normal(loc=mean_val, scale=std_val, size=n)

    t_time_series[f'tseries_{i+1}'] = synthetic


print(p_time_series)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,6))

ax1.plot(dates, p_time_series, color='royalblue', alpha=0.6)
ax1.set_ylabel('precipitation (mm)')


ax2.plot(dates, t_time_series, color='indianred', alpha=0.6)
ax1.set_title('Synthetic Time Series')
ax2.set_ylabel('temperature (°C)')
ax2.set_xlabel('year')
plt.legend().set_visible(False)
plt.show()


#save both p and t time series to an excel file
#p_time_series.to_excel('precip_timeseries.xlsx', index=False)
#t_time_series.to_excel('temp_timeseries.xlsx', index=False)


