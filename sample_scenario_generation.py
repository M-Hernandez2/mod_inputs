#July 24, 2024
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
#create gaussian distribution for each historic model run to compare with historic data
p_hist_means = []
p_hist_std = []
p_hist_stats = []
plt.figure()
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
        plt.plot(x, y, color='gold')
    else:
        plt.plot(x, y, color='royalblue')

plt.title('Historic Precipitation Gaussian Distribution')
plt.legend()
plt.show()


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
t_hist_means = []
t_hist_std = []
t_hist_stats = []
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
        plt.plot(x, y, color='gold')
    else:
        plt.plot(x, y, color='indianred')

plt.title('Historic Temperature Gaussian Distribution')
plt.legend()
plt.show()
'''
plt.hist(t_h, bins=20, color='indianred', edgecolor='brown')
plt.ylabel('frequency')
plt.xlabel('values')
plt.title('Historic Temperature Historgram')
plt.show()
'''

#BIAS CORRECTION FOR MODEL CMIP HISTORIC TO LINE UP WITH NOAA MEASUREMENTS
#save and remove the noaa historic measurements from the data set of means and stds
#precipitation
noaa_p_stats, noaa_p_mean, noaa_p_std = p_hist_stats[0]
print(noaa_p_stats)
print(noaa_p_mean)
print(noaa_p_std)

first_row = p_hist_stats.pop(0)
print(p_hist_stats)

#temperature
noaa_t_stats, noaa_t_mean, noaa_t_std = t_hist_stats[0]
print(noaa_t_stats)
print(noaa_t_mean)
print(noaa_t_std)

first_row2 = t_hist_stats.pop(0)
print(t_hist_stats)

#now align model estimates to historic measures
#MEAN BIAS CORRECTION METHOD
corr_p = []

for model, mean, std in p_hist_stats:
    shift = noaa_p_mean - mean
    scale = noaa_p_std / std
    means = mean + shift
    stds = std * scale

    aligned = {
        'model': model,
        'og mean': mean,
        'og std': std,
        'corr mean': means,
        'corr std': stds,
        'shift': shift,
        'scale': scale
    }
    corr_p.append(aligned)

corr_p_df = pd.DataFrame(corr_p)
#file_output = 'mean_bc_P.xlsx'
#corr_p_df.to_excel(file_output, index=False)
#print(corr_p_df)

#PRINT NEWPLOT OF BIAS CORRECTED CMIP6 HISTORIC DATA TO ALIGN WITH NOAA HISTORIC
#make mean and std seperate lists for simplicity of the loop
p_mean = corr_p_df['corr mean']
minp = min(p_mean) -3
maxp = max(p_mean) +3
p_std = corr_p_df['corr std']
x=0
plt.figure()
for a, b in zip(p_mean, p_std):

    x = np.linspace(550, 2000, 100)
    y = (1 / (b * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - a) / b) ** 2)

    plt.plot(x, y, color='royalblue')

y = (1 / (noaa_p_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - noaa_p_mean) / noaa_p_std) ** 2)
plt.plot(x, y, color='gold')
plt.title('Bias Corrected Historic Precipitation Distribution')
plt.legend()
plt.show()

corr_t = []

for model, mean, std in t_hist_stats:
    shift = noaa_t_mean - mean
    scale = noaa_t_std / std
    means = mean + shift
    stds = std * scale

    aligned = {
        'model': model,
        'og mean': mean,
        'og std': std,
        'corr mean': means,
        'corr std': stds,
        'shift': shift,
        'scale': scale
    }
    corr_t.append(aligned)

corr_t_df = pd.DataFrame(corr_t)

#PRINT NEWPLOT OF BIAS CORRECTED CMIP6 HISTORIC DATA TO ALIGN WITH NOAA HISTORIC
#make mean and std seperate lists for simplicity of the loop
t_mean = corr_t_df['corr mean']
min0 = min(t_mean) -3
max0 = max(t_mean) +3
t_std = corr_t_df['corr std']
x=0
plt.figure()
for a, b in zip(t_mean, t_std):

    x = np.linspace(min0, max0, 100)
    y = (1 / (b * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - a) / b) ** 2)

    plt.plot(x, y, color='indianred')

y = (1 / (noaa_t_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - noaa_t_mean) / noaa_t_std) ** 2)
plt.plot(x, y, color='gold')
plt.title('Bias Corrected Historic Temperature Distribution')
plt.legend()
plt.show()
#file_output2 = 'mean_bc_t.xlsx'
#corr_t_df.to_excel(file_output2, index=False)
#print(corr_t)



#now use the mean bias correction change and apply to the future scenarios
#PRECIPITATION

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


#get mean and std for each model of eash ssp
proj_means26 = []
proj_stds26 = []
proj_means45 = []
proj_stds45 = []
proj_means70 = []
proj_stds70 = []
proj_means85 = []
proj_stds85 = []
for col in s26.columns:
    mean = s26[col].mean()
    std = s26[col].std()
    proj_means26.append(mean)
    proj_stds26.append(std)
for col in s45.columns:
    mean = s45[col].mean()
    std = s45[col].std()
    proj_means45.append(mean)
    proj_stds45.append(std)
for col in s70.columns:
    mean = s70[col].mean()
    std = s70[col].std()
    proj_means70.append(mean)
    proj_stds70.append(std)
for col in s85.columns:
    mean = s85[col].mean()
    std = s85[col].std()
    proj_means85.append(mean)
    proj_stds85.append(std)


#print(proj_means26)

#put all means and std into 1 list
means_list = proj_means26 + proj_means45 + proj_means70 + proj_means85
std_list = proj_stds26 + proj_stds45 + proj_stds70 + proj_stds85
#print(std_list)
#print(len(means_list))
print(means_list)
#now use the mean bias correction change and apply to the future scenarios
for i, means in enumerate(corr_p_df['shift']):
    means_list[i] += means
for i, stds in enumerate(corr_p_df['scale']):
    std_list[i] = std_list[i] * scale
print(means_list)


#create space of all projected means and standard deviations
'''plt.scatter(means_list, std_list, color='royalblue', label='projected')
plt.scatter(mean_hist, std_hist, color='gold', label='historic')
plt.legend()
plt.ylabel('standard deviations')
plt.xlabel('means')
plt.title('Precipitation means and standard deviations')
plt.show()
'''
#combine means and std lists to a dataframe
p_df = pd.DataFrame({'mean': means_list, 'std': std_list})

#plot the gaussian curves for the projected space
plt.figure()
for a, b in zip(means_list, std_list):

    x = np.linspace(500, 2500, 100)
    y = (1 / (b * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - a) / b) ** 2)

    plt.plot(x, y, color='royalblue')

plt.title('Bias Corrected Projected Precipitation Distribution')
plt.legend()
plt.show()





#plot distribution and histogram for projected ssps
ssp_sort = sorted(means_list)
'''
plt.hist(ssp_sort, bins=20, color='royalblue', edgecolor='navy')
plt.ylabel('frequency')
plt.xlabel('values')
plt.title('Projected Precipitation Historgram')
plt.show()
'''

p_mean_min = min(means_list)
p_mean_max = max(means_list)
p_std_min = min(std_list)
p_std_max = max(std_list)




#TEMPERATURE

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


#get mean and std for each model of eash ssp
proj_means26_T = []
proj_stds26_T = []
proj_means45_T = []
proj_stds45_T = []
proj_means70_T = []
proj_stds70_T = []
proj_means85_T = []
proj_stds85_T = []
for col in Ts26.columns:
    mean = Ts26[col].mean()
    std = Ts26[col].std()
    proj_means26_T.append(mean)
    proj_stds26_T.append(std)
for col in Ts45.columns:
    mean = Ts45[col].mean()
    std = Ts45[col].std()
    proj_means45_T.append(mean)
    proj_stds45_T.append(std)
for col in Ts70.columns:
    mean = Ts70[col].mean()
    std = Ts70[col].std()
    proj_means70_T.append(mean)
    proj_stds70_T.append(std)
for col in Ts85.columns:
    mean = Ts85[col].mean()
    std = Ts85[col].std()
    proj_means85_T.append(mean)
    proj_stds85_T.append(std)


#print(proj_means26_T)

#put all means and std into 1 list
means_list_T = proj_means26_T + proj_means45_T + proj_means70_T + proj_means85_T
std_list_T = proj_stds26_T + proj_stds45_T + proj_stds70_T + proj_stds85_T
#print(std_list_T)
#print(len(means_list_T))

#now use the mean bias correction change and apply to the future scenarios
for i, means in enumerate(corr_t_df['shift']):
    means_list_T[i] += means
for i, stds in enumerate(corr_t_df['scale']):
    std_list_T[i] = std_list_T[i] * scale
print(means_list_T)
#combine means and std lists to a dataframe
t_df = pd.DataFrame({'mean': means_list_T, 'std': std_list_T})

#create space of all projected means and standard deviations
'''plt.scatter(means_list_T, std_list_T, color='indianred', label='projected')
plt.scatter(Tmean_hist, Tstd_hist, color='gold', label='historic')
plt.ylabel('standard deviations')
plt.legend()
plt.xlabel('means')
plt.title('Temperature means and standard deviations')
plt.show()
'''


#plot the gaussian curves for the projected space
plt.figure()
for a, b in zip(means_list_T, std_list_T):

    x = np.linspace(10, 30, 100)
    y = (1 / (b * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - a) / b) ** 2)

    plt.plot(x, y, color='indianred')

plt.title('Bias Corrected Projected Temperature Distribution')
plt.legend()
plt.show()


#plot distribution and histogram for projected ssps
means_list_T = sorted(means_list_T)

'''plt.hist(means_list_T, bins=20, color='indianred', edgecolor='brown')
plt.ylabel('frequency')
plt.xlabel('values')
plt.title('Projected Temperature Historgram')
plt.show()
'''
t_mean_min = min(means_list_T)
t_mean_max = max(means_list_T)
t_std_min = min(std_list_T)
t_std_max = max(std_list_T)







#SAMPLING SPACE OUT OF MEANS AND STANDARD DEVIATIONS FOR P AND T TO USE FOR THEN MAKING TIME SERIES
#precipitation
'''
p_mean_range = [p_mean_min, p_mean_max]
p_std_range = [p_std_min, p_std_max]

#create semi random precipitation values within range of possible mean and standard deviation
distribution = cp.J(cp.Uniform(p_mean_range[0], p_mean_range[1]),
                    cp.Uniform(p_std_range[0], p_std_range[1]))
num_p = 50
samples_p = distribution.sample(size=num_p, rule="L")

px = samples_p[0]
py = samples_p[1]

#creates dataframe of precip mean and std
p_df = pd.DataFrame({'mean': px,
                           'standard deviation': py})

print(p_df)

#scatter plot of mean and std precip sample space
fig, ax = plt.subplots()
ax.scatter(x = p_df['mean'], y = p_df['standard deviation'], color = 'royalblue')
ax.set_xlabel('mean (mm/yr)')
ax.set_ylabel('standard deviation')
ax.set_title('Latin-Hypercube Sample of Projected Precipitation Space')
plt.show()


#temperature
t_mean_range = [t_mean_min, t_mean_max]
t_std_range = [t_std_min, t_std_max]

#create semi random precipitation values within range of possible mean and standard deviation
distribution = cp.J(cp.Uniform(t_mean_range[0], t_mean_range[1]),
                    cp.Uniform(t_std_range[0], t_std_range[1]))
num_t = 50
samples_t = distribution.sample(size=num_t, rule="L")

tx = samples_t[0]
ty = samples_t[1]

#creates dataframe of temp mean and std
t_df = pd.DataFrame({'mean': tx,
                           'standard deviation': ty})

print(t_df)

#scatter plot of mean and std temp sample space
fig, ax = plt.subplots()
ax.scatter(x = t_df['mean'], y = t_df['standard deviation'], color = 'indianred')
ax.set_xlabel('mean (°C)')
ax.set_ylabel('standard deviation')
ax.set_title('Latin-Hypercube Sample of Projected Temperature Space')
plt.show()
'''
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

sample_space_df.to_excel('precipTemp_samplingSpace.xlsx', index=False)

#create plot of 4d sampling space
sns.pairplot(sample_space_df, diag_kind='kde', corner='True', palette='mediumturquoise')
plt.show()


#CREATE SYNTHETIC TIME SERIES FOR PRECIPITATION AND TEMP USING SAMPLE SPACE BASED ON MEANS AND STDS

n = 75              #num of years to create time series for

dates = pd.date_range(start='2025', periods=n, freq='Y')

#precipitation
p_time_series = pd.DataFrame(index=dates)
for i, row in p_df.iterrows():
    mean_val = row['mean']
    std_val = row['std']
    synthetic = np.random.normal(loc=mean_val, scale=std_val, size=n)

    p_time_series[f'pseries_{i+1}'] = synthetic


#temperature
t_time_series = pd.DataFrame(index=dates)
for i, row in t_df.iterrows():
    mean_val = row['mean']
    std_val = row['std']
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
p_time_series.to_excel('precip_timeseries.xlsx', index=False)
t_time_series.to_excel('temp_timeseries.xlsx', index=False)


