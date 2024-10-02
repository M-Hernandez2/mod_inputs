#September 24, 2024
#create data visualizations to show differences between origional modflow and my updates

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#import well data
well_2926 = pd.read_excel('C:\\Users\mjh7517\PycharmProjects\pythonProject\\venv\well_irrigation_simulation.xlsx',sheet_name='2926')
well_2754 = pd.read_excel('C:\\Users\mjh7517\PycharmProjects\pythonProject\\venv\well_irrigation_simulation.xlsx',sheet_name='2754')
well_2736 = pd.read_excel('C:\\Users\mjh7517\PycharmProjects\pythonProject\\venv\well_irrigation_simulation.xlsx',sheet_name='2736')
well_2813 = pd.read_excel('C:\\Users\mjh7517\PycharmProjects\pythonProject\\venv\well_irrigation_simulation.xlsx',sheet_name='2813')
well_2756 = pd.read_excel('C:\\Users\mjh7517\PycharmProjects\pythonProject\\venv\well_irrigation_simulation.xlsx',sheet_name='2756')
well_2757 = pd.read_excel('C:\\Users\mjh7517\PycharmProjects\pythonProject\\venv\well_irrigation_simulation.xlsx',sheet_name='2757')

#import recharge data
rech = pd.read_excel('C:/Users\mjh7517\PycharmProjects\pythonProject\\venv\\precip_timeseries.xlsx', sheet_name='recharge')
temp = pd.read_excel('C:/Users\mjh7517\PycharmProjects\pythonProject\\venv\\temp_timeseries.xlsx')

#years from 2025-2100
years = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
         29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
         55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75]

#show changes in recharge
og_rech = 150         #mm/y
#make band around og_rech of plausible values from projected space
rech_min = rech.min(axis=1).tolist()
rech_max = rech.max(axis=1).tolist()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8), sharex=True)
#rech.plot(kind='line', color='royalblue', legend=False, ax=ax)
ax1.fill_between(years, rech_min, rech_max, color='royalblue', alpha=0.5)
ax1.axhline(y=og_rech, color='gold', linewidth=3)
ax1.set_xlim([1, 75])
ax1.set_title('Recharge & Sea Level Rise Model Comparisons')
ax1.set_ylabel('recharge (mm)')
ax1.yaxis.tick_right()
ax1.yaxis.set_label_position('right')


#show changes in sea level rise
low = 0.4627106
midlow = 0.7957106
mid = 1.237711
midhigh = 1.620711
high = 2.441711
origional = 1

#create time series for sea level rises
high1 = []
low1 = []
mid1 = []
midlow1 = []
midhigh1 = []
og = []

o = origional/75
oi = o
for i in range(len(years)):
    og.append(o)
    o=o+oi

h = high/75
hi = h
for i in range(len(years)):
    high1.append(h)
    h=h+hi

l = low/75
li = l
for i in range(len(years)):
    low1.append(l)
    l=l+li

m1 = mid/75
m1i = m1
for i in range(len(years)):
    mid1.append(m1)
    m1=m1+m1i

m2 = midlow/75
m2i = m2
for i in range(len(years)):
    midlow1.append(m2)
    m2=m2+m2i

m3 = midhigh/75
m3i = m3
for i in range(len(years)):
    midhigh1.append(m3)
    m3=m3+m3i

#plot the four sea level rise scenarios
#fix, ax = plt.subplots(figsize=(12,4))
ax2.plot(years, high1, label='high', color='royalblue')
ax2.plot(years, midhigh1, label='medium-high', color='royalblue')
ax2.plot(years, mid1, label='medium', color='royalblue')
ax2.plot(years, midlow1, label='medium-low', color='royalblue')
ax2.plot(years, low1, label='low', color='royalblue' )
ax2.plot(years, og, label='origional SLR', color='gold')
ax2.axhline(y=0, label='origional no SLR', color='gold')
ax2.set_xlim([1, 75])
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position('right')
ax2.set_ylabel('sea level rise (m)')
ax2.set_xlabel('years in future')
ax2.legend()
plt.show()




#show changes in well pumping for sample wells
og_2926 = -127
og_2754 = -81.9

fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(12,8), sharex=True)
well_2926.plot(kind='line', color='royalblue', legend=False, ax=ax3)
ax3.axhline(y=og_2926, color='gold')
ax3.set_title('Well 2926 & 2754 Model Comparisons')
ax3.set_ylabel('(m3/day)')
ax3.set_xlim([1, 75])

well_2754.plot(kind='line', color='royalblue', legend=False, ax=ax4)
ax4.axhline(y=og_2754, color='gold')
ax4.set_ylabel('pumping rate')
ax4.set_xlabel('years')
ax3.set_xlim([1, 75])
plt.show()

#make fill between plots
og_36 = -43.9
og_28 = -195.8
og_56 = -267
og_57 = -74.6

well27_min = well_2754.min(axis=1).tolist()
well27_max = well_2754.max(axis=1).tolist()
well29_min = well_2926.min(axis=1).tolist()
well29_max = well_2926.max(axis=1).tolist()
well36_min = well_2736.min(axis=1).tolist()
well36_max = well_2736.max(axis=1).tolist()
well28_min = well_2813.min(axis=1).tolist()
well28_max = well_2813.max(axis=1).tolist()
well56_min =well_2756.min(axis=1).tolist()
well56_max = well_2756.max(axis=1).tolist()
well57_min = well_2757.min(axis=1).tolist()
well57_max = well_2757.max(axis=1).tolist()

fig, (ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(6, 1, figsize=(12,8), sharex=True)
ax5.fill_between(years, well27_min, well27_max, color='royalblue', alpha=0.5)
ax5.axhline(y=og_2754, color='gold', linewidth=3)
ax6.fill_between(years, well29_min, well29_max, color='royalblue', alpha=0.5)
ax6.axhline(y=og_2926, color='gold', linewidth=3)
ax7.fill_between(years, well36_min, well36_max, color='royalblue', alpha=0.5)
ax7.axhline(y=og_36, color='gold', linewidth=3)
ax8.fill_between(years, well28_min, well28_max, color='royalblue', alpha=0.5)
ax8.axhline(y=og_28, color='gold', linewidth=3)
ax9.fill_between(years, well56_min, well56_max, color='royalblue', alpha=0.5)
ax9.axhline(y=og_56, color='gold', linewidth=3)
ax10.fill_between(years, well57_min, well57_max, color='royalblue', alpha=0.5)
ax10.axhline(y=og_57, color='gold', linewidth=3)

ax10.set_xlim([1, 75])
ax10.set_xlabel('years')
ax5.set_title('Wells Model Comparisons')
ax5.set_ylabel('(m3/day)')
ax6.set_ylabel('pumping rate')
plt.show()

