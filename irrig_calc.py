#August 22, 2024
#method to calculate the amount each well will need to irrigate based off of precipitation, temperature, plot size, etc
#using recharge from precip_timeseries, temp from temp_timeseries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#import rehcarge and temperature data
temp = pd.read_excel('C:/Users\mjh7517\PycharmProjects\pythonProject\\venv\\temp_timeseries.xlsx')
rech = pd.read_excel('C:/Users\mjh7517\PycharmProjects\pythonProject\\venv\\precip_timeseries.xlsx', sheet_name='recharge')
wells = pd.read_excel('C:/Users\mjh7517\OneDrive - The Pennsylvania State University\Downloads\SWI_Research\irrigWellsData.xls')

rootdepth = 500    #in mm
fc = 1.865
wp = 0.9917
raw = rootdepth * (fc - wp)


# using the Thornthwaite method for calculating Evapotranspiration, adjusted to annual scale, not monthly
def apply_et(t):
    I = 12* ((t/5)**1.514)
    alpha = (6.75e-7 * I**3) - (7.71e-5 * I**2) + (0.01792 * I) + (0.49239)

    return (16 * (10 * t / I)**alpha) * 12    #*12 to get annual

eto = apply_et(temp)
print(eto)

#cumulative crop coeficients for all plants in Dover with well, cumulative across all growth stages, Kc
corn_kc = 1.2
soy_kc = 1.65
potato_kc = 1.90
wwheat_kc = 2.175
hay_kc = 2.52

#STEP 1: calc the evapotranspiration of each crop type, ETc
def crop_et(et, crop):
    return et * crop

et_corn = pd.DataFrame(crop_et(eto, corn_kc))
et_soy = pd.DataFrame(crop_et(eto, soy_kc))
et_potato = pd.DataFrame(crop_et(eto, potato_kc))
et_wwheat = pd.DataFrame(crop_et(eto, wwheat_kc))
et_hay = pd.DataFrame(crop_et(eto, hay_kc))


#STEP 2: get net irrigation requierement by subtract ETc from recharge, NIR
nir_corn = pd.DataFrame(np.nan, index=rech.index, columns=rech.columns)
nir_soy = pd.DataFrame(np.nan, index=rech.index, columns=rech.columns)
nir_potato = pd.DataFrame(np.nan, index=rech.index, columns=rech.columns)
nir_wwheat = pd.DataFrame(np.nan, index=rech.index, columns=rech.columns)
for row in range(rech.shape[0]):
    for col in range(rech.shape[1]):
        nir_corn.iloc[row, col] = (et_corn.iloc[row, col] - rech.iloc[row, col])
        nir_soy.iloc[row, col] = (et_soy.iloc[row, col] - rech.iloc[row, col])
        nir_potato.iloc[row, col] = (et_potato.iloc[row, col] - rech.iloc[row, col])
        nir_wwheat.iloc[row, col] = (et_wwheat.iloc[row, col] - rech.iloc[row, col])

print(nir_corn)
print(nir_potato)
#creat demand for double crops whioch are all soy and  winter wheat
nir_dbl = nir_soy + nir_wwheat

#STEP 3: calc gross irrigation requirement, NIR/efficeincy factor, GIR
eff = 0.13   #efficiency factor, 13% of precip = rech
'''gir_corn = pd.DataFrame(np.nan, index=rech.index, columns=rech.columns)
gir_soy = pd.DataFrame(np.nan, index=rech.index, columns=rech.columns)
gir_potato = pd.DataFrame(np.nan, index=rech.index, columns=rech.columns)
gir_wwheat = pd.DataFrame(np.nan, index=rech.index, columns=rech.columns)
gir_dbl = pd.DataFrame(np.nan, index=rech.index, columns=rech.columns)         #gir for double crop farms (soy and wwheat)

for i in range(nir_corn.shape[0]):
    for j in range(nir_corn.shape[1]):
        gir_corn.iloc[i, j] = nir_corn.iloc[i, j] / eff
for i in range(nir_soy.shape[0]):
    for j in range(nir_soy.shape[1]):
        gir_soy.iloc[i, j] = nir_soy.iloc[i, j] / eff
for i in range(nir_potato.shape[0]):
    for j in range(nir_potato.shape[1]):
        gir_potato.iloc[i, j] = nir_potato.iloc[i, j] / eff
for i in range(nir_wwheat.shape[0]):
    for j in range(nir_wwheat.shape[1]):
        gir_wwheat.iloc[i, j] = nir_wwheat.iloc[i, j] / eff
#double crop of soy and winter wheat
gir_dbl = gir_soy + gir_wwheat

print(gir_potato)
print(gir_dbl)
'''
#STEP 4: calc volume of water to be pumped, GIR*area, vol
#make a dictionary to  hold the simulations for each well
well_dict = {}

for i, row in wells.iterrows():
    key = row['name']    #set keys to dict as well ids

    results = [[] for _ in range(nir_corn.shape[1])]
    for i in range(len(nir_corn)):
            for j, col in enumerate(nir_corn.columns):

                if row['crop1'] == 'corn' and row['crop2'] == 'corn':
                    d = (nir_corn[col].iloc[i] * row['area1'] + nir_corn[col].iloc[i] * row['area2']).tolist()
                    results[j].append(d / 1000 * 0.26)
                elif row['crop1'] == 'dbl' and row['crop2'] == 'wwheat':
                    d = (nir_dbl[col].iloc[i] * row['area1'] + nir_wwheat[col].iloc[i] + row['area2']).tolist()
                    results[j].append(d / 1000)
                elif row['crop1'] == 'soy' and row['crop2'] == 'dbl':
                    d = (nir_soy[col].iloc[i] * row['area1'] + nir_dbl[col].iloc[i] * row['area2']).tolist()
                    results[j].append(d / 1000)
                elif row['crop1'] == 'corn' and row['crop2'] == 'dbl':
                    d = (nir_corn[col].iloc[i] * row['area1'] + nir_dbl[col].iloc[i] * row['area2']).tolist()
                    results[j].append(d / 1000)
                elif row['crop1'] == 'soy' and row['crop2'] == 'corn':
                    d = (nir_soy[col].iloc[i] * row['area1'] + nir_corn[col].iloc[i] * row['area2']).tolist()
                    results[j].append(d / 1000 * 0.38)

                elif row['crop1'] == 'corn':
                    d = (nir_corn[col].iloc[i] * row['area1']).tolist()
                    results[j].append(d / 1000 * 0.26)
                elif row['crop1'] == 'soy':
                    d = (nir_soy[col].iloc[i] * row['area1']).tolist()
                    results[j].append(d / 1000 * 0.38)
                elif row['crop1'] == 'potato':
                    d = (nir_potato[col].iloc[i] * row['area1']).tolist()
                    results[j].append(d / 1000 * 0.36)
                elif row['crop1'] == 'dbl':
                    d = (nir_dbl[col].iloc[i] * row['area1']).tolist()
                    results[j].append(d / 1000 * 0.79)

    well_dict[key] = results

print(well_dict)
well_dict = {key: list(map(list, zip(*val))) for key, val in well_dict.items()}

#STEP 5: convert volume to pumping rate, vol/T
#divide by 365 for days and divide by 1000 to go from mm to meters
for key in well_dict:
    well_dict[key] = [[round(val / 365 * -1) for val in sublist] for sublist in well_dict[key]]


#save the dictionary to an excel file with each well being its own sheet
with pd.ExcelWriter('well_irrigation_simulation.xlsx') as writer:
    for key, val in well_dict.items():
        df = pd.DataFrame(val)
        df.to_excel(writer, sheet_name=str(key), index=False)

