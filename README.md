# mod_inputs
code and excel files to be used for modflow inputs

using p_t_data repository
scenerio generation through sampling, scaling factors, and/or markov chain

***important here***
PRECIPITATION AND TEMPERATURE:
sample scenario generation has majority of the important work for creating precipitation and temperature time series to the end of the century using downscaled CMIP6 data downscaled to the USA Mid-Atlantic, downloaded from NOAA, hist data also from NOAA for Dover, DE
it takes excel files noaa_hist_new, all_p_p, all_t_p
it outputs files mean_bc_P, mean_bc_t as the bias corrected data
it also outputs percip_timeseries and temp_timeseries as the generated time series created from the code


SEA LEVEL RISE:
slr_timeseries.py has all the sea level rise data, using information downloaded from the 2022 Global and Regional Sea Level Rise Scenarios for the United States, downsalced to a tide gauge south of Dover, DE


RECHARGE AND IRRIGATION:
recharge calculated to be 13% of precipitation
precipitation and temperature are then used to calculate irrigation needs by calculating the evapotranspiration using the FAO-Blaney-Criddle method as outlined in the 1993 USDA Chapteer 2 Irrigation Water Requierments
irrig_calc.py with inputs from temp_timeseries and precip_timeseries (with using only that 13% of precip)
