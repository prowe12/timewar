#!/usr/bin/env python
# coding: utf-8

# # Cambio 3.1
#
# ## Cambio 3 equations of motion
# The equations of motion for Cambio3 are as follows:
#
# $$
# F_{land->atm} =  k_{la} \ \ \ (1)
# $$
#
# $$
# F_{atm->land} = k_{al0} +  k_{al1} \times \sigma_{floor}(T_{anomaly}) \times [C_{atm}] \ \ \ (2)
# $$
#
# $$
# F_{ocean->atm} = k_{oa} [C_{ocean}] \ \ \ (3)
# $$
#
# $$
# F_{atm->ocean} = k_{ao} [C_{atm}] \ \ \ (4)
# $$
#
# $$
# F_{human->atm} = \epsilon(t) \ \ \ (5)
# $$
#
#
# ### Significant differences between Cambio2 and Cambio3
#
# One difference is that Cambio3 makes use of a new function that
# creates scheduledflows "on the fly" (instead of picking up a
# pre-existing file). This is the function make_emissions_scenario_lte.
#
# Another difference is that Cambio3 includes feedbacks as well as
# diagnostics. One of these is shown explicitly in Eq. (2) above,
# where we calculate the flux of carbon from the atmosphere to the land.
# In Cambio2, this was just $k_{al0} +  k_{al1} \times [C_{atm}]$, meaning
# that a bigger $k_{al1}$ resulted in a bigger value of the atmosphere-
# to-land carbon flux. That's called the $CO_2$ fertilization effect
# -- basically, because plants photosynthesize better when there's
# more $CO_2$ in the air. But it's also known that $CO_2$ fertilization
# suffers diminishing returns in a warmer climate. That's where the
# factor $\sigma_{floor}(T_{anomaly})$ comes in: if the temperature
# anomaly rises above some threshold value (specified here by
# 'F_al_transitionT', in the climate_params dictionary), then
# $\sigma_{floor}(T_{anomaly})$ drops to some new value,
# less than $1$, which in turn means that $F_{atm->land}$ goes up
# a little more slowly in response to higher $CO_2$ levels in the atmosphere.
#
# ### How you're going to use Cambio3.1
# Modify flags in propagate_climate_state. to activate a given feedback,
# impact, or constraint of interest, or modifying parameters in the
# climate_params dictionary, you'll run the entire notebook again from
# top to bottom, and take more snapshots for comparison.
#
# ### Goals
# 1. Generate scheduled flows "on the fly," with a specified long term
#    emission value.
# 2. Activate various feedbacks, impacts, and constraints, including:
#    - the impact of temperature on carbon fluxes, and vice versa
#    - the impact of albedo on temperature, and vice versa
#    - stochasticity in the model
#    - constraint on how fast Earth's albedo can change


import matplotlib.pyplot as plt
import numpy as np


from cambio_utils import make_emissions_scenario_lte, is_same
from cambio import propagate_climate_state
import preindustrial_inputs
from climate_params import ClimateParams


# ### Introducing the "LTE" emissions scenario maker
# The emissions scenario maker.
#
# 1. This is a function call to "make_emissions_scenario_lte".
# 2. LTE stands for "long-term-emissions." That's specified by SF_LTE.
#    In the cell below, we specify 10 GtC as a default, but it can be
#    anything (even negative). If you want to reproduce our original,
#    more or less, you can specify SF_LTE=0.
#
# After generating the scenario, we plot the emissions in GtC/year, and again in GtCO2/year, by dividing by 0.27; the latter is so that we can compare to other models, like EnROADS, which use GtCO2.


# # # # # #     User inputs    # # # # #
# For the LTE emissions maker
start_year = 1750.0
stop_year = 2200.0
dtime = 1.0  # time resolution (years)
sf_k = 0.025
SF_t_trans = 2040.0  # pivot year to start decreasing CO2???
SF_delta_t_trans = 20.0  # years over which to decrease co2
SF_t_0 = 2020.0  # pivot year to start decreasing CO2???
SF_eps_0 = 11.3  # ???
sf_long_term_emissions = 2  # ongoing carbon emissions after decarbonization
# For feedbacks and stochastic runs
stochastic_c_atm_std_dev = 0.1
albedo_with_no_constraint = False
albedo_feedback = False
stochastic_C_atm = False
temp_anomaly_feedback = False
# Desired units
temp_units = "F"  # F, C, or K
c_units = "GtC"  # GtC, GtCO2, atm
flux_type = "/year"  # total, per year
plot_flux_diffs = True  # True, False
# # # # # #  # # # # # # # # # # # #


# Units of variables output by climate model:
# C_atm and C_ocean: GtC, I think
# T_anomaly: K
# F_al, F_la, F_ao, F_oa, F_ha: GtC/year


# Unit conversions
# Convert carbon amounts from GtC
c_conversion_fac = {"GtC": 1, "GtCO2": 1 / 0.27, "ppm": 1 / 2.12}
# TODO: Add conversion for F
temp_conversion_fac = {"K": 1, "C": 273.15, "F": np.nan}
# TODO: Do not use for anomaly, since K and C will be the same.
#       need fancy code to convert to F


# Call the LTE emissions scenario maker with these parameters
# time is in years
# flux_human_atm is in GtC/year
# would be good to output units
time, flux_human_atm = make_emissions_scenario_lte(
    start_year,
    stop_year,
    dtime,
    sf_k,
    SF_eps_0,
    SF_t_0,
    SF_t_trans,
    SF_delta_t_trans,
    sf_long_term_emissions,
)


# ### Creating a preindustrial Climate State
# containing the climate state containing preindustrial parameters.
# We've set the starting year to what was specified above when you
# created your scenario.
climate_params = preindustrial_inputs.climate_params
climateParams = ClimateParams(stochastic_c_atm_std_dev)


# Propagating through time
# Initialize our list of climate states
climatestate_list = []

# Make the starting state the preindustrial
# ['C_atm', 'C_ocean', 'albedo', 'T_anomaly', 'pH',
#  'T_C', 'T_F', 'F_ha', 'F_ao', 'F_oa', 'F_al', 'F_la']

# Create an empty climate state
ClimateState = {}
# Fill in some default (preindustrial) values
ClimateState["C_atm"] = climate_params["preindust_c_atm"]
ClimateState["C_ocean"] = climate_params["preindust_c_ocean"]
ClimateState["albedo"] = climate_params["preindust_albedo"]
ClimateState["T_anomaly"] = 0
# These are just placeholders (values don't mean anything)
ClimateState["pH"] = 0
ClimateState["T_C"] = 0
ClimateState["T_F"] = 0
ClimateState["F_ha"] = 0
ClimateState["F_ao"] = 0
ClimateState["F_oa"] = 0
ClimateState["F_al"] = 0
ClimateState["F_la"] = 0
ClimateState["year"] = time[0] - dtime

climatestate = ClimateState

# Add some times
# This sets the starting year the same as the scheduled flow
dt = time[1] - time[0]

# Initialize the dictionary that will hold the time series
ntimes = len(time)
climate = {}
for key in climatestate:
    climate[key] = np.zeros(ntimes)


# Loop over all the times in the scheduled flow
for i in range(len(time)):

    # Propagate
    climatestate = propagate_climate_state(
        climatestate, climateParams, dtime=dt, F_ha=flux_human_atm[i]
    )

    # Add to our list of climate states
    climatestate_list.append(climatestate)

    for key in climatestate:
        climate[key][i] = climatestate[key]


# QC: make sure the input and output times and human co2 emissions are same
if not is_same(time, climate["year"]):
    raise ValueError("The input and output times differ!")
if not is_same(flux_human_atm, climate["F_ha"]):
    raise ValueError("The input and output anthropogenic emissions differ!")


# # # #   Visualize the results of the run   # # # #
# Plotting parameters:
linewidth = 2
lwidth = 2

# Plot Anthropogenic emissions in GtC/year
c_units = "GtC"  # GtC, GtCO2, atm
flux_type = "/year"  # total, per year
yvals = [flux_human_atm]
lables = ["Anthropogenic Emissions"]
plt.figure()
for i, yval in enumerate(yvals):
    plt.plot(time, yval * c_conversion_fac[c_units], label=lables[i])
plt.legend()
plt.grid(True)
plt.xlabel("year")
plt.ylabel(c_units + flux_type)

# Plot the concentration of carbon in the atmosphere and oceans,
# in GtC (one graph)
c_units = "GtC"
plot_me = ["C_atm", "C_ocean"]
labels = ["C_atm", "C_ocean"]
flux_type = ""
plt.figure()
for i, varname in enumerate(plot_me):
    label = f"{labels[i]} ({c_units})"
    yval = climate[varname] * c_conversion_fac[c_units]
    plt.plot(time, yval, label=label, linewidth=lwidth)
plt.grid(True)
plt.xlabel("time (years)")
plt.ylabel(c_units + flux_type)
plt.legend()

# Re-plot the carbon in the atmosphere, converted to ppm (by dividing
# C_atm_array by 2.12)
c_units = "ppm"
plot_me = ["C_atm"]
labels = ["C_atm"]
flux_type = ""
plt.figure()
for i, varname in enumerate(plot_me):
    yval = climate[varname] * c_conversion_fac[c_units]
    label = f"{labels[i]} ({c_units})"
    plt.plot(time, yval, label=label, linewidth=lwidth)
plt.grid(True)
plt.xlabel("time (years)")
plt.ylabel(c_units + flux_type)
plt.legend()

# Extract and plot the albedo
plot_me = ["albedo"]
labels = ["Albedo"]
plt.figure()
for i, varname in enumerate(plot_me):
    yval = climate[varname]
    plt.plot(time, yval, label=f"{labels[i]}", linewidth=linewidth)
plt.grid(True)
plt.xlabel("time (years)")
plt.ylabel("albedo")
plt.legend()
# TODO: Find a better solution for this
ybottom = yval[0] * climate_params["fractional_albedo_floor"] - 0.01
ytop = yval[0] + 0.001
plt.ylim([ybottom, ytop])

# plot the ocean pH, specifying vertical axis limits of 7.8 to 8.3
varname = "pH"
yval = climate[varname]
plt.figure()
plt.plot(time, yval, label="pH", linewidth=linewidth, color="gray")
plt.grid(True)
plt.xlabel("time (years)")
plt.ylabel(varname)
plt.legend()
# TODO: Find a better way to set the ylims
ybottom = 7.8
ytop = 8.3
plt.ylim([ybottom, ytop])

# TODO: Fix the way colors are handled

# Extract and plot the temperature anomaly
varname = "T_anomaly"
yval = climate[varname]
label = "Temperature anomaly"
plt.figure()
plt.plot(time, yval, label=label, linewidth=linewidth, color="red")
plt.grid(True)
plt.xlabel("time (years)")
plt.ylabel("degrees K")
plt.legend()

# Extract the fluxes, compute net fluxes, and plot them
plot_flux_diffs = True  # True, False
plot_me = ["F_ha", "F_oa", "F_la"]
labels = ["F_ha", "F_la-F_al", "F_oa-F_ao"]
colors = ["black", "brown", "blue"]
if plot_flux_diffs:
    ylabel = "Flux differences (GtC/year)"
else:
    raise ValueError("option not here yet")
plt.figure()
for i, varname in enumerate(plot_me):
    if plot_flux_diffs and varname == "F_oa":
        yval = -climate["F_ao"] + climate["F_oa"]
    elif plot_flux_diffs and varname == "F_la":
        yval = -climate["F_al"] + climate["F_la"]
    else:
        yval = climate[varname]
    plt.plot(time, yval, label=labels[i], color=colors[i], linewidth=lwidth)
plt.grid(True)
plt.xlabel("time (years)")
plt.ylabel(ylabel)
plt.legend()
