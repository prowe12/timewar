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
# Once you've gone through the activites prompted below, the idea is to:
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


from cambio_utils import make_emissions_scenario_lte
from cambio_utils import CreateClimateState
from cambio_utils import CollectClimateTimeSeries
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


# # Parameters of the LTE emissions schedule
# SF_t_start = 1750
# SF_t_stop = 2300
# SF_nsteps = 1000
# SF_k = 0.025
# SF_t_trans = 2050
# SF_delta_t_trans = 20
# SF_t_0 = 2020
# SF_eps_0 = 11.3
# sf_long_term_emissions = 10

# # Calling the LTE emissions scenario maker
# time, eps = make_emissions_scenario_lte(
#     SF_t_start,
#     SF_t_stop,
#     SF_nsteps,
#     SF_k,
#     SF_eps_0,
#     SF_t_0,
#     SF_t_trans,
#     SF_delta_t_trans,
#     sf_long_term_emissions,
# )

# # Plot Anthropogenic Emissions in units GtC/year
# plt.figure()
# plt.plot(time, eps)
# plt.grid(True)
# plt.title("Anthropogenic Emissions")
# plt.xlabel("year")
# plt.ylabel("GtC/year")

# # Also plot in GtCO2/year (for easier comparison with EnROADS)
# plt.figure()
# plt.plot(time, eps / 0.27)
# plt.grid(True)
# plt.title("Anthropogenic Emissions")
# plt.xlabel("year")
# plt.ylabel("GtCO2/year")


# # # # # #     User inputs    # # # # #
# For the LTE emissions maker
SF_t_start = 1750
SF_t_stop = 2200
SF_nsteps = 1000
SF_k = 0.025
SF_t_trans = 2040  # pivot year to start decreasing CO2???
SF_delta_t_trans = 20  # years over which to decrease co2
SF_t_0 = 2020  # pivot year to start decreasing CO2???
SF_eps_0 = 11.3  # ???
sf_long_term_emissions = 2  # ongoing carbon emissions after decarbonization
# For feedbacks and stochastic runs
stochastic_c_atm_std_dev = 0.1
albedo_with_no_constraint = False
albedo_feedback = False
stochastic_C_atm = False
temp_anomaly_feedback = False
# Units
temp_units = "F"  # F, C, or K
c_units = "GtC"  # GtC, or GtCO2
# # # # # #  # # # # # # # # # # # #


# Unit conversions
c_conversion_fac = {"GtC": 1, "GtCO2": 1 / 0.27}  # from GtC


# Call the LTE emissions scenario maker with these parameters
time, flux_human_atm = make_emissions_scenario_lte(
    SF_t_start,
    SF_t_stop,
    SF_nsteps,
    SF_k,
    SF_eps_0,
    SF_t_0,
    SF_t_trans,
    SF_delta_t_trans,
    sf_long_term_emissions,
)


# ### Creating a preindustrial Climate State
# The cell below uses CreateClimateState to create a dictionary
# containing the climate state containing preindustrial parameters.
# We've set the starting year to what was specified above when you
# created your scenario.
climate_params = preindustrial_inputs.climate_params
climateParams = ClimateParams(stochastic_c_atm_std_dev)


# Propagating through time
# Initialize our list of climate states
climatestate_list = []

# Make the starting state the preindustrial
climatestate = CreateClimateState(climate_params)

# Add some times
# This sets the starting year the same as the scheduled flow
climatestate["year"] = time[0]
dt = time[1] - time[0]

# Loop over all the times in the scheduled flow
for i in range(len(time)):

    # Propagate
    climatestate = propagate_climate_state(
        climatestate, climateParams, dtime=dt, F_ha=flux_human_atm[i]
    )

    # Add to our list of climate states
    climatestate_list.append(climatestate)


# ### Visualizing the results of the run
# Below, we use CollectClimateTimeSeries to collect the time array
# from our results. We also collect the albedo, and do some fancy cosmetics,
# like making the lines a little thicker than the default, and specifying
# the vertical scale so that it displays the entire possible range of albedos
# a little better.


# Plot the Anthropogenic emissions
plt.figure()
plt.plot(time, flux_human_atm * c_conversion_fac[c_units])
plt.grid(True)
plt.title("Anthropogenic Emissions")
plt.xlabel("year")
plt.ylabel(c_units + "/year")

# Here we are extracting the times from climatestate_list
time = CollectClimateTimeSeries(climatestate_list, "year")

# Extract and plot the albedo
albedo_array = CollectClimateTimeSeries(climatestate_list, "albedo")

plt.figure()
plt.plot(time, albedo_array, label="albedo", linewidth=2)
plt.grid(True)
plt.xlabel("time (years)")
plt.ylabel("albedo")
plt.legend()
ybottom = albedo_array[0] * climate_params["fractional_albedo_floor"] - 0.01
ytop = albedo_array[0] + 0.001
plt.ylim([ybottom, ytop])


# Extract and plot the ocean pH, specifying vertical axis limits of 7.8 to 8.3
T_array = CollectClimateTimeSeries(climatestate_list, "pH")
plt.figure()
linewidth = 2
plt.plot(time, T_array, label="pH", linewidth=linewidth, color="gray")
plt.grid(True)
plt.xlabel("time (years)")
plt.legend()
ybottom = 7.8
ytop = 8.3
plt.ylim([ybottom, ytop])


# Extract and plot the concentration of carbon in the atmosphere and oceans,
# in GtC (one graph)
C_atm_array = CollectClimateTimeSeries(climatestate_list, "C_atm")
C_ocean_array = CollectClimateTimeSeries(climatestate_list, "C_ocean")
plt.figure()
plt.plot(time, C_atm_array, label="[C_atm](GtC)", linewidth=linewidth)
plt.plot(time, C_ocean_array, label="C_ocean(GtC)", linewidth=linewidth)
plt.grid(True)
plt.xlabel("time (years)")
plt.ylabel("GtC")
plt.legend()

# Re-plot the carbon in the atmosphere, converted to ppm (by dividing
# C_atm_array by 2.12)
plt.figure()
plt.plot(time, C_atm_array / 2.12, label="[C_atm](ppm)", linewidth=linewidth)
plt.grid(True)
plt.xlabel("time (years)")
plt.ylabel("ppm")
plt.legend()


# Extract and plot the temperature anomaly
T_array = CollectClimateTimeSeries(climatestate_list, "T_anomaly")
plt.figure()
plt.plot(
    time,
    T_array,
    label="Temperature anomaly",
    linewidth=linewidth,
    color="red",
)
plt.grid(True)
plt.xlabel("time (years)")
plt.ylabel("degrees K")
plt.legend()


# Extract the fluxes, compute net fluxes, and plot them
F_al_array = CollectClimateTimeSeries(climatestate_list, "F_al")
F_la_array = CollectClimateTimeSeries(climatestate_list, "F_la")
F_ao_array = CollectClimateTimeSeries(climatestate_list, "F_ao")
F_oa_array = CollectClimateTimeSeries(climatestate_list, "F_oa")
F_ha_array = CollectClimateTimeSeries(climatestate_list, "F_ha")
plt.figure()
# fontsize=12
# plt.rcParams.update({'font.size': fontsize})
plt.plot(time, F_ha_array, label="F_ha", color="black", linewidth=linewidth)
plt.plot(
    time,
    -F_al_array + F_la_array,
    label="F_la-F_al",
    color="brown",
    linewidth=linewidth,
)
plt.plot(
    time,
    -F_ao_array + F_oa_array,
    label="F_oa-F_ao",
    color="blue",
    linewidth=linewidth,
)
plt.grid(True)
plt.xlabel("time (years)")
plt.ylabel("Flux differences (GtC/year)")
plt.legend()
