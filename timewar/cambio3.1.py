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


from cambio import cambio

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
inv_time_constant = 0.025
transition_year = 2040.0  # pivot year to start decreasing CO2
transition_duration = 20.0  # years over which to decrease co2
long_term_emissions = 2.0  # ongoing carbon emissions after decarbonization
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

climate, climate_params = cambio(
    start_year,
    stop_year,
    dtime,
    inv_time_constant,
    transition_year,
    transition_duration,
    long_term_emissions,
    stochastic_c_atm_std_dev,
    albedo_with_no_constraint,
    albedo_feedback,
    stochastic_C_atm,
    temp_anomaly_feedback,
    temp_units,
    flux_type,
    plot_flux_diffs,
)


# Test - recreate Steven's plots and make sure they look ok
from make_plots_like_stevens import make_plots_like_stevens

make_plots_like_stevens(climate, climate_params["fractional_albedo_floor"])
