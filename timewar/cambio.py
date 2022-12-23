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
# F_{atm->land} = k_{al0} +  k_{al1} \times \sigma_{floor}(T_{anomaly})
#                  \times [C_{atm}] \ \ \ (2)
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
# One difference is that Cambio3 makes use of a new function that creates
# scheduledflows "on the fly" (instead of picking up a pre-existing file).
# This is the function CL.make_emissions_scenario_lte.
#
# Another difference is that Cambio3 includes feedbacks as well as diagnostics.
# One of these is shown explicitly in Eq. (2) above, where we calculate the
# flux of carbon from the atmosphere to the land. In Cambio2, this was
# just $k_{al0} +  k_{al1} \times [C_{atm}]$, meaning that a
# bigger $k_{al1}$ resulted in a bigger value of the atmosphere-to-land c
# arbon flux. That's called the $CO_2$ fertilization effect -- basically,
# because plants photosynthesize better when there's more $CO_2$ in the air.
# But it's also known that $CO_2$ fertilization suffers diminishing returns
# in a warmer climate. That's where the factor $\sigma_{floor}(T_{anomaly})$
# comes in: if the temperature anomaly rises above some threshold value
# (specified here by 'F_al_transitionT', in the climparams dictionary),
# then $\sigma_{floor}(T_{anomaly})$ drops to some new value, less than $1$,
# which in turn means that $F_{atm->land}$ goes up a little more slowly in
# response to higher $CO_2$ levels in the atmosphere.
#
# ### Using Cambio3.1
# 1. Take snapshots of one some of the results (graphs).
# 2. After modifying flags in propagate_climate_state
# (they start with "I_want ...") to activate a given feedback,
# impact, or constraint of interest, or modifying parameters in the
# climparams dictionary, you'll run the entire notebook again from top to
# bottom, and take more snapshots for comparison.
#
# Goals
# 1. Generate scheduled flows "on the fly," with a specified long term
#    emission value.
# 2. Activate various feedbacks, impacts, and constraints, including:
#    - the impact of temperature on carbon fluxes, and vice versa
#    - the impact of albedo on temperature, and vice versa
#    - stochasticity in the model
#    - constraint on how fast Earth's albedo can change


from typing import Any
import numpy as np
import numpy.typing as npt


from cambio_utils import make_emissions_scenario_lte, is_same
import preindustrial_inputs
from climate_params import ClimateParams
from cambio_utils import Diagnose_actual_temperature

#    Diagnose_degreesF,


def cambio(
    start_year: float,
    stop_year: float,
    dtime: float,
    inv_time_constant: float,
    transition_year: float,
    transition_duration: float,
    long_term_emissions: float,
    stochastic_c_atm_std_dev: float,
    albedo_with_no_constraint: bool,
    albedo_feedback: bool,
    stochastic_C_atm: bool,
    temp_anomaly_feedback: bool,
    temp_units: str,
    flux_type: str,
    plot_flux_diffs: bool,
):
    """
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
    """

    # Units of variables output by climate model:
    # C_atm and C_ocean: GtC, I think
    # T_anomaly: K
    # F_al, F_la, F_ao, F_oa, F_ha: GtC/year

    # Call the LTE emissions scenario maker with these parameters
    # time is in years
    # flux_human_atm is in GtC/year
    # would be good to output units
    time, flux_human_atm = make_emissions_scenario_lte(
        start_year,
        stop_year,
        dtime,
        inv_time_constant,
        transition_year,
        transition_duration,
        long_term_emissions,
    )

    # ### Creating a preindustrial Climate State
    # containing the climate state containing preindustrial parameters.
    # We've set the starting year to what was specified above when you
    # created your scenario.
    climate_params = preindustrial_inputs.climate_params
    climateParams = ClimateParams(stochastic_c_atm_std_dev)

    # Propagating through time

    # Make the starting state the preindustrial
    # Create an empty climate state
    climatestate: dict[str, float] = {}
    # Fill in some default (preindustrial) values
    climatestate["C_atm"] = climate_params["preindust_c_atm"]
    climatestate["C_ocean"] = climate_params["preindust_c_ocean"]
    climatestate["albedo"] = climate_params["preindust_albedo"]
    climatestate["T_anomaly"] = 0
    # These are just placeholders (values don't mean anything)
    climatestate["pH"] = 0
    climatestate["T_C"] = 0
    climatestate["T_F"] = 0
    climatestate["F_ha"] = 0
    climatestate["F_ao"] = 0
    climatestate["F_oa"] = 0
    climatestate["F_al"] = 0
    climatestate["F_la"] = 0
    climatestate["year"] = time[0] - dtime

    # Initialize the dictionary that will hold the time series
    ntimes = len(time)
    climate: dict[str, npt.NDArray[Any]] = {}
    for key in climatestate:
        climate[key] = np.zeros(ntimes)

    # Loop over all the times in the scheduled flow
    for i in range(len(time)):

        # Propagate
        climatestate = propagate_climate_state(
            climatestate, climateParams, dtime, F_ha=flux_human_atm[i]
        )

        # Append to climate variables
        for key in climatestate:
            climate[key][i] = climatestate[key]

    # QC: make sure the input and output times and human co2 emissions are same
    if not is_same(time, climate["year"]):
        raise ValueError("The input and output times differ!")
    if not is_same(flux_human_atm, climate["F_ha"]):
        raise ValueError("The input and output anthropogenic emissions differ!")

    return climate, climate_params


def propagate_climate_state(
    prev_climatestate: dict[str, float],
    climateParams: ClimateParams,
    dtime: float = 1,
    F_ha: float = 0,
    albedo_with_no_constraint: bool = False,
    albedo_feedback: bool = False,
    stochastic_C_atm: bool = False,
    temp_anomaly_feedback: bool = False,
) -> dict[str, float]:
    """
    Propagate the state of the climate, with a specified anthropogenic
    carbon flux

    @param prev_climatestate
    @param ClimateParams  Climate params class
    @param climparams, dtime, F_ha
    @returns dictionary of climate state

    Default anthropogenic carbon flux is zero
    Default time step is 1 year
    Returns a new climate state
    """

    # More inputs (for feedbacks and etc)

    # Extract concentrations from the previous climate state
    c_atm = prev_climatestate["C_atm"]
    c_ocean = prev_climatestate["C_ocean"]

    # Get the temperature anomaly resulting from carbon concentrations
    t_anom = climateParams.diagnose_temp_anomaly(c_atm)

    # Get fluxes (optionally activating the impact temperature has on them)
    if temp_anomaly_feedback:
        F_oa = climateParams.diagnose_flux_ocean_atm(c_ocean, t_anom)
        F_al = climateParams.diagnose_flux_atm_land(t_anom, c_atm)
    else:
        F_oa = climateParams.diagnose_flux_ocean_atm(c_ocean, 0)
        F_al = climateParams.diagnose_flux_atm_land(0, c_atm)

    # Get other fluxes resulting from carbon concentrations
    F_ao = climateParams.diagnose_flux_atm_ocean(c_atm)
    F_la = climateParams.diagnose_flux_land_atm()

    # Update concentrations of carbon based on these fluxes
    c_atm += (F_la + F_oa - F_ao - F_al + F_ha) * dtime
    c_ocean += (F_ao - F_oa) * dtime

    # Get albedo from temperature anomaly (optionally activating a
    # constraint in case it's changing too fast)
    if albedo_with_no_constraint:
        albedo = climateParams.diagnose_albedo_w_constraint(
            t_anom, prev_climatestate["albedo"], dtime
        )
    else:
        albedo = climateParams.diagnose_albedo_w_constraint(t_anom)

    # Get a new temperature anomaly as impacted by albedo (if we want it)
    if albedo_feedback:
        t_anom += climateParams.diagnose_delta_t_from_albedo(albedo)

    # Stochasticity in the model (if we want it)
    if stochastic_C_atm:
        c_atm = climateParams.diagnose_stochastic_c_atm(c_atm)

    # Ordinary diagnostics
    pH = climateParams.diagnose_ocean_surface_ph(c_atm)
    T_C = Diagnose_actual_temperature(t_anom)
    # T_F = Diagnose_degreesF(T_C)

    # Create a new climate state with these updates
    climatestate: dict[str, float] = {}
    climatestate["C_atm"] = c_atm
    climatestate["C_ocean"] = c_ocean
    climatestate["albedo"] = albedo
    climatestate["T_anomaly"] = t_anom
    climatestate["pH"] = pH
    climatestate["T_C"] = T_C
    # climatestate["T_F"] = T_F
    climatestate["F_ha"] = F_ha
    climatestate["F_ao"] = F_ao
    climatestate["F_oa"] = F_oa
    climatestate["F_la"] = F_la
    climatestate["F_al"] = F_al
    climatestate["year"] = prev_climatestate["year"] + dtime

    # Return the new climate state
    return climatestate
