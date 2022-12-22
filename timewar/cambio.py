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


from copy import copy as makeacopy

from cambio_utils import (
    Diagnose_T_anomaly,
    Diagnose_F_oa,
    Diagnose_F_al,
    Diagnose_F_ao,
    Diagnose_F_la,
    Diagnose_albedo_with_constraint,
    Diagnose_albedo,
    Diagnose_Delta_T_from_albedo,
    Diagnose_Stochastic_C_atm,
    Diagnose_actual_temperature,
    Diagnose_degreesF,
)


def propagate_climate_state(
    prevClimateState,
    climateParams,
    climparams,
    dtime=1,
    F_ha=0,
    albedo_with_no_constraint=False,
    albedo_feedback=False,
    stochastic_C_atm=False,
    temp_anomaly_feedback=False,
):
    """
    Propagate the state of the climate, with a specified anthropogenic
    carbon flux

    @param prevClimateState
    @param ClimateParams  Climate params class
    @param climparams, dtime, F_ha
    @returns dictionary of climate state

    Default anthropogenic carbon flux is zero
    Default time step is 1 year
    Returns a new climate state
    """

    # More inputs (for feedbacks and etc)

    # Extract concentrations from the previous climate state
    C_atm = prevClimateState["C_atm"]
    C_ocean = prevClimateState["C_ocean"]

    # Get the temperature anomaly resulting from carbon concentrations
    T_anomaly = Diagnose_T_anomaly(C_atm, climparams)

    # Get fluxes (optionally activating the impact temperature has on them)
    if temp_anomaly_feedback:
        F_oa = Diagnose_F_oa(C_ocean, T_anomaly, climparams)
        F_al = Diagnose_F_al(T_anomaly, C_atm, climparams)
    else:
        F_oa = Diagnose_F_oa(C_ocean, 0, climparams)
        F_al = Diagnose_F_al(0, C_atm, climparams)

    # Get other fluxes resulting from carbon concentrations
    F_ao = Diagnose_F_ao(C_atm, climparams)
    F_la = Diagnose_F_la(climparams)

    # Update concentrations of carbon based on these fluxes
    C_atm += (F_la + F_oa - F_ao - F_al + F_ha) * dtime
    C_ocean += (F_ao - F_oa) * dtime

    # Get albedo from temperature anomaly (optionally activating a
    # constraint in case it's changing too fast)
    if albedo_with_no_constraint:
        albedo = Diagnose_albedo_with_constraint(
            T_anomaly, climparams, prevClimateState["albedo"], dtime
        )
    else:
        albedo = Diagnose_albedo(T_anomaly, climparams)

    # Get a new temperature anomaly as impacted by albedo (if we want it)
    if albedo_feedback:
        T_anomaly += Diagnose_Delta_T_from_albedo(albedo, climparams)

    # Stochasticity in the model (if we want it)
    if stochastic_C_atm:
        C_atm = Diagnose_Stochastic_C_atm(C_atm, climparams)

    # Ordinary diagnostics
    pH = climateParams.diagnose_ocean_surface_ph(C_atm)
    # pH = Diagnose_OceanSurfacepH(C_atm, climparams)

    T_C = Diagnose_actual_temperature(T_anomaly)
    T_F = Diagnose_degreesF(T_C)

    # Create a new climate state with these updates
    ClimateState = makeacopy(prevClimateState)
    ClimateState["C_atm"] = C_atm
    ClimateState["C_ocean"] = C_ocean
    ClimateState["F_oa"] = F_oa
    ClimateState["F_ao"] = F_ao
    ClimateState["F_la"] = F_la
    ClimateState["F_al"] = F_al
    ClimateState["year"] += dtime
    ClimateState["F_ha"] = F_ha
    ClimateState["T_anomaly"] = T_anomaly
    ClimateState["pH"] = pH
    ClimateState["T_C"] = T_C
    ClimateState["T_F"] = T_F
    ClimateState["albedo"] = albedo

    # Return the new climate state
    return ClimateState
