#!/usr/bin/env python
# coding: utf-8
"""
Created on Wed Dec 21 15:49:24 2022

@author: nesh

By Steven Neshyba
With modifications by Penny Rowe and Daniel Neshyba-Rowe
"""
from copy import deepcopy as makeacopy
from typing import Any
import numpy as np

from climate_params import ClimateParams


def propagate_climate_state(
    prevClimateState: dict[str, Any],
    climateParams: ClimateParams,
    dtime: float = 1,
    F_ha: float = 0,
    albedo_with_no_constraint: bool = False,
    albedo_feedback: bool = False,
    stochastic_C_atm: bool = False,
    temp_anomaly_feedback: bool = False,
) -> dict[str, Any]:
    """
    Propagate the state of the climate, with a specified anthropogenic
    carbon flux

    @param prevClimateState
    @param climateParams  Climate params class
    @param climparams, dtime, F_ha
    @returns dictionary of climate state

    Default anthropogenic carbon flux is zero
    Default time step is 1 year
    Returns a new climate state
    """

    # More inputs (for feedbacks and etc)

    # Extract concentrations from the previous climate state
    c_atm = prevClimateState["C_atm"]
    c_ocean = prevClimateState["C_ocean"]

    # Get the temperature anomaly resulting from carbon concentrations
    # T_anomaly = Diagnose_T_anomaly(c_atm, climparams)
    T_anomaly = climateParams.diagnose_temp_anomaly(c_atm)

    # Get fluxes (optionally activating the impact temperature has on them)
    if temp_anomaly_feedback:
        # F_oa = Diagnose_F_oa(c_ocean, T_anomaly, climparams)
        # F_al = Diagnose_F_al(T_anomaly, c_atm, climparams)
        F_oa = climateParams.diagnose_flux_ocean_atm(c_ocean, T_anomaly)
        F_al = climateParams.diagnose_flux_atm_land(T_anomaly, c_atm)
    else:
        # F_oa = Diagnose_F_oa(c_ocean, 0, climparams)
        # F_al = Diagnose_F_al(0, c_atm, climparams)
        F_oa = climateParams.diagnose_flux_ocean_atm(c_ocean, 0)
        F_al = climateParams.diagnose_flux_atm_land(0, c_atm)

    # Get other fluxes resulting from carbon concentrations
    # F_ao = Diagnose_F_ao(c_atm, climparams)
    # F_la = Diagnose_F_la(climparams)
    F_ao = climateParams.diagnose_flux_atm_ocean(c_atm)
    F_la = climateParams.diagnose_flux_land_atm()

    # Update concentrations of carbon based on these fluxes
    c_atm += (F_la + F_oa - F_ao - F_al + F_ha) * dtime
    c_ocean += (F_ao - F_oa) * dtime

    # Get albedo from temperature anomaly (optionally activating a
    # constraint in case it's changing too fast)
    if albedo_with_no_constraint:
        # albedo = Diagnose_albedo_with_constraint(
        #    T_anomaly, climparams, prevClimateState["albedo"], dtime
        # )
        albedo = climateParams.diagnose_albedo_w_constraint(
            T_anomaly, prevClimateState["albedo"], dtime
        )
    else:
        # albedo = Diagnose_albedo(T_anomaly, climparams)
        albedo = climateParams.diagnose_albedo_w_constraint(T_anomaly)

    # Get a new temperature anomaly as impacted by albedo (if we want it)
    if albedo_feedback:
        # T_anomaly += Diagnose_Delta_T_from_albedo(albedo, climparams)
        T_anomaly += climateParams.diagnose_delta_temmp_from_albedo(albedo)

    # Stochasticity in the model (if we want it)
    if stochastic_C_atm:
        # c_atm = Diagnose_Stochastic_C_atm(c_atm, climparams)
        c_atm = climateParams.diagnose_stochastic_c_atm(c_atm)

    # Ordinary diagnostics
    pH = climateParams.diagnose_ocean_surface_ph(c_atm)
    # pH = Diagnose_OceanSurfacepH(c_atm, climparams)

    T_C = Diagnose_actual_temperature(T_anomaly)
    T_F = Diagnose_degreesF(T_C)

    # Create a new climate state with these updates
    ClimateState = makeacopy(prevClimateState)
    ClimateState["C_atm"] = c_atm
    ClimateState["C_ocean"] = c_ocean
    ClimateState["albedo"] = albedo
    ClimateState["T_anomaly"] = T_anomaly
    ClimateState["pH"] = pH
    ClimateState["T_C"] = T_C
    ClimateState["T_F"] = T_F
    ClimateState["F_ha"] = F_ha
    ClimateState["F_ao"] = F_ao
    ClimateState["F_oa"] = F_oa
    ClimateState["F_la"] = F_la
    ClimateState["F_al"] = F_al
    ClimateState["year"] += dtime

    # Return the new climate state
    return ClimateState


def is_same(arr1, arr2):
    """
    Throw an error if the two arrays are not the same

    @param arr1  First array
    @param arr2  Second array
    @thrwos ValueError
    """
    if len(arr1) != len(arr2) or not np.allclose(arr1, arr2):
        raise ValueError("Failed QC: arrays differ!")


def CreateClimateState(climate_params: dict):
    """
    Create a new climate state with default values (preindustrial)

    @param ClimateParams
    """

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

    # Return the climate
    return ClimateState


def sigmafloor(t_in, t_transition, t_interval, floor):
    """
    Generate a sigmoid (smooth step-down) function with a floor

    @param t_in  Starting temperature
    @param t_transition  Transition temperature
    @param t_interval  Interval for transition temperature
    @param floor
    """
    temp = 1 - 1 / (1 + np.exp(-(t_in - t_transition) * 3 / t_interval))
    return temp * (1 - floor) + floor


def sigmaup(t_in, transitiontime, transitiontimeinterval):
    """
    Generate a sigmoid (smooth step-up) function

    @param t_in
    @param transitiontime
    @param transitiontimeinterval
    """
    denom = 1 + np.exp(-(t_in - transitiontime) * 3 / transitiontimeinterval)
    return 1 / denom


def sigmadown(t_in, transitiontime, transitiontimeinterval):
    """
    Generate a sigmoid (smooth step-down) function

    @param t_in
    @param transitiontime
    @param transitiontimeinterval
    """
    return 1 - sigmaup(t_in, transitiontime, transitiontimeinterval)


def CollectClimateTimeSeries(climatestate_list, whatIwant):
    """
    Collect elements from a list of dictionaries

    @param ClimateState_list
    @param whatIwant
    """
    array = np.empty(0)
    for climstate in climatestate_list:
        array = np.append(array, climstate[whatIwant])
    return array


def Diagnose_actual_temperature(T_anomaly: float) -> float:
    """
    Compute degrees C from a temperature anomaly

    @param T_anomaly
    @returns temperature in Celsius
    """
    T_C = T_anomaly + 14
    return T_C


def Diagnose_degreesF(T_C: float) -> float:
    """
    Convert temperature from C to F

    @param T_C
    @returns  temperature in F
    """

    # Do the conversion to F
    T_F = T_C * 9 / 5 + 32  ### END SOLUTION

    # Return the diagnosed temperature in F
    return T_F


def post_peak_flattener(time, eps, transitiontimeinterval, epslongterm):
    """
    Flatten the post peak

    @param time, eps, transitiontimeinterval, epslongterm
    @returns neweps
    """
    ipeak = np.where(eps == np.max(eps))[0][0]
    print("peak", eps[ipeak], ipeak)
    b = eps[ipeak]
    a = epslongterm
    neweps = makeacopy(eps)
    for i in range(ipeak, len(eps)):
        # ipostpeak = i - ipeak
        neweps[i] = a + np.exp(
            -((time[i] - time[ipeak]) ** 2) / transitiontimeinterval**2
        ) * (b - a)
    return neweps


def make_emissions_scenario(
    t_start, t_stop, nsteps, k, eps_0, t_0, t_trans, delta_t_trans
):
    """
    Make the emissions scenario

    @param t_start, t_stop, nsteps, k, eps_0, t_0, t_trans, delta_t_trans
    @returns time
    @returns eps
    """
    time = np.linspace(t_start, t_stop, nsteps)
    myexp = np.exp(k * time)
    myN = eps_0 / (np.exp(k * t_0) * sigmadown(t_0, t_trans, delta_t_trans))
    mysigmadown = sigmadown(time, t_trans, delta_t_trans)
    eps = myN * myexp * mysigmadown
    return time, eps


def make_emissions_scenario_lte(
    t_start, t_stop, nsteps, k, eps_0, t_0, t_trans, delta_t, epslongterm
):
    """
    Make emissions scenario with long term emissions

    @param t_start, t_stop, nsteps, k, eps_0, t_0, t_trans, delta_t_trans, epslongterm
    @returns time
    @returns neweps
    """
    time, eps = make_emissions_scenario(
        t_start, t_stop, nsteps, k, eps_0, t_0, t_trans, delta_t
    )
    neweps = post_peak_flattener(time, eps, delta_t, epslongterm)
    return time, neweps
