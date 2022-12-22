#!/usr/bin/env python
# coding: utf-8
"""
Created on Wed Dec 21 15:49:24 2022

@author: nesh

By Steven Neshyba
With modifications by Penny Rowe and Daniel Neshyba-Rowe
"""
from copy import deepcopy as makeacopy
import numpy as np


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


def Diagnose_actual_temperature(T_anomaly):
    """
    Compute degrees C from a temperature anomaly

    @param T_anomaly
    @returns temperature in Celsius
    """
    T_C = T_anomaly + 14
    return T_C


def Diagnose_degreesF(T_C):
    """
    Convert temperature from C to F

    @param T_C
    @returns  temperature in F
    """

    # Do the conversion to F
    T_F = T_C * 9 / 5 + 32  ### END SOLUTION

    # Return the diagnosed temperature in F
    return T_F


def PostPeakFlattener(time, eps, transitiontimeinterval, epslongterm):
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
    Make emissions scenario LTE

    @param t_start, t_stop, nsteps, k, eps_0, t_0, t_trans, delta_t_trans, epslongterm
    @returns time
    @returns neweps
    """
    time, eps = make_emissions_scenario(
        t_start, t_stop, nsteps, k, eps_0, t_0, t_trans, delta_t
    )
    neweps = PostPeakFlattener(time, eps, delta_t, epslongterm)
    return time, neweps
