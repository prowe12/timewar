#!/usr/bin/env python
# coding: utf-8
"""
Created on Wed Dec 21 15:49:24 2022

@author: nesh

By Steven Neshyba
Refactored by Penny Rowe and Daniel Neshyba-Rowe
"""
from copy import deepcopy as makeacopy
import numpy as np


def is_same(arr1: np.ndarray, arr2: np.ndarray) -> bool:
    """
    Throw an error if the two arrays are not the same

    @param arr1  First array
    @param arr2  Second array
    @return  True if arrays are same, else false
    """
    return len(arr1) == len(arr2) and np.allclose(arr1, arr2)


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
    time: np.ndarray,
    inv_t_const: float,
    eps_0: float,
    t_0,
    transitionyear,
    transitionduration,
):
    """
    Make the emissions scenario

    @param time  Time, in years
    @param inv_t_const  Inverse time constant
    @param eps_0,
    @param t_0,
    @param transitionyear  Transition time (years)
    @param transitionduration  Transition time interval
    @returns eps
    """
    origsigmadown = sigmadown(t_0, transitionyear, transitionduration)
    mysigmadown = sigmadown(time, transitionyear, transitionduration)

    myexp = np.exp(time * inv_t_const)
    origexp = np.exp(t_0 * inv_t_const) * origsigmadown

    return eps_0 * myexp / origexp * mysigmadown


def make_emissions_scenario_lte(
    t_start: float,
    t_stop: float,
    dtime: float,
    k: float,
    eps_0: float,
    t_0: float,
    t_trans: float,
    delta_t: float,
    epslongterm: float,
):
    """
    Make emissions scenario with long term emissions

    @param t_start, t_stop, dtime
    @param k
    @param eps_0,
    @param t_0
    @param t_trans  Transition time (years)
    @param delta_t_trans  Transition time interval
    epslongterm
    @returns time
    @returns neweps
    """
    time = np.arange(t_start, t_stop, dtime)
    eps = make_emissions_scenario(time, k, eps_0, t_0, t_trans, delta_t)
    neweps = post_peak_flattener(time, eps, delta_t, epslongterm)
    return time, neweps
