#!/usr/bin/env python
# coding: utf-8
"""
Created on Wed Dec 21 15:49:24 2022

@author: nesh

By Steven Neshyba
Refactored by Penny Rowe and Daniel Neshyba-Rowe
"""
from copy import deepcopy as makeacopy
from typing import Any
import numpy as np
import numpy.typing as npt


def is_same(arr1: npt.NDArray[Any], arr2: npt.NDArray[Any]) -> bool:
    """
    Throw an error if the two arrays are not the same

    @param arr1  First array
    @param arr2  Second array
    @return  True if arrays are same, else false
    """
    return len(arr1) == len(arr2) and np.allclose(arr1, arr2)


def sigmafloor(
    t_in: float, t_transition: float, t_interval: float, floor: float
) -> float:
    """
    Generate a sigmoid (smooth step-down) function with a floor

    @param t_in  Starting temperature
    @param t_transition  Transition temperature
    @param t_interval  Interval for transition temperature
    @param floor
    """
    temp = 1 - 1 / (1 + np.exp(-(t_in - t_transition) * 3 / t_interval))
    return temp * (1 - floor) + floor


def sigmaup(
    t_in: float | npt.NDArray[Any], transitiontime: float, transitiontimeinterval: float
) -> npt.NDArray[Any]:
    """
    Generate a sigmoid (smooth step-up) function

    @param t_in
    @param transitiontime
    @param transitiontimeinterval
    """
    denom = 1 + np.exp(-(t_in - transitiontime) * 3 / transitiontimeinterval)
    return 1 / denom


def sigmadown(
    t_in: float | npt.NDArray[Any], transitiontime: float, transitiontimeinterval: float
) -> npt.NDArray[Any]:
    """
    Generate a sigmoid (smooth step-down) function

    @param t_in
    @param transitiontime
    @param transitiontimeinterval
    """
    return 1 - sigmaup(t_in, transitiontime, transitiontimeinterval)


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
    # TODO: check if this is unused, and perhaps delete

    # Do the conversion to F
    T_F = T_C * 9 / 5 + 32  ### END SOLUTION

    # Return the diagnosed temperature in F
    return T_F


def post_peak_flattener(
    time: npt.NDArray[Any],
    eps: npt.NDArray[Any],
    transitiontimeinterval: float,
    epslongterm: float,
) -> npt.NDArray[Any]:
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
    time: npt.NDArray[Any],
    inv_t_const: float,
    transitionyear: float,
    transitionduration: float,
) -> npt.NDArray[Any]:
    """
    Make the emissions scenario

    @param time  Time, in years
    @param inv_t_const  Inverse time constant
    @param transitionyear  Transition time (years)
    @param transitionduration  Transition time interval
    @returns eps
    """
    t_0 = 2020.0  # year for normalizing co2 emission
    eps_0 = 11.3  # co2 emission normalization value

    origsigmadown = sigmadown(t_0, transitionyear, transitionduration)
    mysigmadown = sigmadown(time, transitionyear, transitionduration)

    myexp = np.exp(time * inv_t_const)
    origexp = np.exp(t_0 * inv_t_const) * origsigmadown

    return eps_0 * myexp / origexp * mysigmadown


def make_emissions_scenario2(
    time: npt.NDArray[Any], k: float, t_peak: float, delta_t_trans: float
) -> npt.NDArray[Any]:
    """
    Returns an emissions scenario parameterized by the year of peak emissions
    @param time
    @param k
    @param t_peak  Year of peak carbon
    @param delta_t_trans  Transition time interval
    """
    term1 = (
        np.exp(t_peak / delta_t_trans) ** 3
        * (k * delta_t_trans - 3)
        / (-k * delta_t_trans)
    )
    term2 = np.log(term1)
    t_trans = term2 / 3 * delta_t_trans
    return make_emissions_scenario(time, k, t_trans, delta_t_trans)


def make_emissions_scenario_lte(
    t_start: float,
    t_stop: float,
    dtime: float,
    k: float,
    t_peak: float,
    delta_t: float,
    epslongterm: float,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """
    Make emissions scenario with long term emissions

    @param t_start, t_stop, dtime
    @param k
    @param t_peak  Year of peak carbon
    @param delta_t_trans  Transition time interval
    @param epslongterm  Long term CO2 emissions
    @returns time
    @returns neweps  Anthropogenic CO2 emissions, with time
    """
    time = np.arange(t_start, t_stop, dtime)
    eps = make_emissions_scenario2(time, k, t_peak, delta_t)
    neweps = post_peak_flattener(time, eps, delta_t, epslongterm)
    return time, neweps
