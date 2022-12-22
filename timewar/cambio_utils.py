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


# # # #   FUNCTIONS THAT USE CLIMATEPARAMS   # # # #


def CreateClimateState(ClimateParams):
    """
    Create a new climate state with default values (preindustrial)

    @param ClimateParams
    """

    # Create an empty climate state
    ClimateState = {}

    # Fill in some default (preindustrial) values
    ClimateState["C_atm"] = ClimateParams["preindust_c_atm"]
    ClimateState["C_ocean"] = ClimateParams["preindust_c_ocean"]
    ClimateState["albedo"] = ClimateParams["preindust_albedo"]
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


def Diagnose_OceanSurfacepH(C_atm, ClimateParams):
    """
    Compute ocean pH as a function of atmospheric CO2

    @param C_atm
    @param ClimateParams
    @returns pH
    """
    # Specify a default output value (will be over-ridden by your algorithm)
    pH = 0

    # Extract needed climate parameters
    preindust_pH = ClimateParams["preindust_pH"]
    preindust_C_atm = ClimateParams["preindust_c_atm"]

    # Calculate the new pH according to our algorithm
    pH = -np.log10(C_atm / preindust_C_atm) + preindust_pH

    # Return our diagnosed pH value
    return pH


def Diagnose_T_anomaly(C_atm, ClimateParams):
    """
    Compute a temperature anomaly from the atmospheric carbon amount
    @param C_atm
    @param ClimateParams
    @returns  temperature anomaly
    """
    CS = ClimateParams["climate_sensitivity"]
    preindust_C_atm = ClimateParams["preindust_c_atm"]
    T_anomaly = CS * (C_atm - preindust_C_atm)
    return T_anomaly


def Diagnose_F_ao(C_atm, ClimateParams):
    """
    Compute flux of carbon from atm to ocean

    @param C_atm
    @param ClimateParams

    @returns Flux from atmosphere to ocean
    """

    # Calculate the F_ao based on k_ao and the amount of carbon in the atmosphere
    k_ao = ClimateParams["k_ao"]
    F_ao = k_ao * C_atm

    # Return the diagnosed flux
    return F_ao


def Diagnose_F_oa(C_ocean, T_anomaly, ClimateParams):
    """
    Compute a temperature-dependent degassing flux of carbon from the ocean

    @param C_ocean
    @param T_anomaly
    @param ClimateParams
    @returns flux from ocean to atmosphere
    """

    DC = ClimateParams["DC"]
    k_oa = ClimateParams["k_oa"]
    F_oa = k_oa * (1 + DC * T_anomaly) * C_ocean

    # Return the diagnosed flux
    return F_oa


def Diagnose_F_al(T_anomaly, C_atm, ClimateParams):
    """
    Compute the terrestrial carbon sink

    @param T_anomaly
    @param C_atm
    @param ClimateParams
    @returns flux from atmosphere to land
    """

    # Extract parameters we need from ClimateParameters, and calculate a new flux
    k_al0 = ClimateParams["k_al0"]
    k_al1 = ClimateParams["k_al1"]
    F_al_transitionT = ClimateParams["F_al_transitionT"]
    F_al_transitionTinterval = ClimateParams["F_al_transitionTinterval"]
    floor = ClimateParams["fractional_F_al_floor"]
    F_al = (
        k_al0
        + k_al1
        * sigmafloor(
            T_anomaly, F_al_transitionT, F_al_transitionTinterval, floor
        )
        * C_atm
    )

    # Return the diagnosed flux
    return F_al


def Diagnose_F_la(ClimateParams):
    """
    Compute the terrestrial carbon source

    @param ClimateParams
    @returns flux from land to atmosphere
    """

    k_la = ClimateParams["k_la"]
    F_la = k_la
    return F_la


def Diagnose_albedo_with_constraint(
    T_anomaly, ClimateParams, previousalbedo=0, dtime=0
):
    """
    Return the albedo as a function of temperature, constrained so the
    change can't exceed a certain amount per year, if so flagged

    @param T_anomaly
    @param ClimateParams
    @param previousalbedo=0
    @param dtime=0
    @returns albedo
    """

    # Find the albedo without constraint
    albedo = Diagnose_albedo(T_anomaly, ClimateParams)

    # Applying a constraint, if called for
    if (previousalbedo != 0) & (dtime != 0):
        albedo_change = albedo - previousalbedo
        max_albedo_change = ClimateParams["max_albedo_change_rate"] * dtime
        if np.abs(albedo_change) > max_albedo_change:
            this_albedo_change = np.sign(albedo_change) * max_albedo_change
            albedo = previousalbedo + this_albedo_change

    # Return the albedo
    return albedo


def Diagnose_albedo(T_anomaly, ClimateParams):
    """
    Return the albedo as a function of temperature anomaly

    @param T_anomaly
    @param ClimateParams
    """

    # Extract parameters we need from ClimateParameters, and calculate a new albedo
    transitionT = ClimateParams["albedo_transition_temperature"]
    transitionTinterval = ClimateParams["albedo_transition_interval"]
    floor = ClimateParams["fractional_albedo_floor"]
    preindust_albedo = ClimateParams["preindust_albedo"]
    albedo = (
        sigmafloor(T_anomaly, transitionT, transitionTinterval, floor)
        * preindust_albedo
    )

    # Return the diagnosed albedo
    return albedo


def Diagnose_Delta_T_from_albedo(albedo, ClimateParams):
    """
    Compute additional planetary temperature increase resulting
    from a lower albedo. Based on the idea of radiative balance, ASR = OLR

    @param albedo
    @param ClimateParams
    """

    # Extract parameters we need and make the diagnosis
    AS = ClimateParams["albedo_sensitivity"]
    preindust_albedo = ClimateParams["preindust_albedo"]
    Delta_T_from_albedo = (albedo - preindust_albedo) * AS
    return Delta_T_from_albedo


def Diagnose_Stochastic_C_atm(C_atm, ClimateParams):
    """
    Return a noisy version of T

    @param C_atm,
    @param ClimateParams
    """

    # Extract parameters we need and make the diagnosis
    Stochastic_C_atm_std_dev = ClimateParams["Stochastic_C_atm_std_dev"]
    C_atm_new = np.random.normal(C_atm, Stochastic_C_atm_std_dev)
    return C_atm_new
