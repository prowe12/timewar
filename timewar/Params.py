#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 15:49:24 2022

@author: prowe
"""

import numpy as np

from cambio_utils import sigmafloor


class Params:
    """Climate Parameters Class"""

    # Constants

    # Preindustrial climate values
    preindust_c_atm = 615
    preindust_c_ocean = 350
    preindust_albedo = 0.3
    preindust_ph = 8.2

    # Parameter for the basic sensitivity of the climate to increasing CO2
    # IPCC: 3 degrees for doubled CO2
    climate_sensitivity = 3 / preindust_c_atm

    # Carbon flux constants
    k_la = 120
    k_al0 = 113
    k_al1 = 0.0114
    k_oa = 0.2
    k_ao = 0.114

    # Parameter for the ocean degassing flux feedback
    DC = 0.034  # Pretty well known from physical chemistry

    # Parameters for albedo feedback
    # Based on our radiative balance sensitivity analysis
    albedo_sensitivity = -100
    # T at which significant albedo reduction kicks in (a guess)
    albedo_transition_temperature = 4
    # Temperature range over which albedo reduction kicks in (a guess)
    albedo_transition_interval = 1
    # Amount albedo can change in a year (based on measurements)
    max_albedo_change_rate = 0.0006
    # Maximum of 10% reduction in albedo (a guess)
    fractional_albedo_floor = 0.9

    # Parameters for the atmosphere->land flux feedback
    # T anomaly at which photosynthesis will become impaired (a guess)
    F_al_transitionT = 4
    # Temperature range over which photosynthesis impairment kicks in (guess)
    F_al_transitionTinterval = 1
    # Maximum of 10% reduction in F_al (a guess)
    fractional_F_al_floor = 0.9

    def __init__(self, stochastic_c_atm_std_dev=0.1):
        """
        Create an instance of the class

        @param  stochastic_c_atm_std_dev  Std dev of atm. carbon

        """
        # Parameter for stochastic processes (0 for no randomness in c_atm)
        self.stochastic_c_atm_std_dev = stochastic_c_atm_std_dev

    def diagnose_ocean_surface_ph(self, c_atm):
        """
        Compute ocean pH as a function of atmospheric CO2

        @param C_atm
        @param ClimateParams
        @returns pH
        """
        # Specify a default output value (will be over-ridden by your algorithm)
        ph = 0

        # Calculate the new pH according to our algorithm
        ph = -np.log10(c_atm / Params.preindust_c_atm) + Params.preindust_ph

        # Return our diagnosed pH value
        return ph


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


# def CreateClimateState(ClimateParams):
#     """
#     Create a new climate state with default values (preindustrial)
#     """

#     # Create an empty climate state
#     ClimateState = {}

#     # Fill in some default (preindustrial) values
#     ClimateState["C_atm"] = ClimateParams["preindust_C_atm"]
#     ClimateState["C_ocean"] = ClimateParams["preindust_C_ocean"]
#     ClimateState["albedo"] = ClimateParams["preindust_albedo"]
#     ClimateState["T_anomaly"] = 0

#     # These are just placeholders (values don't mean anything)
#     ClimateState["pH"] = 0
#     ClimateState["T_C"] = 0
#     ClimateState["T_F"] = 0
#     ClimateState["F_ha"] = 0
#     ClimateState["F_ao"] = 0
#     ClimateState["F_oa"] = 0
#     ClimateState["F_al"] = 0
#     ClimateState["F_la"] = 0

#     # Return the climate
#     return ClimateState
