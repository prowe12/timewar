#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 15:49:24 2022

@author: prowe
"""

import numpy as np

from cambio_utils import sigmafloor


class ClimateParams:
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
    ocean_degas_flux_feedback = 0.034  # Pretty well known from physical chemistry

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
    flux_al_transition_temp = 4
    # Temperature range over which photosynthesis impairment kicks in (guess)
    flux_al_transition_temp_interval = 1
    # Maximum of 10% reduction in F_al (a guess)
    fractional_flux_al_floor = 0.9

    def __init__(self, stochastic_c_atm_std_dev: float = 0.1) -> None:
        """
        Create an instance of the class

        @param  stochastic_c_atm_std_dev  Std dev of atm. carbon

        """
        # Parameter for stochastic processes (0 for no randomness in c_atm)
        self.stochastic_c_atm_std_dev = stochastic_c_atm_std_dev

    def diagnose_ocean_surface_ph(self, c_atm: float) -> float:
        """
        Compute ocean pH as a function of atmospheric CO2

        @param c_atm
        @param ClimateParams
        @returns pH
        """
        # Calculate the new pH according to our algorithm
        ph = (
            -np.log10(c_atm / ClimateParams.preindust_c_atm)
            + ClimateParams.preindust_ph
        )

        # Return our diagnosed pH value
        return ph

    def diagnose_temp_anomaly(self, c_atm: float) -> float:
        """
        Compute a temperature anomaly from the atmospheric carbon amount
        @param c_atm
        @returns  temperature anomaly
        """
        clim_sens = ClimateParams.climate_sensitivity
        return clim_sens * (c_atm - ClimateParams.preindust_c_atm)

    def diagnose_flux_atm_ocean(self, c_atm: float):
        """
        Compute flux of carbon from atm to ocean

        @param c_atm
        @returns Flux of carbon from the atmosphere to the ocean
        """

        # Calculate the F_ao based on k_ao and the amount of carbon in the atmosphere
        k_ao = ClimateParams.k_ao
        flux_atm_ocean = k_ao * c_atm

        # Return the diagnosed flux
        return flux_atm_ocean

    def diagnose_flux_ocean_atm(self, c_ocean: float, temp_anomaly: float) -> float:
        """
        Compute a temperature-dependent degassing flux of carbon from the ocean

        @param c_ocean
        @param temp_anomaly
        @returns flux from ocean to atmosphere
        """
        ocean_degas_ff = ClimateParams.ocean_degas_flux_feedback
        k_oa = ClimateParams.k_oa
        return k_oa * (1 + ocean_degas_ff * temp_anomaly) * c_ocean

    def diagnose_flux_atm_land(self, temp_anomaly: float, c_atm: float) -> float:
        """
        Compute the terrestrial carbon sink

        @param temp_anomaly
        @param c_atm
        @returns flux from atmosphere to land
        """
        k_al0 = ClimateParams.k_al0
        k_al1 = ClimateParams.k_al1

        sigma_floor_val = sigmafloor(
            temp_anomaly,
            ClimateParams.flux_al_transition_temp,
            ClimateParams.flux_al_transition_temp_interval,
            ClimateParams.fractional_flux_al_floor,
        )
        return k_al0 + k_al1 * sigma_floor_val * c_atm

    def diagnose_flux_land_atm(self):
        """
        Compute the terrestrial carbon source

        @param ClimateParams
        @returns flux from land to atmosphere
        """
        return ClimateParams.k_la

    def diagnose_albedo_w_constraint(
        self, temp_anom: float, prev_albedo: float = 0, dtime: float = 0
    ) -> float:
        """
        Return the albedo as a function of temperature, constrained so the
        change can't exceed a certain amount per year, if so flagged

        @param temp_anomaly
        @param previousalbedo=0
        @param dtime=0
        @returns albedo
        """
        # Find the albedo without constraint
        albedo = self.diagnose_albedo(temp_anom)

        # Applying a constraint, if called for
        if (prev_albedo != 0) & (dtime != 0):
            albedo_change = albedo - prev_albedo
            max_albedo_change = ClimateParams.max_albedo_change_rate * dtime
            if np.abs(albedo_change) > max_albedo_change:
                this_albedo_change = np.sign(albedo_change) * max_albedo_change
                albedo = prev_albedo + this_albedo_change
        return albedo

    def diagnose_albedo(self, temp_anom: float) -> float:
        """
        Return the albedo as a function of temperature anomaly

        @param temp_anomaly
        @returns albedo
        """
        temp = ClimateParams.albedo_transition_temperature
        interval = ClimateParams.albedo_transition_interval
        floor = ClimateParams.fractional_albedo_floor
        preind_albedo = ClimateParams.preindust_albedo
        albedo = sigmafloor(temp_anom, temp, interval, floor) * preind_albedo
        return albedo

    def diagnose_delta_t_from_albedo(self, albedo: float) -> float:
        """
        Compute additional planetary temperature increase resulting
        from a lower albedo. Based on the idea of radiative balance, ASR = OLR

        @param albedo
        @returns  Planetary temperature increase from new albedo
        """
        alb_sens = ClimateParams.albedo_sensitivity
        preindust_albedo = ClimateParams.preindust_albedo
        return (albedo - preindust_albedo) * alb_sens

    def diagnose_stochastic_c_atm(self, c_atm: float):
        """
        Return a noisy version of the atmospheric carbon

        @param c_atm  Atmospheric carbon
        @returns  Atmospheric carbon amount randomized based on std dev
        """
        c_atm_new = np.random.normal(c_atm, self.stochastic_c_atm_std_dev)
        return c_atm_new


# def CreateClimateState(ClimateParams):
#     """
#     Create a new climate state with default values (preindustrial)
#     """

#     # Create an empty climate state
#     ClimateState = {}

#     # Fill in some default (preindustrial) values
#     ClimateState["C_atm"] = ClimateParams.preindust_C_atm"]
#     ClimateState["C_ocean"] = ClimateParams.preindust_C_ocean"]
#     ClimateState["albedo"] = ClimateParams.preindust_albedo"]
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
