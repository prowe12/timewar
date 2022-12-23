#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 16:16:15 2022

@author: prowe
"""

# Start with an empty dictionary
climate_params: dict[str, float] = {}

# Preindustrial climate values
climate_params["preindust_c_atm"] = 615
climate_params["preindust_c_ocean"] = 350
climate_params["preindust_albedo"] = 0.3
climate_params["preindust_pH"] = 8.2

# Parameter for the basic sensitivity of the climate to increasing CO2
# IPCC: 3 degrees for doubled CO2
climate_params["climate_sensitivity"] = 3 / climate_params["preindust_c_atm"]

# Carbon flux constants
climate_params["k_la"] = 120
climate_params["k_al0"] = 113
climate_params["k_al1"] = 0.0114
climate_params["k_oa"] = 0.2
climate_params["k_ao"] = 0.114

# Parameter for the ocean degassing flux feedback
climate_params["DC"] = 0.034  # Pretty well known from physical chemistry

# Parameters for albedo feedback
climate_params["albedo_sensitivity"] = -100
# Based on our radiative balance sensitivity analysis
climate_params["albedo_transition_temperature"] = 4
# T at which significant albedo reduction kicks in (a guess)
climate_params["albedo_transition_interval"] = 1
# Temperature range over which albedo reduction kicks in (a guess)
climate_params["max_albedo_change_rate"] = 0.0006
# Amount albedo can change in a year (based on measurements)
climate_params["fractional_albedo_floor"] = 0.9
# Maximum of 10% reduction in albedo (a guess)

# Parameters for the atmosphere->land flux feedback
climate_params["F_al_transitionT"] = 4
# T anomaly at which photosynthesis will become impaired (a guess)
climate_params["F_al_transitionTinterval"] = 1
# Temperature range over which photosynthesis impairment kicks in (guess)
climate_params["fractional_F_al_floor"] = 0.9
# Maximum of 10% reduction in F_al (a guess)

# Parameter for stochastic processes
climate_params["Stochastic_c_atm_std_dev"] = 0.1
# Set to zero for no randomness in C_atm

# This displays the dictionary contents
# display(climate_params)
print(climate_params)
