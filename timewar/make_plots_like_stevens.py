#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 15:49:31 2022

@author: prowe
"""

import matplotlib.pyplot as plt
import numpy as np


def make_plots_like_stevens(climate, fractional_albedo_floor):
    # # # #   Visualize the results of the run   # # # #
    # Plotting parameters:
    lwidth = 2

    # Unit conversions
    # Convert carbon amounts from GtC
    c_conversion_fac = {"GtC": 1, "GtCO2": 1 / 0.27, "ppm": 1 / 2.12}
    # TODO: Add conversion for F
    temp_conversion_fac = {"K": 1, "C": 273.15, "F": np.nan}
    # TODO: Do not use for anomaly, since K and C will be the same.
    #       need fancy code to convert to F

    time = climate["year"]

    # Plot Anthropogenic emissions in GtC/year
    c_units = "GtC"  # GtC, GtCO2, atm
    flux_type = "/year"  # total, per year
    yvals = [climate["F_ha"]]
    lables = ["Anthropogenic Emissions"]
    plt.figure()
    for i, yval in enumerate(yvals):
        plt.plot(time, yval * c_conversion_fac[c_units], label=lables[i])
    plt.legend()
    plt.grid(True)
    plt.xlabel("year")
    plt.ylabel(c_units + flux_type)

    # Plot the concentration of carbon in the atmosphere and oceans,
    # in GtC (one graph)
    c_units = "GtC"
    plot_me = ["C_atm", "C_ocean"]
    labels = ["C_atm", "C_ocean"]
    flux_type = ""
    plt.figure()
    for i, varname in enumerate(plot_me):
        label = f"{labels[i]} ({c_units})"
        yval = climate[varname] * c_conversion_fac[c_units]
        plt.plot(time, yval, label=label, linewidth=lwidth)
    plt.grid(True)
    plt.xlabel("time (years)")
    plt.ylabel(c_units + flux_type)
    plt.legend()

    # Re-plot the carbon in the atmosphere, converted to ppm (by dividing
    # C_atm_array by 2.12)
    c_units = "ppm"
    plot_me = ["C_atm"]
    labels = ["C_atm"]
    flux_type = ""
    plt.figure()
    for i, varname in enumerate(plot_me):
        yval = climate[varname] * c_conversion_fac[c_units]
        label = f"{labels[i]} ({c_units})"
        plt.plot(time, yval, label=label, linewidth=lwidth)
    plt.grid(True)
    plt.xlabel("time (years)")
    plt.ylabel(c_units + flux_type)
    plt.legend()

    # Extract and plot the albedo
    plot_me = ["albedo"]
    labels = ["Albedo"]
    plt.figure()
    for i, varname in enumerate(plot_me):
        yval = climate[varname]
        plt.plot(time, yval, label=f"{labels[i]}", linewidth=lwidth)
    plt.grid(True)
    plt.xlabel("time (years)")
    plt.ylabel("albedo")
    plt.legend()
    # TODO: Find a better solution for this
    ybottom = yval[0] * fractional_albedo_floor - 0.01
    ytop = yval[0] + 0.001
    plt.ylim([ybottom, ytop])

    # plot the ocean pH, specifying vertical axis limits of 7.8 to 8.3
    varname = "pH"
    yval = climate[varname]
    plt.figure()
    plt.plot(time, yval, label="pH", linewidth=lwidth, color="gray")
    plt.grid(True)
    plt.xlabel("time (years)")
    plt.ylabel(varname)
    plt.legend()
    # TODO: Find a better way to set the ylims
    ybottom = 7.8
    ytop = 8.3
    plt.ylim([ybottom, ytop])

    # TODO: Fix the way colors are handled

    # Extract and plot the temperature anomaly
    varname = "T_anomaly"
    yval = climate[varname]
    label = "Temperature anomaly"
    plt.figure()
    plt.plot(time, yval, label=label, linewidth=lwidth, color="red")
    plt.grid(True)
    plt.xlabel("time (years)")
    plt.ylabel("degrees K")
    plt.legend()

    # Extract the fluxes, compute net fluxes, and plot them
    plot_flux_diffs = True  # True, False
    plot_me = ["F_ha", "F_oa", "F_la"]
    labels = ["F_ha", "F_la-F_al", "F_oa-F_ao"]
    colors = ["black", "brown", "blue"]
    if plot_flux_diffs:
        ylabel = "Flux differences (GtC/year)"
    else:
        raise ValueError("option not here yet")
    plt.figure()
    for i, varname in enumerate(plot_me):
        if plot_flux_diffs and varname == "F_oa":
            yval = -climate["F_ao"] + climate["F_oa"]
        elif plot_flux_diffs and varname == "F_la":
            yval = -climate["F_al"] + climate["F_la"]
        else:
            yval = climate[varname]
        plt.plot(
            time, yval, label=labels[i], color=colors[i], linewidth=lwidth
        )
    plt.grid(True)
    plt.xlabel("time (years)")
    plt.ylabel(ylabel)
    plt.legend()
