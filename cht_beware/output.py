# -*- coding: utf-8 -*-
"""
Created on 2025-06-06 09:26

@author: roelvink (f.e.)
"""
import os
import numpy as np
import pandas as pd
import xarray as xr

class BewareOutput:

    def __init__(self, model):
        self.model = model

    def read_his_file(self,
                            file_name=None,
                            parameters=None,
                            profiles=None,
                            prcs=None,
                            nodata=0.0):
        """
        Reads BEWARE history NetCDF file and returns a dictionary of DataFrames with requested parameters.

        Parameters:
        - file_name: Optional path to the NetCDF file
        - parameters: List of parameters to read (e.g., ['R2', 'Hs', 'Tp', 'WL'])
        - transects: Optional list of transect indices to extract
        - prcs: List of percentiles to extract (e.g., [2, 98])
        - nodata: Value to replace NaNs

        Returns:
        - data: dict of pandas DataFrames (e.g., data["R2"], data["WL"], etc.)
        """

        if file_name is None:
            file_name = os.path.join(self.model.path, "beware_his.nc")


        ds = xr.open_dataset(file_name)

        # Get list of all profile IDs from dataset (adjust key name accordingly)
        if "Profiles" in ds:
            all_profiles = [p.decode() if isinstance(p, bytes) else p for p in ds["Profiles"].values]
        else:
            all_profiles = None

        # Map profile IDs to indices
        if profiles is not None and all_profiles is not None:
            prof_sel = [all_profiles.index(t) for t in profiles if t in all_profiles]
        else:
            prof_sel = None  # read all

        # Default parameters
        if parameters is None:
            parameters = ["R2", "R2_setup", "Hs", "Tp", "WL"]

        data = {}

        def select_profiles(var):
            values = var.values
            if prof_sel is not None:
                values = values[prof_sel, ...]  # assume transects on first dim
            return np.nan_to_num(values, nan=nodata)

        for param in parameters:
            if param in ds:
                values = select_profiles(ds[param])
                data[param] = pd.DataFrame(values)

        # Swash calculation
        if all(p in data for p in ["R2", "R2_setup", "WL"]):
            data["swash"] = data["R2"] - data["R2_setup"] - data["WL"]

        # Coordinates (assumed same indexing)
        for coord in ["x_coast", "y_coast", "x_off", "y_off"]:
            if coord in ds:
                values = select_profiles(ds[coord])
                data[coord] = pd.Series(values)

        # Percentiles
        if prcs:
            for prc in prcs:
                key_r2 = f"R2_{int(round(prc))}"
                key_setup = f"R2_setup_{int(round(prc))}"
                if key_r2 in ds:
                    data[key_r2] = pd.DataFrame(np.nan_to_num(ds[key_r2].values, nan=nodata))
                if key_setup in ds:
                    data[key_setup] = pd.DataFrame(np.nan_to_num(ds[key_setup].values, nan=nodata))

        # Set start time if not already defined
        if not getattr(self.input, "tstart", None):
            if "time" in ds:
                t0 = float(ds["time"].values[0])
                self.input.tstart = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=t0)

        ds.close()
        return data

