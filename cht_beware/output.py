# -*- coding: utf-8 -*-
"""
Created on 2025-06-06 09:26

@author: roelvink (f.e.)
"""
import numpy as np
import pandas as pd
import xarray as xr
import time
class BewareOutput:

    def __init__(self, model):
        """
        The BEWARE Output class contains methods to read the BEWARE output.
        Parameters
        ----------
        model: BEWARE model object
            The BEWARE model instance to which the output belongs.
        """
        self.model = model

    def read_his_file(self,
                            file_name=None,
                            profile=None,
                            parameter=None):
        """Reads a BEWARE history file and returns a DataFrame with timeseries
        
        Parameters
        ----------
        file_name : str, optional
            Path to the BEWARE history file. If None, defaults to "beware_his.nc" in the model path.
        profile : list, optional
            List of profile indices or names to select from the dataset. If None, all profiles are selected.
        parameter : dict, optional
            Dictionary of parameters to select from the dataset. Keys are variable names and values are indices for R2 estimates.
        """
        self.model.logger.info(f"\n--------------- Output --------------\n")

        if file_name is None:
            file_name = self.model.path / "beware_his.nc"

        start_output = time.perf_counter()
        ds = xr.open_dataset(file_name)
        end_output = time.perf_counter()
        self.model.logger.info(f"Time to read output: {end_output - start_output:.2f} seconds")

        # Optional: Select profiles or based on coordinates
        if profile is not None:
            if isinstance(profile[0], int):
                ds = ds.isel(prof_id=profile)
            elif isinstance(profile[0], str):
                ds = ds.sel(prof_id=profile)
        
        # Select parameters
        if parameter is None:
            parameter = {"R2": [0], "Hs": [0], "Tp": [0], "WL": [0]}

        sel_vars = []
        for var, est_indices in parameter.items():
            if "nR2estimates" in ds[var].dims:
                sel_vars.append(ds[var].isel(nR2estimates=est_indices))
            else:
                sel_vars.append(ds[var])

        sel = xr.merge(sel_vars)
        df = sel.to_dataframe().reset_index()
        df = df.set_index(["time", "prof_id", "prof_x", "prof_y", "wave_x", "wave_y", "flow_x", "flow_y"])
        df.drop(columns=["nR2estimates"], inplace=True, errors='ignore')

        return df