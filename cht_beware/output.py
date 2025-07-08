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

        if file_name is None:
            file_name = os.path.join(self.model.path, "beware_his.nc")

        ds = xr.open_dataset(file_name)

        # Optional: Select profiles or based on coordinates
        if profile is not None:
            if isinstance(profile[0], int):
                ds = ds.isel(prof_id=profile)
            elif isinstance(profile[0], str):
                ds = ds.sel(prof_id=profile)

        # Select parameters
        if parameter is None:
            parameter = {"R2": [0,1,2,3,4,5]}

        records = []

        prof_xs = ds["prof_x"].values
        prof_ys = ds["prof_y"].values

        for i, prof_name in enumerate(ds.prof_id.values):
            prof_name = str(prof_name)
            for t_idx, t in enumerate(ds.time.values):
                row = {
                    "time": pd.to_datetime(t),
                    "prof_id": prof_name,
                    "prof_x": prof_xs[i],
                    "prof_y": prof_ys[i]
                }
                for var, est_indices in parameter.items():
                    vals = ds[var].isel(prof_id=i, time=t_idx, nR2estimates=est_indices).values
                    row[var] = vals.item() if len(est_indices) == 1 else vals.tolist()
                records.append(row)

        df = pd.DataFrame(records).set_index(["time", "prof_id", "prof_x", "prof_y"]).sort_index()

        return df