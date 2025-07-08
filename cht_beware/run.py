# -*- coding: utf-8 -*-
"""
Created on 2025-06-06 16:02

@author: roelvink (f.e.)
"""
import xarray as xr
import numpy as np
from pathlib import Path
import os
import geopandas as gpd
import datetime

class BewareRun:

    def __init__(self, model):
        self.model = model

    def execute(self):

        if not self.model.input.variables.xbdatabase:
            raise ValueError("xb database file not set in input variables. Please set 'xbdatabase' in beware.inp")
        
        # Input transects with Prob of Matching to CR2
        profs = self.model.transects.gdf
                        
        # XBeach database
        file_name = Path(self.model.input.variables.xbdatabase)
        if not file_name.is_absolute():
            file_name = Path(self.model.path) / file_name

        if not os.path.exists(file_name):
            print(f"Warning! File {file_name} does not exist!")
            return
        
        ds_xb = xr.open_dataset(file_name)
        xbcoords = {}
        for var in ['Hs', 'Tp', 'WL', 'CfMod', 'BsMod']:
            xbcoords[var] = ds_xb[var].values  # 1D arrays over nConditions

        # Initialize outputs - align dimensions
        n_profiles = len(profs)
        n_forcings = len(self.model.boundary_conditions.gdf_wave.iloc[0]['timeseries']['hs'].values)
        nc_results = {"R2": np.full((n_profiles, n_forcings, 6), np.nan),
                      "R2_CfLow": np.full((n_profiles, n_forcings, 6), np.nan),
                      "R2_CfHigh": np.full((n_profiles, n_forcings, 6), np.nan),
                      "R2_BsLow": np.full((n_profiles, n_forcings, 6), np.nan),
                      "R2_BsHigh": np.full((n_profiles, n_forcings, 6), np.nan),
                      'Hs'  : np.full((n_profiles, n_forcings), np.nan),
                      'Tp'  : np.full((n_profiles, n_forcings), np.nan),
                      'WL'  : np.full((n_profiles, n_forcings), np.nan),
                    }
        gdf_results = []
        
        # Loop through the input profiles
        for iprof, prof in profs.iterrows():
            prof_name = prof["name"]

            print(f"Processing profile {iprof+1}/{n_profiles}")

            # Get the RRP ids and matching probabilities
            prob2BW2 = prof['ProbtoCR2']
            RRPids = np.where(prob2BW2 > 0.01)[0]
            prob = prob2BW2[RRPids]
            prob /= prob.sum()

            # Get the forcing conditions for this profile
            Hs = self.model.boundary_conditions.gdf_wave.iloc[iprof]['timeseries']['hs'].values
            Tp = self.model.boundary_conditions.gdf_wave.iloc[iprof]['timeseries']['tp'].values
            WL = self.model.boundary_conditions.gdf_flow.iloc[iprof]['timeseries']['wl'].values
            time = self.model.boundary_conditions.gdf_wave.iloc[iprof]['timeseries'].index

            # Store forcing conditions in the output arrays
            nc_results['Hs'][iprof, :] = Hs
            nc_results['Tp'][iprof, :] = Tp
            nc_results['WL'][iprof, :] = WL

            # Initialize gdf results for this profile
            gdf_results_iprof = {"R2_all": [],
                        "R2_CfLow_all": [],
                        "R2_CfHigh_all": [],
                        "R2_BsLow_all": [],
                        "R2_BsHigh_all": [],
                        "P_all": [],
                        }
        
            if len(prob)>=1:

                # Loop through the forcing conditions
                for it in range(len(time)):

                    # Initialize arrays per timestep
                    R2_it = np.full((len(RRPids), 5, 8), np.nan) # Nr of RRPs x 5 R2 estimates x 8 neighboring conditions
                    R2_CfLow_it = np.full((len(RRPids), 5, 8), np.nan)
                    R2_CfHigh_it = np.full((len(RRPids), 5, 8), np.nan)
                    R2_BsLow_it = np.full((len(RRPids), 5, 8), np.nan)
                    R2_BsHigh_it = np.full((len(RRPids), 5, 8), np.nan)
                    P_it = np.full((len(RRPids), 5, 8), 0.0)

                    # First find lower and upper bounds for input forcing condition
                    lower_bound = [xbcoords[k][xbcoords[k] <= val].max() for k, val in zip(['Hs', 'Tp', 'WL'], [Hs[it], Tp[it], WL[it]])]
                    upper_bound = [xbcoords[k][xbcoords[k] > val].min() for k, val in zip(['Hs', 'Tp', 'WL'], [Hs[it], Tp[it], WL[it]])]

                    # Create a mask for the xbcoords that match the forcing conditions
                    xb_mask = (
                            (xbcoords['Hs'] >= lower_bound[0]) & (xbcoords['Hs'] <= upper_bound[0]) &
                            (xbcoords['Tp'] >= lower_bound[1]) & (xbcoords['Tp'] <= upper_bound[1]) &
                            (xbcoords['WL'] >= lower_bound[2]) & (xbcoords['WL'] <= upper_bound[2])
                        )
                    id_forcing = np.where(xb_mask)[0]

                    # Get associated NGMiD and probabilities for the forcing interpolation
                    xb_vals = np.vstack([xbcoords[k][id_forcing] for k in  ['Hs', 'Tp', 'WL']])
                    forc_vals = np.array([Hs[it], Tp[it], WL[it]])

                    bounds = np.vstack([lower_bound, upper_bound])
                    denom = bounds[1] - bounds[0]
                    dist = 1 - np.abs(forc_vals[:, None] - xb_vals) / denom[:, None]
                    NGMiD = np.prod(dist, axis=0) ** (1 / 3)
                    P_forcing = NGMiD / NGMiD.sum()

                    # Now loop through the RRPids and extract R2 for each profile
                    for irrp, rrp in enumerate(RRPids):

                        R2data  = ds_xb['R2data'].values[:, id_forcing, rrp]  # R2 values for this profile and xb conditions
                        R2CfMod = ds_xb['R2CfMod'].values[:, id_forcing, rrp] 
                        R2BsMod = ds_xb['R2BsMod'].values[:, id_forcing, rrp]
                        P = P_forcing * prob[irrp] # Probability of this RRP given the forcing conditions

                        # Each condition has multiple R2 estimates (generally 5). Adjust the probabilities for this
                        nReturns = np.count_nonzero(~np.isnan(R2data), axis = 0)
                        tmp =  np.tile(P / nReturns, (5,1))
                        tmp[np.cumsum(tmp, axis=0) > 1]  = 0  # Ensure probabilities sum to 1
                        P_it[irrp, :] = tmp

                        # Now append the R2 data to the R2_it array
                        R2_it[irrp, :] = R2data
                        R2_CfLow_it[irrp, :] = R2data * R2CfMod[0,:]
                        R2_CfHigh_it[irrp, :] = R2data * R2CfMod[1,:]
                        R2_BsLow_it[irrp, :] = R2data * R2BsMod[0,:]
                        R2_BsHigh_it[irrp, :] = R2data * R2BsMod[1,:]
                    
                    # Now remove values where P_it is zero
                    valid_mask = P_it.flatten() > 0
                    R2_flat     = R2_it.flatten()[valid_mask]
                    R2_CfLow_flat = R2_CfLow_it.flatten()[valid_mask]
                    R2_CfHigh_flat = R2_CfHigh_it.flatten()[valid_mask]
                    R2_BsLow_flat = R2_BsLow_it.flatten()[valid_mask]
                    R2_BsHigh_flat = R2_BsHigh_it.flatten()[valid_mask]
                    P_flat      = P_it.flatten()[valid_mask]

                    # Get runup estimates (expected, 10, 25, 50, 75, 90 percentiles)
                    def get_stats(values, weights):
                        mean = np.sum(values * weights)
                        percentiles = np.percentile(values, [10, 25, 50, 75, 90], method = "inverted_cdf", weights=weights)
                        return np.concatenate([[mean], percentiles])

                    # Now store statistics for netcdf
                    nc_results["R2"][iprof, it,:] = get_stats(R2_flat, P_flat)
                    nc_results["R2_CfLow"][iprof, it,:] = get_stats(R2_CfLow_flat, P_flat)
                    nc_results["R2_CfHigh"][iprof, it,:] = get_stats(R2_CfHigh_flat, P_flat)
                    nc_results["R2_BsLow"][iprof, it,:] = get_stats(R2_BsLow_flat, P_flat)
                    nc_results["R2_BsHigh"][iprof, it,:] = get_stats(R2_BsHigh_flat, P_flat)

                    # Now store all (five or less) runup values in the gdf results dictionary
                    gdf_results_iprof['R2_all'].append(R2_flat)
                    gdf_results_iprof['R2_CfLow_all'].append(R2_CfLow_flat)
                    gdf_results_iprof['R2_CfHigh_all'].append(R2_CfHigh_flat)
                    gdf_results_iprof['R2_BsLow_all'].append(R2_BsLow_flat)
                    gdf_results_iprof['R2_BsHigh_all'].append(R2_BsHigh_flat)
                    gdf_results_iprof['P_all'].append(P_flat) 
                
                # Add gdf results to geodataframe per input profile
                output = {
                    'name': prof_name,
                    'geometry': prof.geometry,  # or prof['geometry']
                    'time': time,
                    'Hs': Hs,
                    'Tp': Tp,
                    'WL': WL,
                    'R2': nc_results["R2"][iprof,:],
                    'R2_CfLow': nc_results["R2_CfLow"][iprof,:],
                    'R2_CfHigh': nc_results["R2_CfHigh"][iprof,:],
                    'R2_BsLow': nc_results["R2_BsLow"][iprof,:],
                    'R2_BsHigh': nc_results["R2_BsHigh"][iprof,:],
                    'R2_all': gdf_results_iprof['R2_all'],
                    'R2_CfLow_all': gdf_results_iprof['R2_CfLow_all'],
                    'R2_CfHigh_all': gdf_results_iprof['R2_CfHigh_all'],
                    'R2_BsLow_all': gdf_results_iprof['R2_BsLow_all'],
                    'R2_BsHigh_all': gdf_results_iprof['R2_BsHigh_all'],
                    'P_all': gdf_results_iprof['P_all'],
                }
                gdf_results.append(output)

        # Close the xBeach dataset
        ds_xb.close()

        # Save the results to a GeoDataFrame and array for netcdf
        self.gdf = gpd.GeoDataFrame(gdf_results, geometry='geometry')
        self.nc = nc_results


    def write_his_file(self):

        if not hasattr(self, 'nc'):
            raise RuntimeError("No results to write. Run execute() first.")

        file_name = os.path.join(self.model.path, "beware_his.nc")

        coords = {
            "prof_id": (("prof_id"), self.model.transects.gdf['name'].values),
            "prof_x": (("prof_id"), self.model.transects.gdf.geometry.x.values),
            "prof_y": (("prof_id"), self.model.transects.gdf.geometry.y.values),
            "wave_x": (("prof_id"), self.model.boundary_conditions.gdf_wave.geometry.x.values),
            "wave_y": (("prof_id"), self.model.boundary_conditions.gdf_wave.geometry.y.values),
            "flow_x": (("prof_id"), self.model.boundary_conditions.gdf_flow.geometry.x.values),
            "flow_y": (("prof_id"), self.model.boundary_conditions.gdf_flow.geometry.y.values),
            "time": (("time"), self.model.boundary_conditions.gdf_wave.iloc[0]['timeseries'].index),
        }

        coord_attrs = {
        "prof_id": {"long_name": "Profile identifier", "description": "Unique profile ID"},
        "prof_x": {"units": "m", "long_name": "Profile X-coordinate"},
        "prof_y": {"units": "m", "long_name": "Profile Y-coordinate"},
        "wave_x": {"units": "m", "long_name": "Wave boundary X-coordinate"},
        "wave_y": {"units": "m", "long_name": "Wave boundary Y-coordinate"},
        "flow_x": {"units": "m", "long_name": "Flow boundary X-coordinate"},
        "flow_y": {"units": "m", "long_name": "Flow boundary Y-coordinate"},
        }

        data_vars = {
            "Hs": (["prof_id", "time"], self.nc["Hs"],
                {"units": "m",
                    "long_name": "Significant wave height",
                    "description": "Significant wave height at boundary",
                    "coordinates": "prof_id wave_x wave_y time"}),
            
            "Tp": (["prof_id", "time"], self.nc["Tp"],
                {"units": "sec",
                    "long_name": "Peak wave period",
                    "description": "Peak wave period at boundary",
                    "coordinates": "prof_id wave_x wave_y time"}),
            
            "WL": (["prof_id", "time"], self.nc["WL"],
                {"units": "m",
                    "long_name": "Still water level",
                    "description": "Still water level at boundary",
                    "coordinates": "prof_id flow_x flow_y time"}),
            
            "R2": (["prof_id", "time", "nR2estimates"], self.nc["R2"],
                {"units": "m",
                    "long_name": "Runup",
                    "description": "Runup (R2%) estimates (expected, 10, 25, 50, 75, 90 percentiles) per profile and forcing condition",
                    "coordinates": "prof_id prof_x prof_y time"}),
            
            "R2_CfLow": (["prof_id", "time", "nR2estimates"], self.nc["R2_CfLow"],
                        {"units": "m",
                        "long_name": "Runup for low reef roughness",
                        "description": "Runup (R2%) estimates (expected, 10, 25, 50, 75, 90 percentiles) under low reef roughness (cf = 0.01) scenario per profile and forcing condition",
                        "coordinates": "prof_id prof_x prof_y time"}),
        
            "R2_CfHigh": (["prof_id", "time", "nR2estimates"], self.nc["R2_CfHigh"],
                        {"units": "m",
                            "long_name": "Runup for high reef roughness",
                            "description": "Runup (R2%) estimates (expected, 10, 25, 50, 75, 90 percentiles) under high reef roughness (cf = 0.1) scenario per profile and forcing condition",
                            "coordinates": "prof_id prof_x prof_y time"}),
        
            "R2_BsLow": (["prof_id", "time", "nR2estimates"], self.nc["R2_BsLow"],
                        {"units": "m",
                        "long_name": "Runup for mild beach slope",
                        "description": "Runup (R2%) estimates under mild beach slope (1:20) scenario",
                        "coordinates": "prof_id prof_x prof_y time"}),
        
            "R2_BsHigh": (["prof_id", "time", "nR2estimates"], self.nc["R2_BsHigh"],
                        {"units": "m",
                            "long_name": "Runup for steep beach slope",
                            "description": "Runup (R2%) estimates (expected, 10, 25, 50, 75, 90 percentiles) under steep beach slope (1:5) scenario per profile and forcing condition",
                            "coordinates": "prof_id prof_x prof_y time"})
        }

        ds = xr.Dataset(data_vars=data_vars, coords=coords)

        # Add input variables
        inp = xr.DataArray(np.array(1, dtype='int32'))

        # Add all input variables as attributes on this DataArray
        for key, val in vars(self.model.input.variables).items():
            # Convert values to native types or strings if needed
            if isinstance(val, (int, float, str)):
                inp.attrs[key] = val
            elif hasattr(val, "isoformat"):  # e.g. datetime
                inp.attrs[key] = val.isoformat()
            else:
                # fallback: convert to string (arrays, None, etc)
                inp.attrs[key] = str(val)

        ds["inp"] = inp

        # Add coordinates attributes
        for coord, attrs in coord_attrs.items():
            ds.coords[coord].attrs |= attrs

        # Add general attributes to the dataset
        ds.attrs['title'] = "BEWARE netcdf output"
        ds.attrs['description'] = "BEWARE runup estimates"
        ds.attrs['background'] = "https://doi.org/10.5194/nhess-2024-28"
        ds.attrs['crs'] = self.model.crs.to_string()
        ds.attrs['created'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        compression = {var: {"zlib": True, "complevel": 5} for var in ds.data_vars}

        ds.to_netcdf(file_name, encoding=compression)

        # Print success message and dataset info
        print(f"NetCDF file written to {file_name}")
        print(ds)

        # Close the dataset
        ds.close()



