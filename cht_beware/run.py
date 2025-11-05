# -*- coding: utf-8 -*-
"""
Created on 2025-06-06 16:02

@author: roelvink (f.e.)
"""
import xarray as xr
import numpy as np
from pathlib import Path
import geopandas as gpd
import datetime
import time

class BewareRun:

    def __init__(self, model):
        """
        The BEWARE Run class contains methods to execute and save BEWARE runs.

        Parameters
        ----------
        model: BEWARE model object
            The BEWARE model instance to which the run belongs.
        """
        self.model = model

    def execute(self, write_his=True):
        """Execute the BEWARE run, reading input data and calculating runup estimates."""

        if not self.model.input.variables.xbdatabase:
            self.model.logger.error("XBeach run database (.nc) not set in input variables. Please set 'xbdatabase' in beware.inp")
            return
        
        # Input transects with Prob of Matching to CR2
        profs = self.model.transects.gdf
                        
        # XBeach database
        file_name = Path(self.model.input.variables.xbdatabase)
        if not file_name.is_absolute():
            file_name = self.model.path / file_name

        if not file_name.exists():
            self.model.logger.error(f"XBeach database file {file_name} does not exist!")
            return
        
        start_input = time.perf_counter()
        ds_xb = xr.open_dataset(file_name)
        xbcoords = {}
        for var in ['Hs', 'Tp', 'WL', 'CfMod', 'BsMod']:
            xbcoords[var] = ds_xb[var].values  # 1D arrays over nConditions
        end_input = time.perf_counter()
        self.model.logger.info(f"Read XBeach database in {end_input - start_input:.2f} seconds")

        # Initialize outputs - align dimensions
        self.model.logger.info(f"\n------------- Simulation ------------\n")

        n_profiles = len(profs)
        n_forcings = len(self.model.boundary_conditions.gdf_wave.iloc[0]['timeseries']['hs'].values)
        n_r2estimates = 6
        nc_results = {"R2": np.full((n_profiles, n_forcings, n_r2estimates), np.nan),
                      "R2_base": np.full((n_profiles, n_forcings, n_r2estimates), np.nan),
                      "R2_CfLow": np.full((n_profiles, n_forcings, n_r2estimates), np.nan),
                      "R2_CfHigh": np.full((n_profiles, n_forcings, n_r2estimates), np.nan),
                      "R2_BsLow": np.full((n_profiles, n_forcings, n_r2estimates), np.nan),
                      "R2_BsHigh": np.full((n_profiles, n_forcings, n_r2estimates), np.nan),
                      'Hs'  : np.full((n_profiles, n_forcings), np.nan),
                      'Tp'  : np.full((n_profiles, n_forcings), np.nan),
                      'WL'  : np.full((n_profiles, n_forcings), np.nan),
                    }
        self.model.logger.info(f"Initialized output arrays for {n_profiles} profiles and {n_forcings} forcing conditions.")

        gdf_results = []

        # Loop through the input profiles
        start_interp = time.perf_counter()
        for iprof, prof in profs.iterrows():
            prof_name = prof["name"]
            prof_cf = prof["friction"]
            prof_bs = prof["beachslope"]

            # Get the best R2 correction value (friction and beach slope)
            if prof_cf == 0.05 and prof_bs == 0.1:
                R2string = 'R2_base'
            elif prof_cf == 0.1:
                R2string = 'R2_CfHigh'
            elif prof_cf == 0.01:
                R2string = 'R2_CfLow'
            elif prof_bs == 0.05:
                R2string = 'R2_BsLow'
            elif prof_bs == 0.2:
                R2string = 'R2_BsHigh'
            else:
                self.model.logger.warning(
                    f"Profile '{prof_name}': unexpected friction ({prof_cf}) or beachslope ({prof_bs}). Defaulting to R2_base."
                )
                R2string = 'R2_base'

            self.model.logger.info(f"Processing profile {prof_name} ({iprof+1}/{n_profiles})")

            # Get the RRP ids and matching probabilities
            prob2BW2 = prof['ProbtoCR2']
            RRPids = np.where(prob2BW2 > 0.01)[0]
            prob = prob2BW2[RRPids]
            prob /= prob.sum()

            # Get the forcing conditions for this profile
            Hs = self.model.boundary_conditions.gdf_wave.iloc[iprof]['timeseries']['hs'].values
            Tp = self.model.boundary_conditions.gdf_wave.iloc[iprof]['timeseries']['tp'].values
            WL = self.model.boundary_conditions.gdf_flow.iloc[iprof]['timeseries']['wl'].values
            time_index = self.model.boundary_conditions.gdf_wave.iloc[iprof]['timeseries'].index

            # Store forcing conditions in the output arrays
            nc_results['Hs'][iprof, :] = Hs
            nc_results['Tp'][iprof, :] = Tp
            nc_results['WL'][iprof, :] = WL

            # Initialize gdf results for this profile
            gdf_results_iprof = {"R2_all": [],
                        "R2_base_all": [],
                        "R2_CfLow_all": [],
                        "R2_CfHigh_all": [],
                        "R2_BsLow_all": [],
                        "R2_BsHigh_all": [],
                        "P_all": [],
                        }
        
            if len(prob)>=1:

                # Loop through the forcing conditions
                for it in range(len(time_index)):

                    # Initialize arrays per timestep
                    R2_base_it = np.full((len(RRPids), 5, 8), np.nan) # Nr of RRPs x 5 R2 estimates x 8 neighboring conditions
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

                    # Adjust period when Hs, Tp combi is not present
                    while id_forcing.size<8:
                        Tp[it] = np.ceil(Tp[it] + 1)+0.01
                        lower_bound = [xbcoords[k][xbcoords[k] <= val].max() for k, val in zip(['Hs', 'Tp', 'WL'], [Hs[it], Tp[it], WL[it]])]
                        upper_bound = [xbcoords[k][xbcoords[k] > val].min() for k, val in zip(['Hs', 'Tp', 'WL'], [Hs[it], Tp[it], WL[it]])]
                        xb_mask = (
                                (xbcoords['Hs'] >= lower_bound[0]) & (xbcoords['Hs'] <= upper_bound[0]) &
                                (xbcoords['Tp'] >= lower_bound[1]) & (xbcoords['Tp'] <= upper_bound[1]) &
                                (xbcoords['WL'] >= lower_bound[2]) & (xbcoords['WL'] <= upper_bound[2])
                            )
                        id_forcing = np.where(xb_mask)[0]
                        if id_forcing.size>0:        
                            self.model.logger.info(f"Adjusted period to Tp = {Tp[it]}")
                       
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
                        R2_base_it[irrp, :] = R2data
                        R2_CfLow_it[irrp, :] = R2data * R2CfMod[0,:]
                        R2_CfHigh_it[irrp, :] = R2data * R2CfMod[1,:]
                        R2_BsLow_it[irrp, :] = R2data * R2BsMod[0,:]
                        R2_BsHigh_it[irrp, :] = R2data * R2BsMod[1,:]
                    
                    # Now remove values where P_it is zero
                    valid_mask = P_it.flatten() > 0
                    R2_base_flat     = R2_base_it.flatten()[valid_mask]
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

                    # Get best R2 estimate for this profile friction and beach slope
                    R2_best_flat = eval(f"{R2string}_flat")

                    # Now store statistics for netcdf
                    nc_results["R2"][iprof, it, :] = get_stats(R2_best_flat, P_flat)
                    nc_results["R2_base"][iprof, it,:] = get_stats(R2_base_flat, P_flat)
                    nc_results["R2_CfLow"][iprof, it,:] = get_stats(R2_CfLow_flat, P_flat)
                    nc_results["R2_CfHigh"][iprof, it,:] = get_stats(R2_CfHigh_flat, P_flat)
                    nc_results["R2_BsLow"][iprof, it,:] = get_stats(R2_BsLow_flat, P_flat)
                    nc_results["R2_BsHigh"][iprof, it,:] = get_stats(R2_BsHigh_flat, P_flat)

                    # Now store all (five or less) runup values in the gdf results dictionary
                    gdf_results_iprof['R2_all'].append(R2_best_flat)
                    gdf_results_iprof['R2_base_all'].append(R2_base_flat)
                    gdf_results_iprof['R2_CfLow_all'].append(R2_CfLow_flat)
                    gdf_results_iprof['R2_CfHigh_all'].append(R2_CfHigh_flat)
                    gdf_results_iprof['R2_BsLow_all'].append(R2_BsLow_flat)
                    gdf_results_iprof['R2_BsHigh_all'].append(R2_BsHigh_flat)
                    gdf_results_iprof['P_all'].append(P_flat) 
                
                # Add gdf results to geodataframe per input profile
                output = {
                    'name': prof_name,
                    'geometry': prof.geometry,  # or prof['geometry']
                    'beachslope': prof_bs,
                    'friction': prof_cf,
                    'time': time_index,
                    'Hs': Hs,
                    'Tp': Tp,
                    'WL': WL,
                    'R2': nc_results["R2"][iprof,:],
                    'R2_base': nc_results["R2_base"][iprof,:],
                    'R2_CfLow': nc_results["R2_CfLow"][iprof,:],
                    'R2_CfHigh': nc_results["R2_CfHigh"][iprof,:],
                    'R2_BsLow': nc_results["R2_BsLow"][iprof,:],
                    'R2_BsHigh': nc_results["R2_BsHigh"][iprof,:],
                    'R2_all': gdf_results_iprof['R2_all'],
                    'R2_base_all': gdf_results_iprof['R2_base_all'],
                    'R2_CfLow_all': gdf_results_iprof['R2_CfLow_all'],
                    'R2_CfHigh_all': gdf_results_iprof['R2_CfHigh_all'],
                    'R2_BsLow_all': gdf_results_iprof['R2_BsLow_all'],
                    'R2_BsHigh_all': gdf_results_iprof['R2_BsHigh_all'],
                    'P_all': gdf_results_iprof['P_all'],
                }
                gdf_results.append(output)

        end_interp = time.perf_counter()
        self.model.logger.info(f"Processed {len(gdf_results)} profiles with runup estimates.")

        # Close the xBeach dataset
        ds_xb.close()

        # Save the results to a GeoDataFrame and array for netcdf
        start_output = time.perf_counter()
        self.gdf = gpd.GeoDataFrame(gdf_results, geometry='geometry')
        self.nc = nc_results

        if write_his:
            self.write_his_file()
            self.model.logger.info("Results written to beware_his.nc")
        end_output = time.perf_counter()

        report = generate_report(self.model, sim_time=end_interp - start_interp, output_time=end_output - start_output)     
        self.model.logger.info(report)

    def write_his_file(self):
        """Writes the results to a NetCDF file."""

        if not hasattr(self, 'nc'):
            raise RuntimeError("No results to write. Run execute() first.")

        file_name = self.model.path / "beware_his.nc"

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
                        
            "R2_base": (["prof_id", "time", "nR2estimates"], self.nc["R2_base"],
                {"units": "m",
                    "long_name": "Runup base case",
                    "description": "Runup (R2%) estimates (expected, 10, 25, 50, 75, 90 percentiles) per profile and forcing condition under medium reef roughness (cf = 0.05) and beach slope (1:10)",
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
                        "description": "Runup (R2%) estimates (expected, 10, 25, 50, 75, 90 percentiles) under mild beach slope (1:20) scenario per profile and forcing condition",
                        "coordinates": "prof_id prof_x prof_y time"}),
        
            "R2_BsHigh": (["prof_id", "time", "nR2estimates"], self.nc["R2_BsHigh"],
                        {"units": "m",
                            "long_name": "Runup for steep beach slope",
                            "description": "Runup (R2%) estimates (expected, 10, 25, 50, 75, 90 percentiles) under steep beach slope (1:5) scenario per profile and forcing condition",
                            "coordinates": "prof_id prof_x prof_y time"})
        }

        ds = xr.Dataset(data_vars=data_vars, coords=coords)

        # Add input variables for reproducibility
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

        # Close the dataset
        ds.close()


def generate_report(model, sim_time, output_time):
    """Generate a report of the simulation run times for beware.log"""
    import re

    if (model.path / "beware.log").exists():
        with open(model.path / "beware.log", "r", encoding="utf-8") as f:
            log_text = f.read()
        # Look between the "Reading input" and "Simulation" sections for input times
        match = re.search(r'------- Reading input ----------(.*?)------- Simulation ----------', log_text, re.DOTALL)
        reading_section = match.group(1)
        times = re.findall(r'in ([\d.]+) seconds', reading_section)
        input_time = sum(float(t) for t in times)
    else: 
        input_time = 0.0

    return (
        "\n-------- Simulation finished --------\n\n"
        f" Total time             : {sim_time+input_time+output_time:10.3f}\n"
        f" Total simulation time  : {sim_time:10.3f}\n"
        f" Time in input          : {input_time:10.3f}\n"
        f" Time in output         : {output_time:10.3f}\n"
    )