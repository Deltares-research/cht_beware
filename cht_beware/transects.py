# -*- coding: utf-8 -*-
"""
Created on 2025-06-05 17:17

@author: roelvink (f.e.)
"""
import geopandas as gpd
import pandas as pd
import shapely
import xarray as xr
from pathlib import Path
import time

class BewareTransects():
    def __init__(self, model):
        """
        The BEWARE Transects class contains methods to read and write transect points, and to read the profile matching file and assign matching probabilities to the input transects.
        Parameters
        ----------
        model: BEWARE model object
            The BEWARE model instance to which the transects belong.
        """

        self.model = model
        self.gdf = gpd.GeoDataFrame()

    def read(self):
        self.read_transect_points()
        self.read_transect_matching()

    def read_transect_points(self):
        """Reads transect information (coordinates, beach slope, friction) from the specified file and stores them in a GeoDataFrame."""
        start_time = time.time()
        self.model.logger.info(f"\n----------- Reading input -----------\n")

        # Read .points file
        if not self.model.input.variables.xytfile:
            # print("No transect points file specified in input variables.")
            self.model.logger.error("No transect points file specified in input variables.")
            return

        file_name = self.model.path / self.model.input.variables.xytfile

        if not file_name.exists():
            # print(f"Warning! File {file_name} does not exist!")
            self.model.logger.error(f"File {file_name} does not exist!")
            return

        # Read the bnd file
        df = pd.read_csv(file_name, index_col=False, header=None,
                         names=['x', 'y', 'beachslope', 'friction', 'name'], sep="\s+", encoding='latin1')

        gdf_list = []
        # Loop through points
        for ind in range(len(df.x.values)):
            name = df.name.values[ind]
            beachslope = df.beachslope.values[ind]
            friction = df.friction.values[ind]
            x = df.x.values[ind]
            y = df.y.values[ind]
            point = shapely.geometry.Point(x, y)
            d = {"name": name, "beachslope": beachslope, "friction": friction, "geometry": point}
            gdf_list.append(d)
        self.gdf = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)

        self.model.logger.info(f"Read {len(self.gdf)} transect points in {time.time() - start_time:.2f} seconds")
        # print(f"Read transect points from {file_name} in {time.time() - start_time:.2f} seconds")

    def read_transect_matching(self):
        """Read NetCDF file with profile matching data and assign ProbtoCR2 to gdf based on ProfID"""

        if not self.model.input.variables.r2matchfile:
            # print("No transect matching file specified in input variables.")
            self.model.logger.error("No transect matching file specified in input variables.")
            return
        
        file_name = Path(self.model.input.variables.r2matchfile) # r2matchfile can be an absolute path        
        
        start_time = time.time()

        if not file_name.is_absolute():
            file_name = Path(self.model.path) / file_name

        if not file_name.exists():
            # print(f"Warning! File {file_name} does not exist!")
            self.model.logger.error(f"File {file_name} does not exist!")
            return

        # Load NetCDF dataset
        ds = xr.open_dataset(file_name)

        # print(f"Opened transect matching data from {file_name} in {time.time() - start_time:.2f} seconds")
        self.model.logger.info(f"Opened transect matching data in {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        # Extract data from NETCDF dataset
        prof_ids = ds.ProfID.values.astype(str)     # shape: (n_profiles,)
        prob_to_cr2 = ds.ProbtoCR2.values           # shape: (n_profiles, 195)

        # Prepare a Series with ProfID as index and ProbtoCR2 as values, to assign them to gdf
        start_time = time.time()
        cr2_df = pd.Series(
            [array for array in prob_to_cr2],
            index=prof_ids
        )
        self.gdf["ProbtoCR2"] = self.gdf["name"].map(cr2_df)
    
        ds.close()

        self.model.logger.info(f"Read transect matching data in {time.time() - start_time:.2f} seconds")

    def write(self):
        """Writes the transect points to the .xyt file."""
        if len(self.gdf.index)==0:
            return

        if not self.model.input.variables.xytfile:
            self.model.input.variables.xytfile = "beware.xyt"

        file_name = self.model.path / self.model.input.variables.xytfile

        with open(file_name, "w") as fid:
            for _, row in self.gdf.iterrows():
                x, y = row["geometry"].coords[0]
                name = row["name"]
                beachslope = row["beachslope"]
                friction = row["friction"]
                if self.model.crs.is_geographic:
                    fid.write(f'{x:12.6f}{y:12.6f} {beachslope:12.2f} {friction:12.2f} {name}\n')
                    # fid.write(f'{x:12.6f}{y:12.6f} {name}\n')
                else:
                    fid.write(f'{x:12.1f}{y:12.1f} {beachslope:12.2f} {friction:12.2f} {name}\n')
                    # fid.write(f'{x:12.1f}{y:12.1f} {name}\n')