# -*- coding: utf-8 -*-
"""
Created on 2025-06-05 13:45

@author: roelvink (f.e.)
"""

import geopandas as gpd
import shapely
import pandas as pd
from tabulate import tabulate
import time

class BewareBoundaryConditions:
    
    def __init__(self, model):
        """
        The BEWARE boundary conditions class contains methods to read and write flow and wave boundary conditions.

        Parameters
        ----------
        model: BEWARE model object
            The BEWARE model instance to which the boundary conditions belong.
        gdf_flow: GeoDataFrame
            GeoDataFrame containing flow boundary points with their coordinates, names, and timeseries [wl].
        gdf_wave: GeoDataFrame
            GeoDataFrame containing wave boundary points with their coordinates, names, and timeseries [hs, tp].
        """
        self.model = model
        self.gdf_flow = gpd.GeoDataFrame()
        self.gdf_wave = gpd.GeoDataFrame()
        self.times = []

    def read(self):
        self.read_boundary_points('flow') # Read water level boundary points
        self.read_boundary_points('wave') # Read wave boundary points
        self.read_boundary_conditions_timeseries('flow')
        self.read_boundary_conditions_timeseries('wave')
        
    def write(self):
        # Write all boundary data
        self.write_boundary_points('flow')
        self.write_boundary_points('wave')
        self.write_boundary_conditions_timeseries('flow')
        self.write_boundary_conditions_timeseries('wave')

    def read_boundary_points(self, bc_type:str = 'flow'):
        """ Reads boundary points from the specified file type and stores them in a GeoDataFrame.

        Parameters
        ----------
        bc_type: str
            The type of boundary condition ('flow' or 'wave').
        """
        if bc_type == 'flow':
            file_type = 'bndfile'
        elif bc_type == 'wave':
            file_type = 'bwvfile'
        gdf_name = 'gdf_' + bc_type

        file_name = getattr(self.model.input.variables, file_type)
        if not file_name:
            return

        file_name = self.model.path / file_name

        if not file_name.exists():
            raise FileNotFoundError(f"File {file_name} does not exist!")

        # Read the bnd file
        df = pd.read_csv(file_name, index_col=False, header=None,
                         names=['x', 'y', 'name'], sep=r"\s+", encoding='latin1')

        gdf_list = []
        # Loop through points
        for ind in range(len(df.x.values)):
            name = df.name.values[ind]
            x = df.x.values[ind]
            y = df.y.values[ind]
            point = shapely.geometry.Point(x, y)
            d = {"name": name, "timeseries": pd.DataFrame(), "geometry": point}
            gdf_list.append(d)
        setattr(self, gdf_name, gpd.GeoDataFrame(gdf_list, crs=self.model.crs))

    def write_boundary_points(self, bc_type:str = 'flow'):
        """
        Writes boundary point coordinates to the  (.bnd) or (.bwv) files.
        Parameters
        ----------
        bc_type: str
            The type of boundary condition ('flow' or 'wave').
        """
        if bc_type == 'flow':
            file_type = 'bndfile'
        elif bc_type == 'wave':
            file_type = 'bwvfile'
        gdf_name = 'gdf_' + bc_type
        gdf = getattr(self, gdf_name)
        if len(gdf.index) == 0:
            return

        file_name = getattr(self.model.input.variables, file_type, None)
        if not file_name:
            file_name = "beware.bnd" if gdf_name == 'gdf_flow' else "beware.bwv"
            setattr(self.model.input.variables, file_type, file_name)

        file_name = self.model.path / file_name

        with open(file_name, "w") as fid:
            for _, row in gdf.iterrows():
                x, y = row["geometry"].coords[0]
                name = row["name"]
                if self.model.crs.is_geographic:
                    fid.write(f'{x:12.6f}{y:12.6f} {name}\n')
                else:
                    fid.write(f'{x:12.1f}{y:12.1f} {name}\n')


    def read_boundary_conditions_timeseries(self, bc_type:str = 'flow'):
        """
        Reads boundary time series for water level (zs) or waves (hs, tp) into the GeoDataFrame.

        Parameters
        ----------
        bc_type: str
            The type of boundary condition ('flow' or 'wave').
        """
        start_time = time.time()

        if bc_type == 'flow':
            file_types = ['bzsfile',]
            variables = ['wl']
        elif bc_type == 'wave':
            file_types = ['bhsfile', 'btpfile']
            variables = ['hs', 'tp']
        gdf_name = 'gdf_' + bc_type
        gdf = getattr(self, gdf_name, None)
        if len(gdf) == 0:
            return
        
        tref = self.model.input.variables.tref

        # Now read files 
        for file_type, var in zip(file_types, variables):
            file_name = getattr(self.model.input.variables, file_type, None)
            if not file_name:
                continue

            file_name = self.model.path / file_name
            if not file_name.exists():
                print(f"Warning! File {file_name} does not exist!")
                continue

            df_ts = read_timeseries_file(file_name, tref)

            # Loop through boundary points
            for ip, point in gdf.iterrows():
                if 'time' not in point["timeseries"]:
                    point["timeseries"]["time"] = df_ts.index
                    point["timeseries"].set_index("time", inplace=True)
                point["timeseries"][var] = df_ts.iloc[:, ip].values

        self.model.logger.info(f"Read boundary conditions timeseries for {bc_type} in {time.time() - start_time:.2f} seconds")

    def write_boundary_conditions_timeseries(self, bc_type:str = 'flow'):
        """
        Writes boundary time series for water level (zs) or waves (hs, tp) from the GeoDataFrame.

        Parameters
        ----------
        bc_type: str
            The type of boundary condition ('flow' or 'wave').
        """
        gdf_name = 'gdf_' + bc_type
        gdf = getattr(self, gdf_name, None)
        if len(gdf.index) == 0:
            return
        
        time = gdf.loc[0]["timeseries"].index
        tref = self.model.input.variables.tref
        dt = (time - tref).total_seconds()

        def write_timeseries(file_type, var, gdf):
            if not getattr(self.model.input.variables, file_type):
                setattr(self.model.input.variables, file_type, f"beware.b{var}")
            file_name = self.model.path / getattr(self.model.input.variables, file_type)
            df = pd.DataFrame({ip: point["timeseries"][var] for ip, point in gdf.iterrows()})
            df.index = dt
            to_fwf(df, file_name)

        if bc_type == 'flow':
            write_timeseries("bzsfile", "zs", gdf)
        elif bc_type == 'wave':
            write_timeseries("bhsfile", "hs", gdf)
            write_timeseries("btpfile", "tp", gdf)


def read_timeseries_file(file_name, ref_date):
    """
    Reads a time series file and returns a DataFrame.
    
    Parameters
    ----------
    file_name : str
        Path to the time series file.
    ref_date : datetime
        Reference date for time indexing.
    
    Returns
    -------
    DataFrame
        DataFrame with time series indexed by time.
    """
    df = pd.read_csv(file_name, index_col=0, header=None, sep=r"\s+")
    df.index = ref_date + pd.to_timedelta(df.index, unit="s")
    return df

def to_fwf(df, fname, floatfmt=".3f"):
    """
    Writes a DataFrame to a fixed-width formatted file.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame to write.
    fname : str
        Path to the output file.
    floatfmt : str, optional
        Floating point format (default is '.3f').
    """
    indx = df.index.tolist()
    vals = df.values.tolist()
    for it, t in enumerate(vals):
        t.insert(0, indx[it])
    content = tabulate(vals, [], tablefmt="plain", floatfmt=floatfmt)
    with open(fname, "w") as f:
        f.write(content)
