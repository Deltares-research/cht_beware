# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 13:16:54 2022

@author: roelvink

BEWARE runup & flooding calculation
"""
import numpy as np
from pathlib import Path
from pyproj import CRS
from typing import Union
import logging

from .input import BewareInput
from .transects import BewareTransects
from .boundary_conditions import BewareBoundaryConditions
from .output import BewareOutput
from .run import BewareRun

class BEWARE:
    
    def __init__(
        self,
        root: Union[str, Path] = None,
        crs: Union[str, CRS] = 4326,
        write_log: bool = True
    ):
        """
        The BEWARE class contains methods to read, write and run the BEWARE model.

        Parameters
        ----------
        root: str, Path, optional
            Path to model folder. If None, current working directory is used.
        crs: int, str, CRS, optional
            Coordinate reference system of the model. Can be an EPSG code (int), or a CRS object.
            Default is 4326 (WGS84).
        write_log: bool, optional
            If True, a log file will be created in the model folder. Default is True.
        """
         
        self.path = Path(root) if root else Path.cwd()
        self.input                    = BewareInput(self)
        if isinstance(crs, int):
            crs = CRS.from_epsg(crs)
        self.crs                      = crs

        self.transects                = BewareTransects(self)
        self.boundary_conditions      = BewareBoundaryConditions(self)
        self.output                   = BewareOutput(self)

        # Initialize logger
        if (self.path / 'beware.log').exists():
            (self.path / 'beware.log').unlink()
        self.logger = logging.getLogger(f"beware_{self.path.name}")
        handler = logging.FileHandler(self.path / "beware.log", encoding="utf-8") if write_log else logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def read(self):
        """Reads the input and attribute files"""
        # Reads beware.inp and attribute files
        self.input.read()
        if self.input.variables.epsg is not None:
            self.crs = CRS.from_epsg(self.input.variables.epsg)
        self.read_attribute_files()

    def write(self):
        """Writes the input and attribute files"""
        # Writes beware.inp and attribute files
        self.input.write()
        self.write_attribute_files()

    def read_attribute_files(self):
        """Reads all attribute files"""

        # Read transects
        self.transects.read()

        # Boundary conditions (reads bnd and bzs, bwv and bhs and btp files)
        self.boundary_conditions.read()

    def write_attribute_files(self):
        """Writes all attribute files"""

        # Write transects
        self.transects.write()

        # Boundary conditions
        self.boundary_conditions.write()

    def run_simulation(self):
        """Runs the BEWARE model simulation"""
        self.run = BewareRun(self)
        self.run.execute()
