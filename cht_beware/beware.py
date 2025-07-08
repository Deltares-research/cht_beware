# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 13:16:54 2022

@author: roelvink

BEWARE runup & flooding calculation
"""
import numpy as np
import os
from pyproj import CRS

from .input import BewareInput
from .transects import BewareTransects
from .boundary_conditions import BewareBoundaryConditions
from .output import BewareOutput
from .run import BewareRun

class BEWARE:
    
    def __init__(self,  root=None, crs=4326, mode="w", read_transect_data=True):               
       
        if not root:
            root = os.getcwd()
    
        self.path                     = root  
        self.input                    = BewareInput(self)
        if isinstance(crs, int):
            crs = CRS.from_epsg(crs)
        self.crs                      = crs

        self.transects                = BewareTransects(self)
        self.boundary_conditions      = BewareBoundaryConditions(self)
        self.output                   = BewareOutput(self)
        
        if mode == "r":
            self.input.read()
            if self.input.variables.epsg is not None:
                self.crs = CRS.from_epsg(self.input.variables.epsg)
            self.read_attribute_files(read_transect_data=read_transect_data)

    def read(self):
        # Reads beware.inp and attribute files
        self.input.read()
        if self.input.variables.epsg is not None:
            self.crs = CRS.from_epsg(self.input.variables.epsg)
        self.read_attribute_files()

    def write(self):
        # Writes beware.inp and attribute files
        self.input.write()
        self.write_attribute_files()

    def read_attribute_files(self, read_transect_data=True):
        
        if read_transect_data:
            self.transects.read()

        # Boundary conditions (reads bnd and bzs, bwv and bhs and btp files)
        self.boundary_conditions.read()

    def write_attribute_files(self):
        """Writes all attribute files"""

        # Write transects
        self.transects.write()

        # Boundary conditions
        self.boundary_conditions.write()

    def run_simulation(self, mode = 'run'):
        
        self.run = BewareRun(self)    
        self.run.execute()   
        if mode == 'write':
            self.run.write_his_file()