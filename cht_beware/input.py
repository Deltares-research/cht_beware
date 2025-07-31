# -*- coding: utf-8 -*-
"""
Created on 2025-06-05 13:29

@author: roelvink (f.e.)
"""

import datetime
import copy

class Variables():
    def __init__(self):
        """Initialize the BEWARE variables with default values.
        
        This method sets the default values for various model parameters.

        Parameters
        ----------
        - tref: datetime object, reference time for the model.
        - tstart: datetime object, start time for the model simulation.
        - tstop: datetime object, stop time for the model simulation.
        - r2matchfile: str, file name for probabilistic matching results of test profiles to BW2 profiles.
        - xytfile: str, file name for transect information [x y beach_slope cf profile_name].
        - bndfile: str, file name for water level boundary points [x y profile_name].
        - bzsfile: str, file name for water level time series [time1 zs0 zs1 .. zsn] (n number of profiles).
        - bwvfile: str, file name for wave boundary points [x y profile_name].
        - bhsfile: str, file name for wave height time series [time1 hs0 hs1 .. hsn] .
        - btpfile: str, file name for wave period time series [time1 tp0 tp1 .. tpn].
        - epsg: int, EPSG code for the coordinate reference system.
        - xbdatabase: str, absolute path to the xbeach runup database file.
        """

        tnow = datetime.datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        self.tref = tnow
        self.tstart = tnow
        self.tstop = tnow + datetime.timedelta(hours=1)
        self.r2matchfile = None
        self.xytfile = "beware.xyt"
        self.bndfile = "beware.bnd"
        self.bzsfile= "beware.bzs"
        self.bwvfile= "beware.bwv"
        self.bhsfile= "beware.bhs"
        self.btpfile= "beware.btp"
        self.epsg = 4326
        self.xbdatabase = None # absolute path to the xb database file

class BewareInput():

    def __init__(self, model):
        """
        The BEWARE Input class contains methods to read, write and update the BEWARE input variables:

        Key Methods:

        - __init__:
            Initializes the input parameters with default values.
        - read:
            Reads the "beware.inp" file and updates the model parameters.
        - write: 
            Writes the current input parameters to the "beware.inp" file.
        - update:
            Updates the model's input parameters with new values.

        Parameters
        ----------
        model: BEWARE model object
            The BEWARE model instance to which the input parameters belong.
        """
            
        self.model = model
        self.variables = Variables()

    def read(self):
        """Reads the input parameters from the "beware.inp" file and updates the model variables."""

        input_file = self.model.path / "beware.inp"

        fid = open(input_file, "r")
        lines = fid.readlines()
        fid.close()
        for line in lines:
            # Remove everything from line starting with first #
            if "#" in line:
                line = line.split("#")[0]
            strings = line.split("=")
            if len(strings) == 1:
                # Empty line
                continue
            name = strings[0].strip()
            val = strings[1].strip()
            try:
                # First try to convert to int
                val = int(val)
            except ValueError:
                try:
                    # Now try to convert to float
                    val = float(val)
                except:
                    pass
            if name == "tref":
                try:
                    val = datetime.datetime.strptime(val.rstrip(), "%Y%m%d %H%M%S")
                except:
                    val = None
            if name == "tstart":
                try:
                    val = datetime.datetime.strptime(val.rstrip(), "%Y%m%d %H%M%S")
                except:
                    val = None
            if name == "tstop":
                try:
                    val = datetime.datetime.strptime(val.rstrip(), "%Y%m%d %H%M%S")
                except:
                    val = None

            if hasattr(self.variables, name):
                if type(getattr(self.variables, name)) is bool:
                    if val == 0:
                        val = False       
                    elif val == 1:
                        val = True
                    elif val[0].lower() == "t" or val[0].lower() == "y":
                        val = True
                    elif val[0].lower() == "f" or val[0].lower() == "n":
                        val = False
                    else:
                        # Use default value
                        val = getattr(self.variables, name)

            setattr(self.variables, name, val)

        self.model.logger.info("---------- Model Variables ----------\n")
        for key, value in self.variables.__dict__.items():
            self.model.logger.info(f"  {key.ljust(12)} = {value}")

    def write(self):
        """Writes the current input parameters to the "beware.inp" file."""

        input_file = self.model.path / "beware.inp"

        # Make some adjustments
        variables = copy.copy(self.variables)
        variables.epsg = self.model.crs.to_epsg()

        fid = open(input_file, "w")
        for key, value in variables.__dict__.items():
            if value is not None:
                if type(value) == "float":
                    string = f"{key.ljust(20)} = {float(value)}\n"
                elif type(value) == "int":
                    string = f"{key.ljust(20)} = {int(value)}\n"
                elif isinstance(value, bool):
                    if value:
                        string = f"{key.ljust(20)} = {int(1)}\n"
                    else:    
                        string = f"{key.ljust(20)} = {int(0)}\n"
                elif type(value) == list:
                    valstr = ""
                    for v in value:
                        valstr += str(v) + " "
                    string = f"{key.ljust(20)} = {valstr}\n"
                elif isinstance(value, datetime.date):
                    dstr = value.strftime("%Y%m%d %H%M%S")
                    string = f"{key.ljust(20)} = {dstr}\n"
                else:
                    string = f"{key.ljust(20)} = {value}\n"
                fid.write(string)
        fid.close()

    def update(self, pars):
        """Updates the BEWARE input parameters with new values."""
        for key in pars:
            setattr(self.variables, key, pars[key])