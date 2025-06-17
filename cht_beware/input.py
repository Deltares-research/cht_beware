# -*- coding: utf-8 -*-
"""
Created on 2025-06-05 13:29

@author: roelvink (f.e.)
"""

import os
import datetime
import copy

class Variables():
    def __init__(self):
        tnow = datetime.datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        self.tref = tnow
        self.tstart = tnow
        self.tstop = tnow + datetime.timedelta(hours=1)
       # self.folder = []
        self.dT = None
        self.r2matchfile = None
        self.flmatchfile = None
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
        self.model = model
        self.variables = Variables()

    def read(self):
        # Reads beware.inp

        input_file = os.path.join(self.model.path, "beware.inp")

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

    def write(self):

        # Write beware.inp
        input_file = os.path.join(self.model.path, "beware.inp")

        # Make some adjustments
        variables = copy.copy(self.variables)

        if self.model.crs.is_geographic:
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