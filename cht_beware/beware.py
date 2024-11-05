# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 13:16:54 2022

@author: roelvink

BEWARE runup & flooding calculation
"""
import pandas as pd
import numpy as np
import xarray as xr
import pathlib
from scipy import stats
import os
import netCDF4 as nc
import mat73
from scipy import io
# from sklearn import neighbors
import array as ar
#import datetime as dt
from scipy import interpolate as intp
from pyproj import CRS
import time
import sys
import datetime

import cht_utils.misc_tools

class Point():        
    def __init__(self, x, y, name = None, crs=None):        
        self.x       = x
        self.y       = y
        self.crs     = crs
        self.name    = name
        self.data    = None

class BoundaryPoint():
    def __init__(self, x, y, name=None, crs=None, data=None):        
        self.name                   = name
        self.geometry               = Point(x, y, crs=crs)
        self.data                   = data

class BEWARE:
    
    def __init__(self, input_file=None, crs=None):               
       
        self.input = BewareInput()
        
        self.flow_boundary_point = []
        self.wave_boundary_point = []
        
        if input_file:
            self.path = os.path.dirname(input_file)
            self.load(input_file)
            
    def load(self, inputfile):
        self.read_input_file(inputfile)
                
        self.profiles = BewareProfiles()
        self.profiles.read_profile_characteristics(file_name = os.path.join(self.path, self.input.profsfile))
        
        self.read_wave_boundary_points()
        self.read_flow_boundary_points()
                       
    def run(self, Hs=None, Tp=None, WL=None, betab=None, testprofs=None, xbFile=None):

        Hs = self.Hs if Hs is None else Hs
        Tp = self.Tp if Tp is None else Tp
        WL = self.WL if WL is None else WL
        testprofs = self.profiles.profid if testprofs is None else testprofs
                        
        # Initialize: load matching for runup / flooding and initialize input / output vars
        BWvars=["Hs", "Tp", "WL", 'BWprof']
        outvars1 = []
        if self.input.r2matchfile:
            matchrunup = io.loadmat(os.path.join(self.path, self.input.r2matchfile), simplify_cells=True)
            BWvars.extend(["R2pIndex", "setup"])
            outvars1.extend(["R2", "R2_setup"])
        if self.input.flmatchfile:
            matchflooding = io.loadmat(os.path.join(self.path, self.input.flmatchfile), simplify_cells=True)
            BWvars.extend([r"obs_25m.fp_ig", r"obs_25m.m0_ig", r"obs_25m.setup"])
            outvars1.extend([r"fp_ig",  r"m0_ig",  r"setup"])
            
        self.out = {str(var): np.full((len(testprofs), len(Hs)), np.nan) for var in outvars1}
        self.out['Prof'] = np.full(len(testprofs), np.nan)
        
        # Load XB results        
        ds = nc.Dataset(xbFile)
        BWdata = {str(var): np.array(ds[str(var)][:].data, ndmin=2) for var in BWvars}
        
        # Initialize profile id naming
        if self.input.r2matchfile:
            profid= np.array(matchrunup['ProbNS3']['profid']).astype(float)
        else:
            profid= np.array(matchflooding['ProbNS3']['profid']).astype(float)
            
        for inputprof in range(len(testprofs)):                            
            print(inputprof)
            # t = time.time()
            ID = np.argwhere(testprofs[inputprof]==profid)[0][0]

            # Load forcing file into dictionary
            if np.shape(Hs)[0] == 1:
                forcing=np.array(np.concatenate((Hs[:, inputprof], Tp[:, inputprof], WL[:, inputprof])), ndmin=2)
            else:
                forcing=np.transpose((Hs[:, inputprof], Tp[:, inputprof], WL[:, inputprof]))

            # Runup
            if self.input.r2matchfile:
                prob=matchrunup['ProbNS3']['ProbtoCR2'][ID]            # Get matching % of input profile to BW profiles
                idx= [i for i,v in enumerate(prob) if v > 0.01]             # Delete profiles with less than 1% matching
                
                bwProfiles= matchrunup['ProbNS3']['CR2repProf'][idx]             # Get id of matched bwprofiles
                prob=prob[idx] / sum(prob[idx])                             # correct probability of matching for deleted profiles
            
                if len(prob)>=1:
    
                    savevars= ["R2", "R2_setup"]
                    save = {var: np.zeros((len(Hs), len(prob))) for var in savevars}
                            
                    for iforcings in range(np.shape(forcing)[0]): # Loop through forcing conditions    
                        for iprof in range(len(bwProfiles)): # Loop through matched BEWARE profiles range(len(prob))
                            if np.isnan(forcing).any():
                                pass
                            else:
    
                                profval= np.where(bwProfiles[iprof]==BWdata['BWprof'])[0][0]
                                BWforcing=np.transpose((BWdata['Hs'][:,profval], BWdata['Tp'][:,profval],
                                                                            BWdata['WL'][:,profval])) 
                        
                                # Find nearest conditions (same for all profiles so only run once per forcing condition)  
                                df = forcing[iforcings,:]-BWforcing
                                lims=[]
                                for ilim in range(3):
                                    lims.append(BWforcing[df[:,ilim]>=0, ilim].max())
                                    lims.append(BWforcing[df[:,ilim]<0, ilim].min())
                                BWinds=np.where(np.all(((BWforcing[:,0]==lims[0]) | (BWforcing[:,0]==lims[1]), (BWforcing[:,1]==lims[2]) | (BWforcing[:,1]==lims[3]), 
                                    (BWforcing[:,2]==lims[4]) | (BWforcing[:,2]==lims[5])), axis=0))
                                    
                                limsdim=[lims[1]-lims[0], lims[3]-lims[2], lims[5]-lims[4]] # distance between BW conditions        
                                intpData= np.zeros((np.shape(BWinds)[1], 3))
                                intpData[0:np.shape(BWinds)[1], 0:3]= BWforcing[BWinds,:]
                            
                                # Calculate normalized geometric mean inverse distance
                                NGM= (1-abs((forcing[iforcings,:] - intpData) / (limsdim)))
                                NGMiD=np.prod(NGM, axis=1)**(1/len(intpData[0]))
                                P= NGMiD/ sum(NGMiD)
                                
                                R2= np.squeeze(BWdata['R2pIndex'][BWinds, profval])
                                setup= np.squeeze(BWdata['setup'][BWinds, profval])
                                    
                                save['R2'][iforcings, iprof]=np.sum(R2*P*prob[iprof])
                                save['R2_setup'][iforcings,iprof]= np.sum(setup*P*prob[iprof]) 
                            
                    self.out['R2'][inputprof,:]       = np.sum(save['R2'],1)
                    self.out['R2_setup'][inputprof,:] = np.sum(save['R2_setup'],1)
                    self.out['Prof'][inputprof]        =  int(matchrunup['ProbNS3']['profid'][ID])    
                              
                
            # Flooding
            if self.input.flmatchfile:
                prob=matchflooding['ProbNS3']['ProbtoCR2'][ID]            # Get matching % of input profile to BW profiles
                idx= [i for i,v in enumerate(prob) if v > 0.01]             # Delete profiles with less than 1% matching
                
                bwProfiles= matchflooding['ProbNS3']['CR2repProf'][idx]             # Get id of matched bwprofiles
                prob=prob[idx] / sum(prob[idx])                             # correct probability of matching for deleted profiles
                  
                if len(prob)>=1:
    
                    savevars= [r"fp_ig", r"m0_ig", r"setup"]
                    save = {var: np.zeros((len(Hs), len(prob))) for var in savevars}
        
                    for iforcings in range(np.shape(forcing)[0]): # Loop through forcing conditions
                    
                        for iprof in range(len(bwProfiles)): # Loop through matched BEWARE profiles range(len(prob))
                            if np.isnan(forcing).any():
                                pass
                            else:
    
                                profval= np.where(bwProfiles[iprof]==BWdata['BWprof'])[0][0]
                                BWforcing=np.transpose((BWdata['Hs'][:,profval], BWdata['Tp'][:,profval],
                                                                            BWdata['WL'][:,profval])) 
                        
                                # Find nearest conditions (same for all profiles so only run once per forcing condition)  
                                df = forcing[iforcings,:]-BWforcing
                                lims=[]
                                for ilim in range(3):
                                    lims.append(BWforcing[df[:,ilim]>=0, ilim].max())
                                    lims.append(BWforcing[df[:,ilim]<0, ilim].min())
                                BWinds=np.where(np.all(((BWforcing[:,0]==lims[0]) | (BWforcing[:,0]==lims[1]), (BWforcing[:,1]==lims[2]) | (BWforcing[:,1]==lims[3]), 
                                    (BWforcing[:,2]==lims[4]) | (BWforcing[:,2]==lims[5])), axis=0))
                                    
                                limsdim=[lims[1]-lims[0], lims[3]-lims[2], lims[5]-lims[4]] # distance between BW conditions        
                                intpData= np.zeros((np.shape(BWinds)[1], 3))
                                intpData[0:np.shape(BWinds)[1], 0:3]= BWforcing[BWinds,:]
                            
                                # Calculate normalized geometric mean inverse distance
                                NGM= (1-abs((forcing[iforcings,:] - intpData) / (limsdim)))
                                NGMiD=np.prod(NGM, axis=1)**(1/len(intpData[0]))
                                P= NGMiD/ sum(NGMiD)
                                                                                                
                                save['fp_ig'][iforcings, iprof]= np.sum(np.squeeze(BWdata[r'obs_25m.fp_ig'][BWinds, profval])*P*prob[iprof]) 
                                save['m0_ig'][iforcings, iprof]= np.sum(np.squeeze(BWdata[r'obs_25m.m0_ig'][BWinds, profval])*P*prob[iprof])          
                                save['setup'][iforcings, iprof]= np.sum(np.squeeze(BWdata[r'obs_25m.setup'][BWinds, profval])*P*prob[iprof])        
                                    
                    self.out['fp_ig'][inputprof,:]  =  np.sum(save['fp_ig'],1)
                    self.out['m0_ig'][inputprof,:]  = np.sum(save['m0_ig'],1)
                    self.out['setup'][inputprof,:]   = np.sum(save['setup'],1)

    def write_flow_boundary_points(self, file_name=None):
    
        # Write BEWARE profs file
        if not file_name:
            if not self.input.bndfile:
                return
            file_name = os.path.join(self.path,
                                     self.input.bndfile)
        if not file_name:
            return

        fid = open(file_name, "w")
        for point in self.flow_boundary_point:
            if point.data is not None:
                string = f'{point.geometry.x:12.6f}{point.geometry.y:12.6f}'
                fid.write(string + r' ' + str(point.name) + '\n')
        fid.close()    

    def write_wave_boundary_points(self, file_name=None):
    
        # Write BEWARE profs file
        if not file_name:
            if not self.input.bwvfile:
                return
            file_name = os.path.join(self.path,
                                     self.input.bwvfile)
            
        if not file_name:
            return
            
        fid = open(file_name, "w")
        for point in self.wave_boundary_point:
            if point.data is not None:
                string = f'{point.geometry.x:12.6f}{point.geometry.y:12.6f}'
                fid.write(string + r' ' + str(point.name) + '\n')
        fid.close()  

    def read_flow_boundary_points(self):
        
        # Read BEWARE profs file
        # from cht_sfincs.sfincs import FlowBoundaryPoint
        
        # Loop through points
        for ind in range(len(self.profiles.xf)):
            name = self.profiles.profid[ind]
            point = BoundaryPoint(self.profiles.xf[ind],
                                      self.profiles.yf[ind],
                                       name=r'transect_' + str(int(name)))
                                      # name= str(int(name))) 
            self.flow_boundary_point.append(point)
            
    def read_wave_boundary_points(self):
        
        # Read BEWARE profs file
        # from cht_sfincs.sfincs import FlowBoundaryPoint
                        
        # Loop through points
        for ind in range(len(self.profiles.xo)):
            name = self.profiles.profid[ind]
            point = BoundaryPoint(self.profiles.xo[ind],
                                      self.profiles.yo[ind],
                                       name=r'transect_' + str(int(name)))
                                      # name= str(int(name))) 
            self.wave_boundary_point.append(point)             
        
    def read_wave_boundary_conditions(self):

        # Hm0, Tp
        self.read_bhs_file()
        self.read_btp_file()

    def read_bhs_file(self, file_name=None, interpolate = True):
        
        # Read BEWARE bhs file
        if not file_name:
            if not self.input.bhsfile:
                return
            file_name = os.path.join(self.path,
                                     self.input.bhsfile)
            
        if not file_name:
            return        

        hs= pd.read_csv(file_name, index_col=0, header=None,
                  delim_whitespace=True)
        
        # Interpolate to required time intervals
        if interpolate:
            tstart = (self.input.tstart - self.input.tref)
            tstop  = (self.input.tstop - self.input.tref)

            hs.index=pd.to_timedelta(hs.index, unit="s")
            hs = hs.resample(self.input.dT).interpolate(method='time')
            indexes = hs[(hs.index<tstart) | (hs.index>=tstop)].index
            hs.drop(indexes, inplace=True)

        for icol, point in enumerate(self.wave_boundary_point):
            point.hs = pd.Series(hs.iloc[:,icol].values, index=hs.index)
            
        self.Hs= hs

    def read_btp_file(self, file_name=None, interpolate = True):

        # Read BEWARE btp file
        if not file_name:
            if not self.input.btpfile:
                return
            file_name = os.path.join(self.path,
                                     self.input.btpfile)
            
        if not file_name:
            return        

        tp= pd.read_csv(file_name, index_col=0, header=None,
                  delim_whitespace=True)
        
        # Interpolate to required time intervals
        if interpolate:
            tstart = (self.input.tstart - self.input.tref)
            tstop  = (self.input.tstop - self.input.tref)

            tp.index=pd.to_timedelta(tp.index, unit="s")
            tp = tp.resample(self.input.dT).interpolate(method='time')
            indexes = tp[(tp.index<tstart) | (tp.index>=tstop)].index
            tp.drop(indexes, inplace=True)

        for icol, point in enumerate(self.wave_boundary_point):
            point.tp = pd.Series(tp.iloc[:,icol].values, index=tp.index)
            
        self.Tp= tp

    def read_flow_boundary_conditions(self, file_name=None, interpolate = True):
        # Read BEWARE bhs file
        if not file_name:
            if not self.input.bzsfile:
                return
            file_name = os.path.join(self.path,
                                     self.input.bzsfile)
            
        if not file_name:
            return        
        
        if not os.path.exists(file_name):
            return

        wl= pd.read_csv(file_name, index_col=0, header=None,
            delim_whitespace=True)

        if interpolate:
            tstart = (self.input.tstart - self.input.tref)
            tstop  = (self.input.tstop - self.input.tref)

            wl.index=pd.to_timedelta(wl.index, unit="s")
            wl = wl.resample(self.input.dT).interpolate(method='time')
            indexes = wl[(wl.index<tstart) | (wl.index>=tstop)].index
            wl.drop(indexes, inplace=True)
        
        for icol, point in enumerate(self.flow_boundary_point):
            point.data = pd.Series(wl.iloc[:,icol].values, index=wl.index)

        self.WL=wl

    def write_wave_boundary_conditions(self, path=None):

        # Hm0, Tp, etc given (probably forced with SnapWave)
        self.write_bhs_file(path=path)
        self.write_btp_file(path=path)

    def write_bhs_file(self, file_name=None, path=None):
        # Write BEWARE bhs file
        if not path:
            path=self.path
        if not file_name and not self.input.bhsfile:
            return
        if not file_name:
            file_name = os.path.join(path, self.input.bhsfile)  
        
        point_data = []
        for point in self.wave_boundary_point:
            if point.data is not None:
                # df = pd.concat([df, point.data['hm0']], axis=1)
                point_data.append(point.data['hm0'])
        df = pd.concat(point_data, axis=1)
        
        tmsec = pd.to_timedelta(df.index - self.input.tref, unit="s")
        df.index = tmsec.total_seconds()
        # df=df.replace(np.NaN, 0.1)
        df.to_csv(file_name,
                  index=True,
                  sep=" ",
                  header=False,
                  float_format="%0.3f")
        
    def write_btp_file(self, file_name=None, path=None):
        # Write BEWARE btp file
        if not path:
            path=self.path
        if not file_name and not self.input.btpfile:
            return
        if not file_name:
            file_name = os.path.join(path, self.input.btpfile)
            
        point_data = []
        for point in self.wave_boundary_point:
            if point.data is not None:
                point_data.append(point.data['tp'])
        df = pd.concat(point_data, axis=1)

        tmsec = pd.to_timedelta(df.index - self.input.tref, unit="s")
        df.index = tmsec.total_seconds()
        #df=df.replace(np.NaN, 5.0)        
        df.to_csv(file_name,
                  index=True,
                  sep=" ",
                  header=False,
                  float_format="%0.3f")
        
    def write_flow_boundary_conditions(self, file_name=None):

        if not file_name and not self.input.bzsfile:
            return
        if not file_name:
            file_name = os.path.join(self.path, self.input.bzsfile)

        # Build a new DataFrame
        point_data = []
        for point in self.flow_boundary_point:
            point_data.append(point.data)
            # df = pd.concat([df, point.data], axis=1)
            
        df = pd.concat(point_data, axis=1)

        tmsec = pd.to_timedelta(df.index - self.input.tref, unit="s")
        df.index = tmsec.total_seconds()
        df.to_csv(file_name,
                  index=True,
                  sep=" ",
                  header=False,
                  float_format="%0.3f")
    
    def write_input_file(self, input_file=None):

        if not input_file:
            input_file = os.path.join(self.path, "beware.inp")
            
        fid = open(input_file, "w")
        for key, value in self.input.__dict__.items():
            if not value is None:
                if type(value) == "float":
                    string = f'{key.ljust(20)} = {float(value)}\n'
                elif type(value) == "int":
                    string = f'{key.ljust(20)} = {int(value)}\n'
                elif type(value) == list:
                    valstr = ""
                    for v in value:
                        valstr += str(v) + " "
                    string = f'{key.ljust(20)} = {valstr}\n'
                elif isinstance(value, datetime.date):
                    dstr = value.strftime("%Y%m%d %H%M%S")
                    string = f'{key.ljust(20)} = {dstr}\n'
                else:
                    string = f'{key.ljust(20)} = {value}\n'                
                fid.write(string)
        fid.close()    

        
    def read_input_file(self, inputfile):
        
        # Reads beware.inp
        
        fid = open(inputfile, 'r')
        lines = fid.readlines()
        fid.close()
        for line in lines:
            str = line.split("=")
            if len(str)==1:
               # Empty line
               continue
            name = str[0].strip()
            val  = str[1].strip()
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
                    val = datetime.datetime.strptime(val.rstrip(), '%Y%m%d %H%M%S')
                except:
                    val = None
            if name == "tstart":
                try:
                    val = datetime.datetime.strptime(val.rstrip(), '%Y%m%d %H%M%S')
                except:
                    val = None
            if name == "tstop":
                try:
                    val = datetime.datetime.strptime(val.rstrip(), '%Y%m%d %H%M%S')
                except:
                    val = None
            setattr(self.input, name, val)

    def read_data(self, input_file=None, prcs=None):
        if not input_file:
            output_path = os.path.join(self.cycle_path, "output\\")
            input_file= os.path.join(output_path, 'beware_his.nc')

        ds = nc.Dataset(input_file)
        self.R2=np.nan_to_num(ds[r"R2"][:].data, copy=False, nan=0.0)
        self.R2_setup=np.nan_to_num(ds[r"R2_setup"][:].data, copy=False, nan=0.0)
        self.Hs=np.nan_to_num(ds[r"Hs"][:].data, copy=False, nan=0.0)
        self.Tp=np.nan_to_num(ds[r"Tp"][:].data, copy=False, nan=0.0)
        self.WL=np.nan_to_num(ds[r"WL"][:].data, copy=False, nan=0.0)
        self.filename=ds[r"Profiles"][:].data
        self.swash=self.R2-self.R2_setup-self.WL

        self.xp=ds[r"x_coast"][:].data
        self.yp=ds[r"y_coast"][:].data
        
        self.xo=ds[r"x_off"][:].data
        self.yo=ds[r"y_off"][:].data
        self.R2_prc, self.R2_setup_prc={},{}
        if prcs is not None:
            for i,v in enumerate(prcs):
                self.R2_prc[str(round(v))]= np.nan_to_num(ds[r"R2_"+str(round(v))][:].data, copy=False, nan=0.0)
                self.R2_setup_prc[str(round(v))]= np.nan_to_num(ds[r"R2_setup_"+str(round(v))][:].data, copy=False, nan=0.0)

        if not self.input.tstart:
            ttt = ds["time"][:]
            dt  = datetime.timedelta(seconds=ttt[0])
            tout = datetime.datetime(1970,1,1) + dt
            self.input.tstart = tout
        
    def outline(self, crs=None):

        xg = xo
        yg = yo
        
        if crs:
            transformer = Transformer.from_crs(self.crs,
                                               crs,
                                               always_xy=True)
            xg, yg = transformer.transform(xg, yg)
        
        xp = [ xg[0,0], xg[0,-1], xg[-1,-1], xg[-1,0], xg[0,0] ]
        yp = [ yg[0,0], yg[0,-1], yg[-1,-1], yg[-1,0], yg[0,0] ]
        
        return xp, yp
#     def write_to_geojson(self, output_path, scenario):
#         from geojson import Point, Feature, FeatureCollection, dump
#         from pyproj import Transformer

#         features = []
#         transformer = Transformer.from_crs(self.crs,
#                                            'WGS 84',
#                                            always_xy=True)
        
#         for ip in range(len(self.filename)):
#             x, y = transformer.transform(self.xp[ip],
#                                          self.yp[ip])
#             point = Point((x, y))
#             name = 'Loc nr: ' +  str(self.filename[ip])
                        
#             id = np.argmax(self.R2p[ip,:])                                                                       
#             features.append(Feature(geometry=point,
#                                     properties={"LocNr":int(self.filename[ip]),
#                                                 "Lon":x,
#                                                 "Lat":y,                                                
#                                                 "Setup":round(self.setup[ip, id],2),
#                                                 "Swash":round(self.swash[ip, id],2),
#                                                 "TWL":round(self.R2p[ip, id],2)}))
        
#         feature_collection = FeatureCollection(features)
        
#         if features:
#             feature_collection = FeatureCollection(features)
#             output_path_runup =  os.path.join(output_path, 'extreme_runup_height\\')
#             os.mkdir(output_path_runup)
#             file_name = os.path.join(output_path_runup,
#                                     "extreme_runup_height.geojson.js")
#             cht_utils.misc_tools.write_json_js(file_name, feature_collection, "var runup =")
            
#         features = []
            
#         for ip in range(len(self.filename)):
#             x, y = transformer.transform(self.xo[ip],
#                                          self.yo[ip])
#             point = Point((x, y))
#             name = 'Loc nr: ' +  str(self.filename[ip])
                        
#             id = np.argmax(self.R2p[ip,:])                                                                       
#             features.append(Feature(geometry=point,
#                                     properties={"LocNr":int(self.filename[ip]),
#                                                 "Lon": x,
#                                                 "Lat": y,
#                                                 "Hs":round(self.Hs[ip, id],2),
#                                                 "Tp":round(self.Tp[ip, id],1),
#                                                 "WL":round(self.WL[ip, id],2)}))
        
#         feature_collection = FeatureCollection(features)
        
#         if features:
#             feature_collection = FeatureCollection(features)
#             output_path_waves =  os.path.join(output_path, 'extreme_sea_level_and_wave_height\\')
#             os.mkdir(output_path_waves)
#             file_name = os.path.join(output_path_waves,     
#                                     "extreme_sea_level_and_wave_height.geojson.js")
#             cht_utils.misc_tools.write_json_js(file_name, feature_collection, "var swl =")
#         # with open(output_path + r"\\" + scenario + '.TWL.geojson.js', 'w') as fl:
#         #     fl.write('const point_' + scenario + '_TWL = ')
#         #     dump(feature_collection, fl)
#         #     fl.write("  \n   \n")
#         #     fl.write('pt_' + scenario + '_' + 'BT' + '_TWL.addData(point_' + scenario + '_TWL);')
    
        
#     def write_to_csv(self, output_path, scenario):
#         from geojson import Point, Feature, FeatureCollection, dump
#         from pyproj import Transformer

#         transformer = Transformer.from_crs(self.crs,
#                                            'WGS 84',
#                                            always_xy=True)
#         features = []
            
#         for ip in range(len(self.filename)):
#             x, y = transformer.transform(self.xp[ip],
#                                          self.yp[ip])
#             point = Point((x, y))
#             name = 'Loc nr: ' +  str(self.filename[ip])
                        
#             obs_file = "extreme_runup_height." + self.runid + "." +str(self.filename[ip]) + ".csv.js"
                                                          
#             features.append(Feature(geometry=point,
#                                     properties={"name":int(self.filename[ip]),
#                                                 "LocNr":int(self.filename[ip]),
#                                                 "Lon":x,
#                                                 "Lat":y,
#                                                 "model_name":self.name,
#                                                 "model_type":self.type,
#                                                 "TWL":  np.round(np.max(self.R2p[ip,:]),2),
#                                                 "obs_file": obs_file}))
#             d= {'WL': self.WL[ip,:],'Setup': self.setup[ip,:], 'Swash': self.swash[ip,:], 'Runup': self.R2p[ip,:]}       
#             v= pd.DataFrame(data=d, index =  pd.date_range(self.input.tstart, periods=len(self.swash[ip,:]), freq= '0.5H'))
    
#             local_file_path = os.path.join(output_path,  "timeseries",
#                                                obs_file)
# #            local_file_path = os.path.join(output_path,  
# #                                           obs_file)
#             s= v.to_csv(path_or_buf=None,
#                          date_format='%Y-%m-%dT%H:%M:%S',
#                          float_format='%.3f',
#                          header= False, index_label= 'datetime')        
            
#             cht_utils.misc_tools.write_csv_js(local_file_path, s, "var csv = `date_time,wl,setup,swash,runup")
                             
#         if features:
#             feature_collection = FeatureCollection(features)
#             runup_file = os.path.join(output_path,
#                                     "twls.geojson.js")
#             cht_utils.misc_tools.write_json_js(runup_file, feature_collection, "var TWL =")
            

class BewareInput():
    def __init__(self):
        
        self.tref = None
        self.tstart = None
        self.tstop = None
       # self.folder = []
        self.dT = None
        self.r2matchfile = None
        self.flmatchfile = None
        self.profsfile = "beware.profs"
        self.bndfile = "beware.bnd"
        self.bzsfile= "beware.bzs"
        self.bwvfile= "beware.bwv"
        self.bhsfile= "beware.bhs"
        self.btpfile= "beware.btp"

class BewareProfiles():
    def __init__(self):
        self.betab=None
        self.xc= None
        self.yc= None
        self.xo= None
        self.yo= None
        self.xf= None
        self.yf= None
        self.profid= None
        self.profsfile = "beware.profs"

    def read_profile_characteristics(self, file_name= None):

        # Read profs file
        if not file_name:
            if not self.profsfile:
                return
            if not self.path:
                return
            file_name = os.path.join(self.path,
                                     self.profsfile)

        if not file_name:
            return

        df = pd.read_csv(file_name, index_col=False,
                         sep="\s+")

        self.betab= df.beachslope.values
        self.xc= df.x_coast.values
        self.yc= df.y_coast.values
        self.xo= df.x_off.values
        self.yo= df.y_off.values
        self.xf= df.x_flow.values
        self.yf= df.y_flow.values
        self.profid= df.profid.values