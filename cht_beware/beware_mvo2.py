"""BEWARE runup and flooding calculation — MVO2 variant.

Extended version of the BEWARE model that includes beach-slope (betab) as an
additional forcing dimension and decomposes runup into its constituent
components (water level, setup, VLF, IG, HF).
"""

import datetime
import os

import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy import io


class BEWARE:
    """BEWARE model object (MVO2 variant with beach-slope forcing dimension).

    Parameters
    ----------
    input_file : str
        Path to ``beware.inp``.  The file is read immediately on construction.
    """

    def __init__(self, input_file: str) -> None:
        self.input = BewareInput()

        # Get the path of beware.inp
        self.path = os.path.dirname(input_file)
        self.read_input_file(input_file)

        self.flow_boundary_point = []
        self.wave_boundary_point = []
        self.testprofs = []

        self.read_wave_boundary_points()
        self.read_flow_boundary_points()

    def run(
        self,
        Hs,
        Tp,
        WL,
        betab,
        testprofs,
        xbFile: str,
    ) -> None:
        """Execute the BEWARE runup/flooding interpolation.

        Parameters
        ----------
        Hs : array-like
            Significant wave height time series, shape ``(ntimes, nprofs)``.
        Tp : array-like
            Peak wave period time series, shape ``(ntimes, nprofs)``.
        WL : array-like
            Water level time series, shape ``(ntimes, nprofs)``.
        betab : array-like
            Beach slopes for each profile, length ``nprofs``.
        testprofs : array-like
            Profile IDs to process.
        xbFile : str
            Path to the XBeach NetCDF lookup file.
        """
        self.Hs = Hs
        self.Tp = Tp
        self.WL = WL
        self.betab = betab

        # Initialize: load matching for runup / flooding and initialize input / output vars
        BWvars = ["Hs", "Tp", "WL", "betab", "BWprof"]
        outvars1 = []
        matchrunup = None
        matchflooding = None
        if self.input.r2matchfile:
            matchrunup = io.loadmat(
                os.path.join(self.path, self.input.r2matchfile), simplify_cells=True
            )
            BWvars.extend(["R2pIndex", "runupComponents"])
            outvars1.extend(
                ["R2", "R2_wl", "R2_setup", "R2_vlf", "R2_ig", "R2_hf", r"R2_tot"]
            )
        if self.input.flmatchfile:
            matchflooding = io.loadmat(
                os.path.join(self.path, self.input.flmatchfile), simplify_cells=True
            )
            BWvars.extend(
                [
                    r"obs_05m.fp",
                    r"obs_05m.infra_m0",
                    r"obs_05m.fsplit",
                    r"obs_05m.gauss_scale",
                    r"obs_05m.Hm0_HF",
                    r"obs_05m.setup",
                ]
            )
            outvars1.extend([r"fsplit", r"scale", r"fp", r"Hhf", r"setup", r"m0"])

        # Load XB results
        ds = nc.Dataset(xbFile)
        BWdata = {}
        for var in BWvars:
            BWdata[str(var)] = np.array(ds[str(var)][:].data, ndmin=2)

        # Initialize BEWARE profile output
        outvars2 = [r"Prof", r"Xc", r"Yc", r"Xo", r"Yo"]
        self.out = {}
        for var in outvars1:
            a = np.empty((len(testprofs), len(Hs)))
            a[:] = np.nan
            self.out[str(var)] = a
        for var in outvars2:
            a = np.empty((len(testprofs)))
            a[:] = np.nan
            self.out[str(var)] = a

        # Initialize profile id naming
        if matchrunup:
            profid = np.zeros(len(matchrunup["ProbNS3"]["profid"]))
            for i in range(len(matchrunup["ProbNS3"]["profid"])):
                profid[i] = matchrunup["ProbNS3"]["profid"][i]
        else:
            profid = np.zeros(len(matchflooding["ProbNS3"]["profid"]))
            for i in range(len(matchflooding["ProbNS3"]["profid"])):
                profid[i] = matchflooding["ProbNS3"]["profid"][i]

        for inputprof in range(len(testprofs)):
            print(inputprof)
            ID = np.argwhere(testprofs[inputprof] == profid)[0][0]

            # Load forcing file into dictionary
            if np.shape(self.Hs)[0] == 1:
                forcing = np.array(
                    np.concatenate(
                        (
                            self.Hs[:, inputprof],
                            self.Tp[:, inputprof],
                            self.WL[:, inputprof],
                            self.betab[inputprof] * np.ones(np.shape(self.Hs)[0]),
                        )
                    ),
                    ndmin=2,
                )
            else:
                forcing = np.transpose(
                    (
                        self.Hs[:, inputprof],
                        self.Tp[:, inputprof],
                        self.WL[:, inputprof],
                        self.betab[inputprof] * np.ones(np.shape(self.Hs)[0]),
                    )
                )

            # Runup
            if matchrunup:
                prob = matchrunup["ProbNS3"]["ProbtoCR2"][
                    ID
                ]  # Get matching % of input profile to BW profiles
                idx = [
                    i for i, v in enumerate(prob) if v > 0.01
                ]  # Delete profiles with less than 1% matching

                bwProfiles = matchrunup["ProbNS3"]["CR2repProf"][
                    idx
                ]  # Get id of matched bwprofiles
                prob = prob[idx] / sum(
                    prob[idx]
                )  # correct probability of matching for deleted profiles

                if len(prob) >= 1:
                    savevars = [
                        "R2",
                        "R2_wl",
                        "R2_setup",
                        "R2_vlf",
                        "R2_ig",
                        "R2_hf",
                        r"R2_tot",
                        r"fsplit",
                        r"scale",
                        r"fp",
                        r"Hhf",
                        r"setup",
                        r"m0",
                    ]
                    save = {}
                    for var in savevars:
                        save[str(var)] = np.zeros((len(self.Hs), len(prob)))

                    for iforcings in range(
                        np.shape(forcing)[0]
                    ):  # Loop through forcing conditions
                        for iprof in range(
                            len(bwProfiles)
                        ):  # Loop through matched BEWARE profiles range(len(prob))
                            if np.isnan(forcing).any():
                                pass
                            else:
                                profval = np.where(
                                    bwProfiles[iprof] == BWdata["BWprof"]
                                )[0][0]
                                BWforcing = np.transpose(
                                    (
                                        BWdata["Hs"][:, profval],
                                        BWdata["Tp"][:, profval],
                                        BWdata["WL"][:, profval],
                                        BWdata["betab"][:, profval],
                                    )
                                )

                                # Find nearest conditions (same for all profiles so only run once per forcing condition)
                                df = forcing[iforcings, :] - BWforcing
                                lims = []
                                for ilim in range(4):
                                    lims.append(BWforcing[df[:, ilim] >= 0, ilim].max())
                                    lims.append(BWforcing[df[:, ilim] < 0, ilim].min())
                                BWinds = np.where(
                                    np.all(
                                        (
                                            (BWforcing[:, 0] == lims[0])
                                            | (BWforcing[:, 0] == lims[1]),
                                            (BWforcing[:, 1] == lims[2])
                                            | (BWforcing[:, 1] == lims[3]),
                                            (BWforcing[:, 2] == lims[4])
                                            | (BWforcing[:, 2] == lims[5]),
                                            (BWforcing[:, 3] == lims[6])
                                            | (BWforcing[:, 3] == lims[7]),
                                        ),
                                        axis=0,
                                    )
                                )

                                limsdim = [
                                    lims[1] - lims[0],
                                    lims[3] - lims[2],
                                    lims[5] - lims[4],
                                    lims[7] - lims[6],
                                ]  # distance between BW conditions
                                intpData = np.zeros((np.shape(BWinds)[1], 4))
                                intpData[0 : np.shape(BWinds)[1], 0:4] = BWforcing[
                                    BWinds, :
                                ]

                                # Calculate normalized geometric mean inverse distance
                                NGM = 1 - abs(
                                    (forcing[iforcings, :] - intpData) / (limsdim)
                                )
                                NGMiD = np.prod(NGM, axis=1) ** (1 / len(intpData[0]))
                                P = NGMiD / sum(NGMiD)

                                R2 = np.squeeze(BWdata["R2pIndex"][BWinds, profval])
                                R2comp = np.squeeze(
                                    BWdata["runupComponents"][BWinds, :, profval]
                                )

                                save["R2"][iforcings, iprof] = np.sum(
                                    R2 * P * prob[iprof]
                                )
                                save["R2_wl"][iforcings, iprof] = np.sum(
                                    R2comp[:, 0] * P * prob[iprof]
                                )
                                save["R2_setup"][iforcings, iprof] = np.sum(
                                    R2comp[:, 1] * P * prob[iprof]
                                )
                                save["R2_vlf"][iforcings, iprof] = np.sum(
                                    R2comp[:, 2] * P * prob[iprof]
                                )
                                save["R2_ig"][iforcings, iprof] = np.sum(
                                    R2comp[:, 3] * P * prob[iprof]
                                )
                                save["R2_hf"][iforcings, iprof] = np.sum(
                                    R2comp[:, 4] * P * prob[iprof]
                                )
                                save["R2_tot"][iforcings, iprof] = np.sum(
                                    R2comp[:, 5] * P * prob[iprof]
                                )

                    self.out["R2"][inputprof, :] = np.sum(save["R2"], 1)
                    self.out["R2_wl"][inputprof, :] = np.sum(save["R2_wl"], 1)
                    self.out["R2_setup"][inputprof, :] = np.sum(save["R2_setup"], 1)
                    self.out["R2_vlf"][inputprof, :] = np.sum(save["R2_vlf"], 1)
                    self.out["R2_ig"][inputprof, :] = np.sum(save["R2_ig"], 1)
                    self.out["R2_hf"][inputprof, :] = np.sum(save["R2_hf"], 1)
                    self.out["R2_tot"][inputprof, :] = np.sum(save["R2_tot"], 1)
                    self.out["Prof"][inputprof] = int(
                        matchrunup["ProbNS3"]["profid"][ID]
                    )

            # Flooding
            if matchflooding:
                prob = matchflooding["ProbNS3"]["ProbtoCR2"][
                    ID
                ]  # Get matching % of input profile to BW profiles
                idx = [
                    i for i, v in enumerate(prob) if v > 0.01
                ]  # Delete profiles with less than 1% matching

                bwProfiles = matchflooding["ProbNS3"]["CR2repProf"][
                    idx
                ]  # Get id of matched bwprofiles
                prob = prob[idx] / sum(
                    prob[idx]
                )  # correct probability of matching for deleted profiles

                if len(prob) >= 1:
                    savevars = [r"fsplit", r"scale", r"fp", r"Hhf", r"setup", r"m0"]
                    save = {}
                    for var in savevars:
                        save[str(var)] = np.zeros((len(self.Hs), len(prob)))

                    for iforcings in range(
                        np.shape(forcing)[0]
                    ):  # Loop through forcing conditions
                        for iprof in range(
                            len(bwProfiles)
                        ):  # Loop through matched BEWARE profiles range(len(prob))
                            if np.isnan(forcing).any():
                                pass
                            else:
                                profval = np.where(
                                    bwProfiles[iprof] == BWdata["BWprof"]
                                )[0][0]
                                BWforcing = np.squeeze(
                                    np.array(
                                        np.transpose(
                                            [
                                                BWdata["Hs"][:, profval],
                                                BWdata["Tp"][:, profval],
                                                BWdata["WL"][:, profval],
                                                BWdata["betab"][:, profval],
                                            ]
                                        )
                                    )
                                )

                                # Find nearest conditions
                                df = forcing[iforcings, :] - BWforcing
                                lims = []
                                for ilim in range(4):
                                    lims.append(BWforcing[df[:, ilim] >= 0, ilim].max())
                                    lims.append(BWforcing[df[:, ilim] < 0, ilim].min())
                                BWinds = np.where(
                                    np.all(
                                        (
                                            (BWforcing[:, 0] == lims[0])
                                            | (BWforcing[:, 0] == lims[1]),
                                            (BWforcing[:, 1] == lims[2])
                                            | (BWforcing[:, 1] == lims[3]),
                                            (BWforcing[:, 2] == lims[4])
                                            | (BWforcing[:, 2] == lims[5]),
                                            (BWforcing[:, 3] == lims[6])
                                            | (BWforcing[:, 3] == lims[7]),
                                        ),
                                        axis=0,
                                    )
                                )

                                limsdim = [
                                    lims[1] - lims[0],
                                    lims[3] - lims[2],
                                    lims[5] - lims[4],
                                    lims[7] - lims[6],
                                ]  # distance between BW conditions
                                intpData = np.zeros((np.shape(BWinds)[1], 4))
                                intpData[0 : np.shape(BWinds)[1], 0:4] = BWforcing[
                                    BWinds, :
                                ]

                                # Calculate normalized geometric mean inverse distance
                                NGM = 1 - abs(
                                    (forcing[iforcings, :] - intpData) / (limsdim)
                                )
                                NGMiD = np.prod(NGM, axis=1) ** (1 / len(intpData[0]))
                                P = NGMiD / sum(NGMiD)

                                save["fsplit"][iforcings, iprof] = np.sum(
                                    np.squeeze(
                                        BWdata[r"obs_05m.fsplit"][BWinds, profval]
                                    )
                                    * P
                                    * prob[iprof]
                                )
                                save["scale"][iforcings, iprof] = np.sum(
                                    np.squeeze(
                                        BWdata[r"obs_05m.gauss_scale"][BWinds, profval]
                                    )
                                    * P
                                    * prob[iprof]
                                )
                                save["m0"][iforcings, iprof] = np.sum(
                                    np.squeeze(
                                        BWdata[r"obs_05m.infra_m0"][BWinds, profval]
                                    )
                                    * P
                                    * prob[iprof]
                                )
                                save["fp"][iforcings, iprof] = np.sum(
                                    np.squeeze(BWdata[r"obs_05m.fp"][BWinds, profval])
                                    * P
                                    * prob[iprof]
                                )
                                save["Hhf"][iforcings, iprof] = np.sum(
                                    np.squeeze(
                                        BWdata[r"obs_05m.Hm0_HF"][BWinds, profval]
                                    )
                                    * P
                                    * prob[iprof]
                                )
                                save["setup"][iforcings, iprof] = np.sum(
                                    np.squeeze(
                                        BWdata[r"obs_05m.setup"][BWinds, profval]
                                    )
                                    * P
                                    * prob[iprof]
                                )

                    self.out["fp"][inputprof, :] = 0.5 * np.sum(save["fsplit"], 1)
                    self.out["m0"][inputprof, :] = np.sum(save["m0"], 1)
                    self.out["scale"][inputprof, :] = np.sum(save["scale"], 1)
                    self.out["setup"][inputprof, :] = np.sum(save["setup"], 1)

    def write_flow_boundary_points(self, file_name: str = None) -> None:
        """Write flow boundary point coordinates to the ``.bnd`` file.

        Parameters
        ----------
        file_name : str, optional
            Output path.  Defaults to ``<path>/<input.bndfile>``.
        """
        if not file_name:
            if not self.input.bndfile:
                return
            file_name = os.path.join(self.path, self.input.bndfile)
        if not file_name:
            return

        with open(file_name, "w") as fid:
            for point in self.flow_boundary_point:
                if point.data is not None:
                    string = f"{point.geometry.x:12.1f}{point.geometry.y:12.1f}"
                    fid.write(f"{string} {point.name}\n")

    def write_wave_boundary_points(self, file_name: str = None) -> None:
        """Write wave boundary point coordinates to the ``.bwv`` file.

        Parameters
        ----------
        file_name : str, optional
            Output path.  Defaults to ``<path>/<input.bwvfile>``.
        """
        if not file_name:
            if not self.input.bwvfile:
                return
            file_name = os.path.join(self.path, self.input.bwvfile)

        if not file_name:
            return

        with open(file_name, "w") as fid:
            for point in self.wave_boundary_point:
                if point.data is not None:
                    string = f"{point.geometry.x:12.1f}{point.geometry.y:12.1f}"
                    fid.write(f"{string} {point.name}\n")

    def read_flow_boundary_points(self) -> None:
        """Populate ``self.flow_boundary_point`` and ``self.testprofs`` from the profiles file."""
        from cht_sfincs.sfincs import FlowBoundaryPoint

        prof_file = os.path.join(self.path, self.input.profsfile)
        df = pd.read_csv(prof_file, index_col=False, delim_whitespace=True)

        for ind in range(len(df.x_flow.values)):
            name = df.profid.values[ind]
            point = FlowBoundaryPoint(
                df.x_flow.values[ind],
                df.y_flow.values[ind],
                name=f"transect_{int(name)}",
            )
            self.flow_boundary_point.append(point)
            self.testprofs.append(name)

    def read_wave_boundary_points(self) -> None:
        """Populate ``self.wave_boundary_point`` from the profiles file."""
        from cht_sfincs.sfincs import FlowBoundaryPoint

        prof_file = os.path.join(self.path, self.input.profsfile)
        df = pd.read_csv(prof_file, index_col=False, delim_whitespace=True)

        for ind in range(len(df.x_off.values)):
            name = df.profid.values[ind]
            point = FlowBoundaryPoint(
                df.x_off.values[ind],
                df.y_off.values[ind],
                name=f"transect_{int(name)}",
            )
            self.wave_boundary_point.append(point)

    def read_wave_boundary_conditions(self) -> None:
        """Read Hm0 and Tp wave boundary condition files."""
        self.read_bhs_file()
        self.read_btp_file()

    def read_bhs_file(self, file_name: str = None, interpolate: bool = True) -> None:
        """Read the significant wave height boundary file (``.bhs``).

        Parameters
        ----------
        file_name : str, optional
            Path to the file.  Defaults to ``<path>/<input.bhsfile>``.
        interpolate : bool, optional
            Whether to resample to the model time step.  Default is ``True``.
        """
        if not file_name:
            if not self.input.bhsfile:
                return
            file_name = os.path.join(self.path, self.input.bhsfile)

        if not file_name:
            return

        hs = pd.read_csv(file_name, index_col=0, header=None, delim_whitespace=True)

        if interpolate:
            tstart = self.input.tstart - self.input.tref
            tstop = self.input.tstop - self.input.tref

            hs.index = pd.to_timedelta(hs.index, unit="s")
            hs = hs.resample(self.input.dT).interpolate(method="time")
            indexes = hs[(hs.index < tstart) | (hs.index >= tstop)].index
            hs.drop(indexes, inplace=True)

        self.hs = hs

    def read_btp_file(self, file_name: str = None, interpolate: bool = True) -> None:
        """Read the peak wave period boundary file (``.btp``).

        Parameters
        ----------
        file_name : str, optional
            Path to the file.  Defaults to ``<path>/<input.btpfile>``.
        interpolate : bool, optional
            Whether to resample to the model time step.  Default is ``True``.
        """
        if not file_name:
            if not self.input.btpfile:
                return
            file_name = os.path.join(self.path, self.input.btpfile)

        if not file_name:
            return

        tp = pd.read_csv(file_name, index_col=0, header=None, delim_whitespace=True)

        if interpolate:
            tstart = self.input.tstart - self.input.tref
            tstop = self.input.tstop - self.input.tref

            tp.index = pd.to_timedelta(tp.index, unit="s")
            tp = tp.resample(self.input.dT).interpolate(method="time")
            indexes = tp[(tp.index < tstart) | (tp.index >= tstop)].index
            tp.drop(indexes, inplace=True)

        self.tp = tp

    def read_flow_boundary_conditions(
        self, file_name: str = None, interpolate: bool = True
    ) -> None:
        """Read the water-level boundary file (``.bzs``).

        Parameters
        ----------
        file_name : str, optional
            Path to the file.  Defaults to ``<path>/<input.bzsfile>``.
        interpolate : bool, optional
            Whether to resample to the model time step.  Default is ``True``.
        """
        if not file_name:
            if not self.input.bzsfile:
                return
            file_name = os.path.join(self.path, self.input.bzsfile)

        if not file_name:
            return

        wl = pd.read_csv(file_name, index_col=0, header=None, delim_whitespace=True)

        if interpolate:
            tstart = self.input.tstart - self.input.tref
            tstop = self.input.tstop - self.input.tref

            wl.index = pd.to_timedelta(wl.index, unit="s")
            wl = wl.resample(self.input.dT).interpolate(method="time")
            indexes = wl[(wl.index < tstart) | (wl.index >= tstop)].index
            wl.drop(indexes, inplace=True)

        self.wl = wl

    def write_wave_boundary_conditions(self) -> None:
        """Write Hm0 and Tp wave boundary condition files."""
        self.write_bhs_file()
        self.write_btp_file()

    def write_bhs_file(self, file_name: str = None) -> None:
        """Write the significant wave height boundary file (``.bhs``).

        Parameters
        ----------
        file_name : str, optional
            Output path.  Defaults to ``<path>/<input.bhsfile>``.
        """
        if not file_name:
            if not self.input.bhsfile:
                return
            file_name = os.path.join(self.path, self.input.bhsfile)

        if not file_name:
            return

        point_data = []
        for point in self.wave_boundary_point:
            if point.data is not None:
                point_data.append(point.data["hm0"])
        df = pd.concat(point_data, axis=1)

        tmsec = pd.to_timedelta(df.index - self.input.tref, unit="s")
        df.index = tmsec.total_seconds()
        df.to_csv(file_name, index=True, sep=" ", header=False, float_format="%0.3f")

    def write_btp_file(self, file_name: str = None) -> None:
        """Write the peak wave period boundary file (``.btp``).

        Parameters
        ----------
        file_name : str, optional
            Output path.  Defaults to ``<path>/<input.btpfile>``.
        """
        if not file_name:
            if not self.input.btpfile:
                return
            file_name = os.path.join(self.path, self.input.btpfile)

        if not file_name:
            return

        point_data = []
        for point in self.wave_boundary_point:
            if point.data is not None:
                point_data.append(point.data["tp"])
        df = pd.concat(point_data, axis=1)

        tmsec = pd.to_timedelta(df.index - self.input.tref, unit="s")
        df.index = tmsec.total_seconds()
        df.to_csv(file_name, index=True, sep=" ", header=False, float_format="%0.3f")

    def write_flow_boundary_conditions(self, file_name: str = None) -> None:
        """Write the water-level boundary file (``.bzs``).

        Parameters
        ----------
        file_name : str, optional
            Output path.  Defaults to ``<path>/<input.bzsfile>``.
        """
        if not file_name:
            if not self.input.bzsfile:
                return
            file_name = os.path.join(self.path, self.input.bzsfile)

        if not file_name:
            return

        point_data = []
        for point in self.flow_boundary_point:
            point_data.append(point.data)

        df = pd.concat(point_data, axis=1)
        print("write flow data")

        tmsec = pd.to_timedelta(df.index - self.input.tref, unit="s")
        df.index = tmsec.total_seconds()
        df.to_csv(file_name, index=True, sep=" ", header=False, float_format="%0.3f")
        print("Finish write flow data")

    def write_input_file(self, input_file: str = None) -> None:
        """Write the BEWARE input file (``beware.inp``).

        Parameters
        ----------
        input_file : str, optional
            Output path.  Defaults to ``<path>/beware.inp``.
        """
        if not input_file:
            input_file = os.path.join(self.path, "beware.inp")

        with open(input_file, "w") as fid:
            for key, value in self.input.__dict__.items():
                if value is not None:
                    if type(value) == "float":
                        string = f"{key.ljust(20)} = {float(value)}\n"
                    elif type(value) == "int":
                        string = f"{key.ljust(20)} = {int(value)}\n"
                    elif type(value) == list:
                        valstr = " ".join(str(v) for v in value)
                        string = f"{key.ljust(20)} = {valstr}\n"
                    elif isinstance(value, datetime.date):
                        dstr = value.strftime("%Y%m%d %H%M%S")
                        string = f"{key.ljust(20)} = {dstr}\n"
                    else:
                        string = f"{key.ljust(20)} = {value}\n"
                    fid.write(string)

    def read_input_file(self, inputfile: str) -> None:
        """Parse a ``beware.inp`` key=value file and populate ``self.input``.

        Parameters
        ----------
        inputfile : str
            Path to the input file.
        """
        with open(inputfile, "r") as fid:
            lines = fid.readlines()

        for line in lines:
            parts = line.split("=")
            if len(parts) == 1:
                # Empty line
                continue
            name = parts[0].strip()
            val = parts[1].strip()
            try:
                # First try to convert to int
                val = int(val)
            except ValueError:
                try:
                    # Now try to convert to float
                    val = float(val)
                except Exception:
                    pass
            if name == "tref":
                try:
                    val = datetime.datetime.strptime(val.rstrip(), "%Y%m%d %H%M%S")
                except Exception:
                    val = None
            if name == "tstart":
                try:
                    val = datetime.datetime.strptime(val.rstrip(), "%Y%m%d %H%M%S")
                except Exception:
                    val = None
            if name == "tstop":
                try:
                    val = datetime.datetime.strptime(val.rstrip(), "%Y%m%d %H%M%S")
                except Exception:
                    val = None
            setattr(self.input, name, val)

    def read_data(self, input_file: str = None, prcs=None) -> None:
        """Load BEWARE output from a NetCDF history file.

        Parameters
        ----------
        input_file : str, optional
            Path to ``beware_his.nc``.  Defaults to
            ``<cycle_path>/output/beware_his.nc``.
        prcs : array-like, optional
            Fractional percentile values to also read (e.g. ``[0.5, 0.95]``).
        """
        if not input_file:
            output_path = os.path.join(self.cycle_path, "output\\")
            input_file = os.path.join(output_path, "beware_his.nc")

        ds = nc.Dataset(input_file)
        self.R2p = np.nan_to_num(ds[r"R2_tot"][:].data, copy=False, nan=0.0)
        self.setup = np.nan_to_num(ds[r"R2_set"][:].data, copy=False, nan=0.0)
        self.Hs = np.nan_to_num(ds[r"Hs"][:].data, copy=False, nan=0.0)
        self.Tp = np.nan_to_num(ds[r"Tp"][:].data, copy=False, nan=0.0)
        self.WL = np.nan_to_num(ds[r"WL"][:].data, copy=False, nan=0.0)
        self.filename = ds[r"Profiles"][:].data
        self.swash = self.R2p - self.setup - self.WL

        self.xp = ds[r"x_coast"][:].data
        self.yp = ds[r"y_coast"][:].data

        self.xo = ds[r"x_off"][:].data
        self.yo = ds[r"y_off"][:].data
        self.R2p_prc, self.setup_prc = {}, {}
        if prcs is not None:
            for i, v in enumerate(prcs):
                self.R2p_prc[str(round(v * 100))] = np.nan_to_num(
                    ds[f"R2_tot_{round(v * 100)}"][:].data, copy=False, nan=0.0
                )
                self.setup_prc[str(round(v * 100))] = np.nan_to_num(
                    ds[f"R2_set_{round(v * 100)}"][:].data, copy=False, nan=0.0
                )

        if not self.input.tstart:
            ttt = ds["time"][:]
            dt = datetime.timedelta(seconds=ttt[0])
            tout = datetime.datetime(1970, 1, 1) + dt
            self.input.tstart = tout


class BewareInput:
    """Container for BEWARE input parameters read from ``beware.inp``."""

    def __init__(self) -> None:
        self.tref = None
        self.tstart = None
        self.tstop = None
        self.dT = None
        self.r2matchfile = None
        self.flmatchfile = None
        self.profsfile = "beware.profs"
        self.bndfile = "beware.bnd"
        self.bzsfile = "beware.bzs"
        self.bwvfile = "beware.bwv"
        self.bhsfile = "beware.bhs"
        self.btpfile = "beware.btp"


class BewareProfiles:
    """Container for BEWARE profile characteristics read from the ``.profs`` file."""

    def __init__(self) -> None:
        self.betab = None
        self.xc = None
        self.yc = None
        self.xo = None
        self.yo = None
        self.coasttype = None
        self.profid = None
        self.profsfile = "beware.profs"

    def read_profile_characteristics(self, file_name: str = None) -> None:
        """Read profile properties from a whitespace-delimited CSV file.

        Parameters
        ----------
        file_name : str, optional
            Path to the profiles file.  Defaults to
            ``<path>/<profsfile>``.
        """
        if not file_name:
            if not self.profsfile:
                return
            if not self.path:
                return
            file_name = os.path.join(self.path, self.profsfile)

        if not file_name:
            return

        df = pd.read_csv(file_name, index_col=False, delim_whitespace=True)

        self.betab = df.beachslope.values
        self.xc = df.x_coast.values
        self.yc = df.y_coast.values
        self.xo = df.x_off.values
        self.yo = df.y_off.values
        self.xf = df.x_flow.values
        self.yf = df.y_flow.values
        self.coasttype = df.coasttype.values
        self.profid = df.profid.values
