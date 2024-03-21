# MIT License
#
# Copyright (c) 2024, Geir Drange
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from enum import Enum, unique
import toml

__all__ = ["Source", "ILoad", "PLoad", "RLoad", "Loss", "Converter", "LinReg"]


@unique
class ComponentTypes(Enum):
    """Component types"""

    SOURCE = 1
    LOAD = 2
    LOSS = 3
    CONVERTER = 4
    LINREG = 5


MAX_DEFAULT = 1.0e6
IQ_DEFAULT = 0.0
IIS_DEFAULT = 0.0
RS_DEFAULT = 0.0
VDROP_DEFAULT = 0.0
PWRS_DEFAULT = 0.0
LIMITS_DEFAULT = {
    "vi": [0.0, MAX_DEFAULT],
    "vo": [0.0, MAX_DEFAULT],
    "ii": [0.0, MAX_DEFAULT],
    "io": [0.0, MAX_DEFAULT],
}


def _get_opt(params, key, default):
    """Get optional parameter from dict"""
    if key in params:
        return params[key]
    return default


def _get_mand(params, key):
    """Get mandatory parameter from dict"""
    if key in params:
        return params[key]
    raise KeyError("Parameter dict is missing entry for '{}'".format(key))


def _get_warns(limits, checks):
    """Check parameter values against limits"""
    warn = ""
    keys = list(checks.keys())
    for key in keys:
        lim = _get_opt(limits, key, [0, MAX_DEFAULT])
        if abs(checks[key]) > abs(lim[1]) or abs(checks[key]) < abs(lim[0]):
            warn += key + " "
    return warn


def _get_eff(ipwr, opwr, def_eff=100.0):
    """Calculate efficiency in %"""
    if ipwr > 0.0:
        return 100.0 * abs(opwr / ipwr)
    return def_eff


class ComponentMeta(type):
    """An component metaclass that will be used for component class creation."""

    def __instancecheck__(cls, instance):
        return cls.__subclasscheck__(type(instance))

    def __subclasscheck__(cls, subclass):
        return (
            hasattr(subclass, "component_type")
            and hasattr(subclass, "child_types")
            and hasattr(subclass, "from_file")
            and callable(subclass.from_file)
        )


class ComponentInterface(metaclass=ComponentMeta):
    """This interface is used for concrete component classes to inherit from.
    There is no need to define the ComponentMeta methods of any class
    as they are implicitly made available via .__subclasscheck__().
    """

    pass


class Source:
    """The Source component must the root of a system.
    A system can only have one source.

    Attributes
    ----------
    component_type : ComponentTypes.SOURCE (enum)
        type of component
    """

    @property
    def component_type(self):
        """Defines the component type"""
        return ComponentTypes.SOURCE

    @property
    def child_types(self):
        """Defines allowable child component types"""
        et = list(ComponentTypes)
        et.remove(ComponentTypes.SOURCE)
        return et

    def __init__(
        self,
        name: str,
        *,
        vo: float,
        rs: float = RS_DEFAULT,
        limits: dict = LIMITS_DEFAULT,
    ):
        """Set source name, voltage, internal resistance and max current"""
        self.params = {}
        self.params["name"] = name
        self.params["vo"] = vo
        self.params["rs"] = abs(rs)
        self.limits = limits

    @classmethod
    def from_file(cls, name: str, *, fname: str = ""):
        """Configure source from configuration file"""
        with open(fname, "r") as f:
            config = toml.load(f)

        v = _get_mand(config["source"], "vo")
        r = _get_opt(config["source"], "rs", RS_DEFAULT)
        lim = _get_opt(config, "limits", LIMITS_DEFAULT)
        return cls(name, vo=v, rs=r, limits=lim)

    def _get_inp_current(self, phase):
        return 0.0

    def _get_outp_voltage(self, phase):
        return self.params["vo"]

    def _solv_inp_curr(self, vi, vo, io, phase):
        """Calculate component input current from vi, vo and io"""
        return io

    def _solv_outp_volt(self, vi, ii, io, phase):
        """Calculate component output voltage from vi, ii and io"""
        return self.params["vo"] - self.params["rs"] * io

    def _solv_pwr_loss(self, vi, vo, ii, io, phase):
        """Calculate power and loss in component"""
        opwr = abs(vo * io)
        loss = self.params["rs"] * io * io
        ipwr = opwr + loss
        return ipwr, loss, _get_eff(ipwr, opwr)

    def _solv_get_warns(self, vi, vo, ii, io, phase):
        """Check limits"""
        return _get_warns(self.limits, {"ii": ii, "io": io})


class PLoad:
    """The Load component .

    Attributes
    ----------
    component_type : ComponentTypes.LOAD (enum)
        type of component
    """

    @property
    def component_type(self):
        """Defines the component type"""
        return ComponentTypes.LOAD

    @property
    def child_types(self):
        """The Load component cannot have childs"""
        return [None]

    def __init__(
        self,
        name: str,
        *,
        pwr: float,
        limits: dict = LIMITS_DEFAULT,
        pwrs: float = PWRS_DEFAULT,
        phase_loads: dict = {},
    ):
        """Set load power"""
        self.params = {}
        self.params["name"] = name
        self.params["pwr"] = abs(pwr)
        self.params["pwrs"] = abs(pwrs)
        self.params["phase_loads"] = phase_loads
        self.limits = limits

    @classmethod
    def from_file(cls, name: str, *, fname: str = ""):
        """Configure load from configuration file"""
        with open(fname, "r") as f:
            config = toml.load(f)

        p = _get_mand(config["pload"], "pwr")
        lim = _get_opt(config, "limits", LIMITS_DEFAULT)
        pwrs = _get_opt(config["pload"], "pwrs", PWRS_DEFAULT)
        pl = _get_opt(config["pload"], "phase_loads", {})
        return cls(name, pwr=p, limits=lim, pwrs=pwrs, phase_loads=pl)

    def _get_inp_current(self, phase):
        return 0.0

    def _get_outp_voltage(self, phase):
        return 0.0

    def _solv_inp_curr(self, vi, vo, io, phase):
        """Calculate component input current from vi, vo and io"""
        if vi == 0.0:
            return 0.0
        if phase == "" or not self.params["phase_loads"]:
            p = self.params["pwr"]
        elif phase not in self.params["phase_loads"]:
            p = self.params["pwrs"]
        else:
            p = self.params["phase_loads"][phase]

        return p / abs(vi)

    def _solv_outp_volt(self, vi, ii, io, phase):
        """Load output voltage is always 0"""
        return 0.0

    def _solv_pwr_loss(self, vi, vo, ii, io, phase):
        """Calculate power and loss in component"""
        return abs(vi * ii), 0.0, 100.0

    def _solv_get_warns(self, vi, vo, ii, io, phase):
        """Check limits"""
        if self.params["phase_loads"] and phase != "":
            if phase not in self.params["phase_loads"]:
                return ""
        return _get_warns(self.limits, {"vi": vi, "ii": ii})


class ILoad(PLoad):
    """The Load component .

    Attributes
    ----------
    component_type : ComponentTypes.LOAD (enum)
        type of component
    """

    def __init__(
        self,
        name: str,
        *,
        ii: float,
        limits: dict = LIMITS_DEFAULT,
        iis: float = IIS_DEFAULT,
        phase_loads: dict = {},
    ):
        """Set load current"""
        self.params = {}
        self.params["name"] = name
        self.params["ii"] = abs(ii)
        self.limits = limits
        self.params["iis"] = abs(iis)
        self.params["phase_loads"] = phase_loads

    @classmethod
    def from_file(cls, name: str, *, fname: str = ""):
        """Configure load from configuration file"""
        with open(fname, "r") as f:
            config = toml.load(f)

        i = _get_mand(config["iload"], "ii")
        lim = _get_opt(config, "limits", LIMITS_DEFAULT)
        iis = _get_opt(config["iload"], "iis", IIS_DEFAULT)
        pl = _get_opt(config["iload"], "phase_loads", {})
        return cls(name, ii=i, limits=lim, iis=iis, phase_loads=pl)

    def _get_inp_current(self, phase):
        return self.params["ii"]

    def _solv_inp_curr(self, vi, vo, io, phase):
        if vi == 0.0:
            return 0.0
        if phase == "" or not self.params["phase_loads"]:
            i = self.params["ii"]
        elif phase not in self.params["phase_loads"]:
            i = self.params["iis"]
        else:
            i = self.params["phase_loads"][phase]

        return abs(i)


class RLoad(PLoad):
    """The Load component .

    Attributes
    ----------
    component_type : ComponentTypes.LOAD (enum)
        type of component
    """

    def __init__(
        self,
        name: str,
        *,
        rs: float,
        limits: dict = LIMITS_DEFAULT,
        phase_loads: dict = {},
    ):
        """Set load current"""
        self.params = {}
        self.params["name"] = name
        if abs(rs) == 0.0:
            raise ValueError("Error: rs must be > 0!")
        self.params["rs"] = abs(rs)
        self.params["phase_loads"] = phase_loads
        self.limits = limits

    @classmethod
    def from_file(cls, name: str, *, fname: str = ""):
        """Configure load from configuration file"""
        with open(fname, "r") as f:
            config = toml.load(f)

        r = _get_mand(config["rload"], "rs")
        lim = _get_opt(config, "limits", LIMITS_DEFAULT)
        pl = _get_opt(config["rload"], "phase_loads", {})
        return cls(name, rs=r, limits=lim, phase_loads=pl)

    def _solv_inp_curr(self, vi, vo, io, phase):
        r = self.params["rs"]
        if phase == "" or not self.params["phase_loads"]:
            pass
        elif phase not in self.params["phase_loads"]:
            pass
        else:
            r = self.params["phase_loads"][phase]
        return abs(vi) / r


class Loss:
    """The Loss component

    Attributes
    ----------
    component_type : ComponentTypes.LOSS (enum)
        type of component

    """

    @property
    def component_type(self):
        """Defines the component type"""
        return ComponentTypes.LOSS

    @property
    def child_types(self):
        """Defines allowable child component types"""
        et = list(ComponentTypes)
        et.remove(ComponentTypes.SOURCE)
        return et

    def __init__(
        self,
        name,
        *,
        rs: float = RS_DEFAULT,
        vdrop: float = VDROP_DEFAULT,
        limits: dict = LIMITS_DEFAULT,
    ):
        """Set series resistance and/or vdrop"""
        self.params = {}
        self.params["name"] = name
        self.params["rs"] = abs(rs)
        self.params["vdrop"] = abs(vdrop)
        self.limits = limits

    @classmethod
    def from_file(cls, name: str, *, fname: str = ""):
        """Configure source from configuration file"""
        with open(fname, "r") as f:
            config = toml.load(f)

        r = _get_mand(config["loss"], "rs")
        vd = _get_mand(config["loss"], "vdrop")
        lim = _get_opt(config, "limits", LIMITS_DEFAULT)
        return cls(name, rs=r, vdrop=vd, limits=lim)

    def _get_inp_current(self, phase):
        return 0.0

    def _get_outp_voltage(self, phase):
        return 0.0

    def _solv_inp_curr(self, vi, vo, io, phase):
        """Calculate component input current from vi, vo and io"""
        if abs(vi) == 0.0:
            return 0.0
        return io  # TODO: iq?

    def _solv_outp_volt(self, vi, ii, io, phase):
        """Calculate component output voltage from vi, ii and io"""
        if abs(vi) == 0.0:
            return 0.0
        if vi >= 0.0:
            return vi - self.params["rs"] * io - self.params["vdrop"]
        else:
            return vi + self.params["rs"] * io + self.params["vdrop"]

    def _solv_pwr_loss(self, vi, vo, ii, io, phase):
        """Calculate power and loss in component"""
        loss = abs(self.params["rs"] * ii * ii + self.params["vdrop"] * ii)
        ipwr = abs(vi * ii)
        opwr = abs(vo * io)
        return 0.0, loss, _get_eff(ipwr, opwr, 0.0)

    def _solv_get_warns(self, vi, vo, ii, io, phase):
        """Check limits"""
        return _get_warns(self.limits, {"vi": vi, "vo": vo, "ii": ii, "io": io})


class Converter:
    """The Converter component

    Attributes
    ----------
    component_type : ComponentTypes.LOSS (enum)
        type of component
    """

    @property
    def component_type(self):
        """Defines the component type"""
        return ComponentTypes.CONVERTER

    @property
    def child_types(self):
        """Defines allowable child component types"""
        et = list(ComponentTypes)
        et.remove(ComponentTypes.SOURCE)
        return et

    def __init__(
        self,
        name: str,
        *,
        vo: float,
        eff: float,
        iq: float = IQ_DEFAULT,
        limits: dict = LIMITS_DEFAULT,
        iis: float = IIS_DEFAULT,
        active_phases: list = [],
    ):
        """Set converter parameters"""
        self.params = {}
        self.params["name"] = name
        self.params["vo"] = vo
        if not (eff > 0.0):
            raise ValueError("Efficiency must be > 0.0")
        if not (eff < 1.0):
            raise ValueError("Efficiency must be < 1.0")
        self.params["eff"] = eff
        self.params["iq"] = abs(iq)
        self.params["iis"] = abs(iis)
        self.params["active_phases"] = active_phases
        self.limits = limits

    @classmethod
    def from_file(cls, name: str, *, fname: str = ""):
        """Configure source from configuration file"""
        with open(fname, "r") as f:
            config = toml.load(f)

        v = _get_mand(config["converter"], "vo")
        e = _get_mand(config["converter"], "eff")
        iq = _get_opt(config["converter"], "iq", IQ_DEFAULT)
        lim = _get_opt(config, "limits", LIMITS_DEFAULT)
        iis = _get_opt(config["converter"], "iis", IIS_DEFAULT)
        ap = _get_opt(config["converter"], "active_phases", [])
        return cls(name, vo=v, eff=e, iq=iq, limits=lim, iis=iis, active_phases=ap)

    def _get_inp_current(self, phase):
        i = self.params["iq"]
        if phase == "" or not self.params["active_phases"]:
            pass
        elif phase not in self.params["active_phases"]:
            i = self.params["iis"]
        return i

    def _get_outp_voltage(self, phase):
        v = self.params["vo"]
        if phase == "" or not self.params["active_phases"]:
            pass
        elif phase not in self.params["active_phases"]:
            v = 0.0
        return v

    def _solv_inp_curr(self, vi, vo, io, phase):
        """Calculate component input current from vi, vo and io"""
        if abs(vi) == 0.0:
            return 0.0
        ve = self.params["eff"] * vi
        if phase == "" or not self.params["active_phases"]:
            pass
        elif phase not in self.params["active_phases"]:
            return self.params["iis"]
        if ve > 0.0:
            return self.params["iq"] + abs(vo * io / ve)
        return 0.0

    def _solv_outp_volt(self, vi, ii, io, phase):
        """Calculate component output voltage from vi, ii and io"""
        v = self.params["vo"]
        if phase == "" or not self.params["active_phases"]:
            pass
        elif phase not in self.params["active_phases"]:
            v = 0.0
        return v

    def _solv_pwr_loss(self, vi, vo, ii, io, phase):
        """Calculate power and loss in component"""
        loss = abs(
            self.params["iq"] * vi
            + (ii - self.params["iq"]) * vi * (1.0 - self.params["eff"])
        )
        pwr = abs(vi * ii)
        if phase == "" or not self.params["active_phases"]:
            pass
        elif phase not in self.params["active_phases"]:
            loss = abs(self.params["iis"] * vi)
            pwr = 0.0
        return 0.0, loss, _get_eff(pwr, pwr - loss, 0.0)

    def _solv_get_warns(self, vi, vo, ii, io, phase):
        """Check limits"""
        return _get_warns(self.limits, {"vi": vi, "vo": vo, "ii": ii, "io": io})


class LinReg:
    """The Linear regulator component

    Attributes
    ----------
    component_type : ComponentTypes.LINREG (enum)
        type of component
    """

    @property
    def component_type(self):
        """Defines the component type"""
        return ComponentTypes.LINREG

    @property
    def child_types(self):
        """Defines allowable child component types"""
        et = list(ComponentTypes)
        et.remove(ComponentTypes.SOURCE)
        return et

    def __init__(
        self,
        name,
        *,
        vo: float,
        vdrop: float = VDROP_DEFAULT,
        iq: float = IQ_DEFAULT,
        limits: dict = LIMITS_DEFAULT,
        iis: float = IIS_DEFAULT,
        active_phases: list = [],
    ):
        """Set linear regulator parameters"""
        self.params = {}
        self.params["name"] = name
        self.params["vo"] = vo
        if not (abs(vdrop) < abs(vo)):
            raise ValueError("Voltage drop must be < vo")
        self.params["vdrop"] = abs(vdrop)
        self.params["iq"] = abs(iq)
        self.params["iis"] = abs(iis)
        self.params["active_phases"] = active_phases
        self.limits = limits

    @classmethod
    def from_file(cls, name: str, *, fname: str = ""):
        """Configure source from configuration file"""
        with open(fname, "r") as f:
            config = toml.load(f)

        v = _get_mand(config["linreg"], "vo")
        vd = _get_opt(config["linreg"], "vdrop", VDROP_DEFAULT)
        iq = _get_opt(config["linreg"], "iq", IQ_DEFAULT)
        lim = _get_opt(config, "limits", LIMITS_DEFAULT)
        iis = _get_opt(config["linreg"], "iis", IIS_DEFAULT)
        ap = _get_opt(config["linreg"], "active_phases", [])
        return cls(name, vo=v, vdrop=vd, iq=iq, limits=lim, iis=iis, active_phases=ap)

    def _get_inp_current(self, phase):
        i = self.params["iq"]
        if phase == "" or not self.params["active_phases"]:
            pass
        elif phase not in self.params["active_phases"]:
            i = self.params["iis"]
        return i

    def _get_outp_voltage(self, phase):
        v = self.params["vo"]
        if phase == "" or not self.params["active_phases"]:
            pass
        elif phase not in self.params["active_phases"]:
            v = 0.0
        return v

    def _solv_inp_curr(self, vi, vo, io, phase):
        """Calculate component input current from vi, vo and io"""
        if abs(vi) == 0.0:
            return 0.0
        i = io + self.params["iq"]
        if phase == "" or not self.params["active_phases"]:
            pass
        elif phase not in self.params["active_phases"]:
            i = self.params["iis"]
        return i

    def _solv_outp_volt(self, vi, ii, io, phase):
        """Calculate component output voltage from vi, ii and io"""
        v = min(abs(self.params["vo"]), max(abs(vi) - self.params["vdrop"], 0.0))
        if phase == "" or not self.params["active_phases"]:
            pass
        elif phase not in self.params["active_phases"]:
            v = 0.0
        if self.params["vo"] >= 0.0:
            return v
        return -v

    def _solv_pwr_loss(self, vi, vo, ii, io, phase):
        """Calculate power and loss in component"""
        loss = (abs(vi) - abs(vo)) * io + abs(vi) * self.params["iq"]
        pwr = abs(vi * ii)
        if phase == "" or not self.params["active_phases"]:
            pass
        elif phase not in self.params["active_phases"]:
            loss = abs(self.params["iis"] * vi)
            pwr = 0.0
        return 0.0, loss, _get_eff(pwr, pwr - loss, 0.0)

    def _solv_get_warns(self, vi, vo, ii, io, phase):
        """Check limits"""
        return _get_warns(self.limits, {"vi": vi, "vo": vo, "ii": ii, "io": io})
