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


@unique
class ElementTypes(Enum):
    """Element types"""

    SOURCE = 1
    LOAD = 2
    LOSS = 3
    CONVERTER = 4
    LINREG = 5


IMAX_DEFAULT = 1.0e6
IQ_DEFAULT = 0.0
RS_DEFAULT = 0.0
VDROP_DEFAULT = 0.0


def __get_opt(params, key, default):
    """Get optional parameter from dict"""
    if key in params:
        return params[key]
    return default


def __get_mand(params, key):
    """Get mandatory parameter from dict"""
    if key in params:
        return params[key]
    raise KeyError("Parameter dict is missing entry for '{}'".format(key))


class ElementMeta(type):
    """An element metaclass that will be used for element class creation."""

    def __instancecheck__(cls, instance):
        return cls.__subclasscheck__(type(instance))

    def __subclasscheck__(cls, subclass):
        return (
            hasattr(subclass, "element_type")
            and hasattr(subclass, "child_types")
            and hasattr(subclass, "from_file")
            and callable(subclass.from_file)
        )


class ElementInterface(metaclass=ElementMeta):
    """This interface is used for concrete element classes to inherit from.
    There is no need to define the ElementMeta methods of any class
    as they are implicitly made available via .__subclasscheck__().
    """

    pass


class Source:
    """The Source element must the root of a system.
    A system can only have one source.

    Attributes
    ----------
    element_type : ElementTypes.SOURCE (enum)
        type of element
    """

    @property
    def element_type(self):
        """Defines the element type"""
        return ElementTypes.SOURCE

    @property
    def child_types(self):
        """Defines allowable child element types"""
        et = list(ElementTypes)
        et.remove(ElementTypes.SOURCE)
        return et

    def __init__(
        self,
        name: str,
        *,
        vo: float,
        rs: float = RS_DEFAULT,
        imax: float = IMAX_DEFAULT
    ):
        """Set source name, voltage, internal resistance and max current"""
        self.params = {}
        self.params["name"] = name
        self.params["vo"] = vo
        self.params["imax"] = abs(imax)
        self.params["rs"] = abs(rs)

    @classmethod
    def from_file(cls, name: str, *, fname: str = ""):
        """Configure source from configuration file"""
        with open(fname, "r") as f:
            config = toml.load(f)

        v = __get_mand(config, "vo")
        mc = __get_opt(config, "imax", IMAX_DEFAULT)
        r = __get_opt(config, "rs", RS_DEFAULT)
        return cls(name, vo=v, rs=r, imax=mc)

    def _get_inp_current(self):
        return 0.0

    def _get_outp_voltage(self):
        return self.params["vo"]

    def _solv_inp_curr(self, vi, vo, io):
        """Calculate element input current from vi, vo and io"""
        return io

    def _solv_outp_volt(self, vi, ii, io):
        """Calculate element output voltage from vi, ii and io"""
        return self.params["vo"] - self.params["rs"] * io

    def _solv_pwr_loss(self, vi, vo, ii, io):
        """Calculate power and loss in element"""
        pwr = abs(vo * io)
        loss = abs(self.params["rs"] * io * io)
        return pwr, loss, (pwr - loss) / pwr

    def _solv_get_warns(self, vi, vo, ii, io):
        """Check limits"""
        if io > self.params["imax"]:
            return "io: {:.2f} > {:.2f}".format(io, self.params["imax"])

        return ""


class PLoad:
    """The Load element .

    Attributes
    ----------
    element_type : ElementTypes.LOAD (enum)
        type of element
    """

    @property
    def element_type(self):
        """Defines the element type"""
        return ElementTypes.LOAD

    @property
    def child_types(self):
        """The Load element cannot have childs"""
        return {None}

    def __init__(self, name: str, *, pwr: float):
        """Set load power"""
        self.params = {}
        self.params["name"] = name
        self.params["pwr"] = abs(pwr)

    @classmethod
    def from_file(cls, name: str, *, fname: str = ""):
        """Configure load from configuration file"""
        with open(fname, "r") as f:
            config = toml.load(f)

        p = __get_mand(config, "pwr")
        return cls(name, pwr=p)

    def _get_inp_current(self):
        return 0.0

    def _get_outp_voltage(self):
        return 0.0

    def _solv_inp_curr(self, vi, vo, io):
        """Calculate element input current from vi, vo and io"""
        if vi != 0.0:
            return self.params["pwr"] / abs(vi)
        return 0.0

    def _solv_outp_volt(self, vi, ii, io):
        """Load output voltage is always 0"""
        return 0.0

    def _solv_pwr_loss(self, vi, vo, ii, io):
        """Calculate power and loss in element"""
        return abs(vi * ii), 0.0, 100.0

    def _solv_get_warns(self, vi, vo, ii, io):
        """Check limits"""

        return ""


class ILoad(PLoad):
    """The Load element .

    Attributes
    ----------
    element_type : ElementTypes.LOAD (enum)
        type of element
    """

    def __init__(self, name: str, *, ii: float):
        """Set load current"""
        self.params = {}
        self.params["name"] = name
        self.params["ii"] = abs(ii)

    @classmethod
    def from_file(cls, name: str, *, fname: str = ""):
        """Configure load from configuration file"""
        with open(fname, "r") as f:
            config = toml.load(f)

        i = __get_mand(config, "ii")
        return cls(name, ii=i)

    def _get_inp_current(self):
        return self.params["ii"]

    def _solv_inp_curr(self, vi, vo, io):
        return self.params["ii"]


class Loss:
    """The Loss element

    Attributes
    ----------
    element_type : ElementTypes.LOSS (enum)
        type of element

    """

    @property
    def element_type(self):
        """Defines the element type"""
        return ElementTypes.LOSS

    @property
    def child_types(self):
        """Defines allowable child element types"""
        et = list(ElementTypes)
        et.remove(ElementTypes.SOURCE)
        return et

    def __init__(self, name, *, rs: float = RS_DEFAULT, vdrop: float = VDROP_DEFAULT):
        """Set series resistance and/or vdrop"""
        self.params = {}
        self.params["name"] = name
        self.params["rs"] = abs(rs)
        self.params["vdrop"] = abs(vdrop)

    @classmethod
    def from_file(cls, name: str, *, fname: str = ""):
        """Configure source from configuration file"""
        with open(fname, "r") as f:
            config = toml.load(f)

        vd = __get_opt(config, "vdrop", VDROP_DEFAULT)
        r = __get_opt(config, "rs", RS_DEFAULT)
        return cls(name, rs=r, vdrop=vd)

    def _get_inp_current(self):
        return 0.0

    def _get_outp_voltage(self):
        return 0.0

    def _solv_inp_curr(self, vi, vo, io):
        """Calculate element input current from vi, vo and io"""
        return io  # TODO: iq?

    def _solv_outp_volt(self, vi, ii, io):
        """Calculate element output voltage from vi, ii and io"""
        if vi >= 0.0:
            return vi - self.params["rs"] * io - self.params["vdrop"]
        else:
            return vi + self.params["rs"] * io + self.params["vdrop"]

    def _solv_pwr_loss(self, vi, vo, ii, io):
        """Calculate power and loss in element"""
        loss = abs(self.params["rs"] * ii * ii + self.params["vdrop"] * ii)
        pwr = abs(vi * ii)
        if pwr > 0.0:
            return 0.0, loss, (pwr - loss) / pwr
        return 0.0, loss, 0.0

    def _solv_get_warns(self, vi, vo, ii, io):
        """Check limits"""

        return ""


class Converter:
    """The Converter element

    Attributes
    ----------
    element_type : ElementTypes.LOSS (enum)
        type of element
    """

    @property
    def element_type(self):
        """Defines the element type"""
        return ElementTypes.CONVERTER

    @property
    def child_types(self):
        """Defines allowable child element types"""
        et = list(ElementTypes)
        et.remove(ElementTypes.SOURCE)
        return et

    def __init__(
        self,
        name: str,
        *,
        vo: float,
        eff: float,
        iq: float = IQ_DEFAULT,
        imax: float = IMAX_DEFAULT
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
        self.params["imax"] = abs(imax)
        self.params["iq"] = abs(iq)

    @classmethod
    def from_file(cls, name: str, *, fname: str = ""):
        """Configure source from configuration file"""
        with open(fname, "r") as f:
            config = toml.load(f)

        v = __get_mand(config, "vo")
        e = __get_mand(config, "eff")
        iq = __get_opt(config, "iq", IQ_DEFAULT)
        mc = __get_opt(config, "imax", IMAX_DEFAULT)
        return cls(name, vo=v, eff=e, iq=iq, imax=mc)

    def _get_inp_current(self):
        return self.params["iq"]

    def _get_outp_voltage(self):
        return self.params["vo"]

    def _solv_inp_curr(self, vi, vo, io):
        """Calculate element input current from vi, vo and io"""
        ve = self.params["eff"] * vi
        if ve > 0.0:
            return self.params["iq"] + abs(vo * io / ve)
        return 0.0

    def _solv_outp_volt(self, vi, ii, io):
        """Calculate element output voltage from vi, ii and io"""
        return self.params["vo"]

    def _solv_pwr_loss(self, vi, vo, ii, io):
        """Calculate power and loss in element"""
        loss = abs(
            self.params["iq"] * vi
            + (ii - self.params["iq"]) * vi * (1.0 - self.params["eff"])
        )
        pwr = abs(vi * ii)
        if pwr > 0.0:
            return 0.0, loss, (pwr - loss) / pwr
        return 0.0, loss, 0.0

    def _solv_get_warns(self, vi, vo, ii, io):
        """Check limits"""

        return ""


class LinReg:
    """The Linear regulator element

    Attributes
    ----------
    element_type : ElementTypes.LINREG (enum)
        type of element
    """

    @property
    def element_type(self):
        """Defines the element type"""
        return ElementTypes.LINREG

    @property
    def child_types(self):
        """Defines allowable child element types"""
        et = list(ElementTypes)
        et.remove(ElementTypes.SOURCE)
        return et

    def __init__(
        self,
        name,
        *,
        vo: float,
        vdrop: float = VDROP_DEFAULT,
        iq: float = IQ_DEFAULT,
        imax: float = IMAX_DEFAULT
    ):
        """Set linear regulator parameters"""
        self.params = {}
        self.params["name"] = name
        self.params["vo"] = vo
        if not (abs(vdrop) < abs(vo)):
            raise ValueError("Voltage drop must be < vo")
        self.params["vdrop"] = abs(vdrop)
        self.params["iq"] = abs(iq)
        self.params["imax"] = abs(imax)

    @classmethod
    def from_file(cls, name: str, *, fname: str = ""):
        """Configure source from configuration file"""
        with open(fname, "r") as f:
            config = toml.load(f)

        v = __get_mand(config, "vo")
        vd = __get_opt(condfig, "vdrop", VDROP_DEFAULT)
        iq = __get_opt(config, "iq", IQ_DEFAULT)
        mc = __get_opt(config, "imax", IMAX_DEFAULT)
        return cls(name, vo=v, vdrop=vd, iq=iq, imax=mc)

    def _get_inp_current(self):
        return self.params["iq"]

    def _get_outp_voltage(self):
        return self.params["vo"]

    def _solv_inp_curr(self, vi, vo, io):
        """Calculate element input current from vi, vo and io"""
        return io + self.params["iq"]

    def _solv_outp_volt(self, vi, ii, io):
        """Calculate element output voltage from vi, ii and io"""
        return self.params["vo"]

    def _solv_pwr_loss(self, vi, vo, ii, io):
        """Calculate power and loss in element"""
        loss = abs((vi - vo) * io + vi * self.params["iq"])
        pwr = abs(vi * ii)
        if pwr >= 0.0:
            return 0.0, loss, (pwr - loss) / pwr
        return 0.0, loss, 0.0

    def _solv_get_warns(self, vi, vo, ii, io):
        """Check limits"""

        return ""
