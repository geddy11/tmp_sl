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


from sysloss.element import *
import pytest


def test_classes():
    """Check informal interface on elements"""
    assert issubclass(Source, ElementInterface), "subclass Source"
    assert issubclass(PLoad, ElementInterface), "subclass ILoad"
    assert issubclass(PLoad, ElementInterface), "subclass PLoad"
    assert issubclass(Loss, ElementInterface), "subclass Loss"
    assert issubclass(Converter, ElementInterface), "subclass Converter"
    assert issubclass(LinReg, ElementInterface), "subclass LinReg"


def test_source():
    """Check Source element"""
    sa = Source("Battery 3V", vo=3.0, rs=7e-3)
    assert sa.element_type == ElementTypes.SOURCE, "Source element type"
    assert ElementTypes.SOURCE not in list(sa.child_types), "Source child types"
    sb = Source.from_file("Battery 3V", fname="tests/data/source.toml")
    assert sa.params == sb.params, "Source parameters from file"
    assert sa.limits == sb.limits, "Source limits from file"
    assert isinstance(sa, ElementInterface), "instance Source"


def test_pload():
    """Check PLoad element"""
    pa = PLoad("Load 1", pwr=27e-3)
    assert pa.element_type == ElementTypes.LOAD, "PLoad element type"
    assert list(pa.child_types) == [None], "PLoad child types"
    pb = PLoad.from_file("Load 1", fname="tests/data/pload.toml")
    assert pa.params == pb.params, "PLoad parameters from file"
    assert pa.limits == pb.limits, "PLoad limits from file"
    assert isinstance(pa, ElementInterface), "instance PLoad"


def test_iload():
    """Check ILoad element"""
    ia = ILoad("Load 1", ii=15e-3)
    assert ia.element_type == ElementTypes.LOAD, "ILoad element type"
    assert list(ia.child_types) == [None], "ILoad child types"
    ib = ILoad.from_file("Load 1", fname="tests/data/iload.toml")
    assert ia.params == ib.params, "ILoad parameters from file"
    assert ia.limits == ib.limits, "ILoad limits from file"
    assert isinstance(ia, ElementInterface), "instance ILoad"


def test_loss():
    """Check Loss element"""
    la = Loss("Loss 1", rs=30e-3, vdrop=0.0, limits=LIMITS_DEFAULT)
    assert la.element_type == ElementTypes.LOSS, "Loss element type"
    assert ElementTypes.SOURCE not in list(la.child_types), "Loss child types"
    lb = Loss.from_file("Loss 1", fname="tests/data/loss.toml")
    assert la.params == lb.params, "Loss parameters from file"
    assert la.limits == lb.limits, "Loss limits from file"
    assert isinstance(la, ElementInterface), "instance Loss"


def test_converter():
    """Check Converter element"""
    ca = Converter("Conv 1", vo=5.0, eff=0.87)
    assert ca.element_type == ElementTypes.CONVERTER, "Converter element type"
    assert ElementTypes.SOURCE not in list(ca.child_types), "Converter child types"
    with pytest.raises(ValueError):
        cb = Converter("Conv 1", vo=5.0, eff=1.01)
    with pytest.raises(ValueError):
        cb = Converter("Conv 1", vo=5.0, eff=-0.1)
    with pytest.raises(ValueError):
        cb = Converter("Conv 1", vo=5.0, eff=0.0)
    cb = Converter.from_file("Conv 1", fname="tests/data/converter.toml")
    assert ca.params == cb.params, "Converter parameters from file"
    assert ca.limits == cb.limits, "Converter limits from file"
    assert isinstance(ca, ElementInterface), "instance Converter"


def test_linreg():
    """Check LinReg element"""
    la = LinReg("LDO 1", vo=2.5, vdrop=0.3, iq=0.0)
    assert la.element_type == ElementTypes.LINREG, "LinReg element type"
    assert ElementTypes.SOURCE not in list(la.child_types), "LinReg child types"
    with pytest.raises(ValueError):
        lb = LinReg("LDO 2", vo=1.8, vdrop=2.0)
    with pytest.raises(KeyError):
        lb = LinReg.from_file("LDO 1", fname="tests/data/linreg_bad.toml")
    lb = LinReg.from_file("LDO 1", fname="tests/data/linreg.toml")
    assert la.params == lb.params, "LinReg parameters from file"
    assert la.limits == lb.limits, "LinReg limits from file"
    assert isinstance(la, ElementInterface), "instance LinReg"
