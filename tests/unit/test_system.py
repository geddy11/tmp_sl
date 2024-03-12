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

import pytest
import numpy as np
import rich
from sysloss.system import System
from sysloss.elements import *


def test_case1():
    """Check system consisting of all element types"""
    case1 = System("Case1 system", Source("3V coin", vo=3, rs=13e-3))
    case1.add_element(
        "3V coin", element=Converter("1.8V buck", vo=1.8, eff=0.87, iq=12e-6)
    )
    case1.add_element("1.8V buck", element=PLoad("MCU", pwr=27e-3))
    case1.add_element(
        "3V coin", element=Converter("5V boost", vo=5, eff=0.91, iq=42e-6)
    )
    case1.add_element("5V boost", element=ILoad("Sensor", ii=15e-3))
    case1.add_element(parent="5V boost", element=Loss("RC filter", rs=33.0))
    case1.add_element(
        "RC filter", element=LinReg("LDO 2.5V", vo=2.5, vdrop=0.27, iq=150e-6)
    )
    case1.add_element("LDO 2.5V", element=PLoad("ADC", pwr=15e-3))
    case1.add_element("5V boost", element=RLoad("Res divider", rs=200e3))
    df = case1.solve(maxiter=1)
    assert df == None, "Case1 aborted on maxiter"
    df = case1.solve(quiet=False)
    assert len(df) == 10, "Case1 result row count"
    assert np.allclose(
        df[df["Element"] == "System total"]["Efficiency (%)"][9],
        0.793669,
        rtol=1e-6,
    ), "Case1 efficiency"
    case1.save("tests/unit/case1.json")
    dfp = case1.params(limits=True)
    assert len(dfp) == 9, "Case1 parameters row count"
    t = case1.tree()
    assert type(t) == rich.tree.Tree, "Case1 tree output"
    with pytest.raises(ValueError):
        case1.tree("Dummy")

    # reload system from json
    case1b = System.from_file("tests/unit/case1.json")
    df2 = case1b.solve()
    assert len(df2) == 10, "Case1b result row count"

    assert np.allclose(
        df2[df2["Element"] == "System total"]["Efficiency (%)"][9],
        df[df["Element"] == "System total"]["Efficiency (%)"][9],
        rtol=1e-6,
    ), "Case1 vs case1b efficiency"

    assert np.allclose(
        df2[df2["Element"] == "System total"]["Power (W)"][9],
        df[df["Element"] == "System total"]["Power (W)"][9],
        rtol=1e-6,
    ), "Case1 vs case1b power"


def test_case2():
    """Check system consisting of only Source"""
    case2 = System("Case2 system", Source("12V input", vo=12.0))
    df = case2.solve()
    assert len(df) == 2, "Case2 result row count"
    assert (
        df[df["Element"] == "System total"]["Efficiency (%)"][1] == 0.0
    ), "Case2 efficiency"


def test_case3():
    """Check system with negative output converter"""
    case3 = System("Case3 system", Source("5V USB", vo=5.0))
    case3.add_element(
        "5V USB",
        element=Converter("-12V inverter", vo=-12.0, eff=0.88),
    )
    case3.add_element("-12V inverter", element=Loss("Resistor", rs=25.0))
    df = case3.solve()
    assert len(df) == 4, "Case3 result row count"
    assert (
        df[df["Element"] == "System total"]["Efficiency (%)"][3] == 0.0
    ), "Case2 efficiency"


def test_case4():
    """Converter with zero input voltage"""
    case4 = System("Case4 system", Source("0V system", vo=0.0))
    case4.add_element("0V system", element=Converter("Buck", vo=3.3, eff=0.50))
    df = case4.solve()
    assert len(df) == 3, "Case4 result row count"


def test_case5():
    """LinReg with zero input voltage"""
    case5 = System("Case4 system", Source("0V system", vo=0.0))
    case5.add_element("0V system", element=LinReg("LDO", vo=-3.3))
    df = case5.solve()
    assert len(df) == 3, "Case5 result row count"


def test_case6():
    """Create new system with root as non-Source"""
    with pytest.raises(ValueError):
        case6 = System("Case6 system", PLoad("Load", pwr=1))


def test_case7():
    """Add element to non-existing element"""
    case7 = System("Case7 system", Source("10V system", vo=10.0))
    with pytest.raises(ValueError):
        case7.add_element("5V input", element=Converter("Buck", vo=2.5, eff=0.75))


def test_case8():
    """Add element with already used name"""
    case8 = System("Case8 system", Source("10V system", vo=10.0))
    case8.add_element("10V system", element=Converter("Buck", vo=2.5, eff=0.75))
    with pytest.raises(ValueError):
        case8.add_element("10V system", element=Converter("Buck", vo=2.5, eff=0.75))


def test_case9():
    """Try adding element of wrong type"""
    case9 = System("Case9 system", Source("10V system", vo=10.0))
    with pytest.raises(ValueError):
        case9.add_element("10V system", element=Source("5V", vo=5.0))


def test_case10():
    """Change element"""
    case10 = System("Case10 system", Source("24V system", vo=24.0, rs=12e-3))
    case10.add_element("24V system", element=Converter("Buck", vo=3.3, eff=0.80))
    case10.change_element("Buck", element=LinReg("LDO", vo=3.3))
    with pytest.raises(ValueError):
        case10.change_element("LDO", element=Source("5V", vo=5.0))


def test_case11():
    """Delete element"""
    case11 = System("Case11 system", Source("CR2032", vo=3.0))
    case11.add_element("CR2032", element=Converter("1.8V buck", vo=1.8, eff=0.87))
    case11.add_element("1.8V buck", element=PLoad("MCU", pwr=27e-3))
    case11.del_element("1.8V buck", del_childs=False)
    dfp = case11.params()
    assert len(dfp) == 2, "Case11 parameters row count"
    with pytest.raises(ValueError):
        case11.del_element("CR2032")
    with pytest.raises(ValueError):
        case11.del_element("Non-existent")
    case11.add_element("CR2032", element=Converter("1.8V buck", vo=1.8, eff=0.87))
    case11.add_element("1.8V buck", element=PLoad("MCU2", pwr=27e-3))
    case11.del_element("1.8V buck")
    dfp = case11.params()
    assert len(dfp) == 2, "Case11 parameters row count"
