import sys

sys.path.append("../../src")


from sysloss.element import *


# import sysloss


def test_classes():
    assert issubclass(Source, ElementInterface), "subclass Source"
    assert issubclass(PLoad, ElementInterface), "subclass ILoad"
    assert issubclass(PLoad, ElementInterface), "subclass PLoad"
    assert issubclass(Loss, ElementInterface), "subclass Loss"
    assert issubclass(Converter, ElementInterface), "subclass Converter"
    assert issubclass(LinReg, ElementInterface), "subclass LinReg"
