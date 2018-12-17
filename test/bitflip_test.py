import torchFI
import numpy as np
from bitstring import BitArray
from util.log import *

def test_flipFloat(val):
    # fix random seed
    np.random.seed(4)

    faultValue = BitArray(float=val, length=32)
    bit = np.random.randint(0, faultValue.len)
    faultValue.invert(bit)
    tret = faultValue.float

    fret = torchFI.flipFloat(val, bit)
    
    return tret == fret


def main():
    logTestStart("testing flipFloat")
    if test_flipFloat(4):
        logTestPass()
    else:
        logTestError()


if __name__ == '__main__':
    main()