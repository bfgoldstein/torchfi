import sys
import numpy as np
from bitstring import BitArray

from util.log import *


def flipFloat(val, bit=None, log=False):
    # Cast float to BitArray and flip (invert) random bit 0-31

    faultValue = BitArray(float=val, length=32)
    if bit == None:
        bit = np.random.randint(0, faultValue.len)
    faultValue.invert(bit)

    # TODO: apply log functions
    if log:
        logInjectionBit("\tFlipping bit ", bit)
        logInjectionVal("\tOriginal: ", float(val), " Corrupted: ", faultValue.float)

    return faultValue.float


def bitFlip(value):
    vType = value.dtype

    if vType == "Float":
        return flipFloat(value)
    else:
        # TODO: apply log functions
        print("\t bitFlipt for " + str(vType) + " was not implemented \n")
        sys.exit()
