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

    if log:
        logInjectionBit("\tFlipping bit ", bit)
        logInjectionVal("\tOriginal: ", float(val), " Corrupted: ", faultValue.float)

    return faultValue.float

def flipInt(val, size, bit=None, log=False):
    # Cast integer to BitArray and flip (invert) random bit 0-N
    val = int(val)

    faultValue = BitArray(int=val, length=size)
    if bit == None:
        bit = np.random.randint(0, faultValue.len)
    faultValue.invert(bit)

    if log:
        logInjectionBit("\tFlipping bit ", bit)
        logInjectionVal("\tOriginal: ", int(val), " Corrupted: ", faultValue.int)

    return faultValue.int


def bitFlip(value, size=8, bit=None, log=False, quantized=False):
    if quantized:
        return flipInt(value, size, bit, log)
    else:
        return flipFloat(value, bit, log)