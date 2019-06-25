import numpy as np


from .format_display import *

logMsg = "Log ===>"

def logWarning(message):
    print(bcolors.HEADER + logMsg + bcolors.ENDC + bcolors.WARNING + message + bcolors.ENDC)


def logError(message):
    print(bcolors.HEADER + logMsg + bcolors.ENDC + bcolors.WARNING + message + bcolors.ENDC)


##
##  Config Loging
##

configLogMsg = "Config Log ===>\t "

def logConfig(argName, argValue):
    print(configLogMsg + argName + "\t" + argValue)


##
##  Injection Loging
##

injectionLogMsg = "Injection Log ===>\t "

def logInjectionWarning(message):
    print(bcolors.HEADER + injectionLogMsg + bcolors.ENDC + bcolors.WARNING + message + bcolors.ENDC)

def logInjectionBit(message, bit):
    print(bcolors.HEADER + injectionLogMsg + bcolors.ENDC + bcolors.WARNING + message + bcolors.ENDC
            + bcolors.FAIL + str(bit) + bcolors.ENDC)

def logInjectionNode(message, index):
    print(bcolors.HEADER + injectionLogMsg + bcolors.ENDC + bcolors.WARNING + message + bcolors.ENDC
            + bcolors.FAIL + str(index) + bcolors.ENDC)

def logInjectionVal(message_orig, orig_val, message_corr, corr_val):
    print(bcolors.HEADER + injectionLogMsg + bcolors.ENDC 
            + bcolors.OKGREEN + message_orig + bcolors.ENDC
            + bcolors.WARNING + str(orig_val) + bcolors.ENDC 
            + bcolors.OKBLUE + message_corr + bcolors.ENDC
            + bcolors.WARNING + str(corr_val) + bcolors.ENDC)


##
##  Test Loging
##

def logTestStart(testName):
    print(bcolors.WARNING + testName + bcolors.ENDC),

def logTestPass():
    print(bcolors.OKGREEN + "    passed!" + bcolors.ENDC)

def logTestError():
    print(bcolors.FAIL + "    error!" + bcolors.ENDC)