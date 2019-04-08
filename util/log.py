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




class Record(object):

    def __init__(self, model, batch_size, fiLayer, fiFeatures, fiWeights, quant_bfeats, quant_bwts, quant_baccum, 
                injection=None, quantization=None):
        """Record constructor
            
        Args:
            model (String): Name of the model in torchvision
            
            fiFeatures (bool): If True, faults can be applied on input features

            fiWeights (bool): If True, faults can be applied on weights

                fiFeatures and fiWeights == False: Faults are applied randomly on input features and weighs
            
            fiFeatures (1d ndarray): array of integers with bit flipped position of each batch
            
            fiBit (1d ndarray): array of integers with bit flipped position of each batch
            
            originalValue (1d ndarray): array of floats with value before injection of each batch
            
            fiValue (1d ndarray): array of floats with value after injection of each batch
            
            fiLocation (tuple of ints): typle indicating location of fault injection
                e.g.: (filter, row, column)

            quantization (tuple with # bits for (feat, wts, acc)): If not None, quantization 
                was applied with #bits for features, weights and accumulator

            quant_bfeats (integer): number of bits for input features during quantization
            
            quant_bwts (integer): number of bits for weights during quantization
            
            quant_baccum (integer): number of bits for accumulators during quantization
            
            scores (2d tensor): array of scores (only top 5 values) of each batch

            predictions (2d tensor): array of labels predicted (only top 5 labels) of each batch
            
            targets (2d tensor): array of targets list of each batch. Each target list contains 
                an integer and float that represents correct label and obtained accuracy respectively.
        """

        self.model = model
        self.batch_size = batch_size

        self.injection = injection
        self.fiLayer = fiLayer
        self.fiFeatures = fiFeatures
        self.fiWeights = fiWeights
        # self.fiBit = []
        # self.originalValue = []
        # self.fiValue = []
        # self.fiLocation = [] 

        self.quantization = quantization
        self.quant_bfeats = quant_bfeats
        self.quant_bwts = quant_bwts
        self.quant_baccum = quant_baccum

        self.scores = []
        self.predictions = []
        self.targets = []
        self.acc1 = 0
        self.acc5 = 0

    def addScores(self, tensors):
        self.scores.append(tensors.cpu())
    
    def addPredictions(self, tensors):
        self.predictions.append(tensors.cpu())

    def addTargets(self, arr):
        for val in arr:
            self.targets.append(val)

    def addFiBit(self, bit):
        pass
        # self.fiBit.append(bit)

    def addOriginalValue(self, value):
        pass
        # self.originalValue.append(value)

    def addFiValue(self, value):
        pass
        # self.fiValue.append(value)

    def addFiLocation(self, loc):
        pass
        # self.fiLocation.append(loc)

    def setAccuracies(self, acc1, acc5):
        self.acc1 = acc1
        self.acc5 = acc5