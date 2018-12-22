from torchFI import *
import torchvision.models as models


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def test_instFI(model, layer):
    fi = FI(model, layer)
    
    return type(fi) == FI


def main():
    model_arch = 'resnet50'    
    model = models.__dict__[model_arch](pretrained=True)
    
    logTestStart("testing FI instantiation")
    
    if test_instFI(model, 0):
        logTestPass()
    else:
        logTestError()


if __name__ == '__main__':
    main()