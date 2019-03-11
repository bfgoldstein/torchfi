# TorchFI
--------------------------------------------------------------------------------
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/bfgoldstein/torchfi/LICENSE)

TorchFI is a fault injection framework build on top of [pytorch](https://pytorch.org/) for research purposes.

- [Installation](#installation)
  - [Clone Project](#clone-project)
  - [Install Dependencies](#install-dependencies)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

### Clone Project

  ```bash
    git clone git@github.com:bfgoldstein/torchfi.git
  ```

### Install Dependencies

We highly recommend installing an [Anaconda](https://www.continuum.io/downloads) environment.

  ```bash
  conda env create -f torchfi.yml
  conda activate torchfi
  cd ${PROJECT_PATH}
  export PYTHON_PATH=${PYTHON_PATH}
  ```

## Examples/Experiments

Check out [this](https://github.com/bfgoldstein/torchfi/tree/master/experiments) torchFI example using ResNet50 model.

## Citation

Please cite XXX in your publications if it helps your research:

    @article{,
      Author = {Goldstein, Brunno and Srinivasan, Sudarshan and Mellempudi, Naveen K and Das, Dipankar 
                and Marzulo, Leandro A. J. and Solon, Alexandre N. and Kundu, Sandip and França, Felipe M. G.},
      Journal = {},
      Title = {},
      Year = {2019}
    }

## License

TorchFI code is released under the [Apache license 2.0](https://github.com/bfgoldstein/torchfi/LICENSE).

## Acknowledgments

- [PyTorch](https://github.com/pytorch/pytorch) - Python package for fast tensors computation and DNNs execution
- [Distiller](https://github.com/NervanaSystems/distiller) - Ported post-quantization code to TorchFI


Copyright (c) 2018 Brunno Goldstein