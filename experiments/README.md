# ResNet50 Expirements
--------------------------------------------------------------------------------

This tutorial explain how to run torchFI experiments using [ResNet50](https://arxiv.org/pdf/1512.03385.pdf) model.

- [Dataset](#dataset)
  - [Pre-processing](#pre-processing)
- [Experiments](#citation)
  - [Requirements](#requirements)
  - [Runing Experiments](#runing-experiments)

## Dataset

  All experiments require 2012 [ImageNet](http://image-net.org/) validation set (6.4GB).

### Pre-processing

  PyTorch requires a pre-processing step over Imagenet validation set using the script [valprep.sh](https://github.com/bfgoldstein/torchfi/util/valprep.sh).

  ```bash
    cp ${project}/torchfi/util/valprep.sh ILSVRC2012_img_val/
    cd ILSVRC2012_img_val/
    chmod 755 valprep.sh
    ./valprep.sh
  ```

  where ILSVRC2012_img_val is the folder containing the Imagenet 2012 validation set.

  The folder name also should be *val*. One way to keep the orignal name is creating a symbolik link:

  ```bash
    ln -s /full/path/to/ILSVRC2012_img_val/ /full/path/to/val/
  ```

## Experiments

### Requirements

  ResNet50 experiments require some computational and storage resources. Each job/run uses around 4GB from the main memory and up to 4 CPU cores (threads to load and feed data into GPU). Resutls are stored into a Python object with size around 6MB each. Keep in mind that runing all 6820 jobs from just one script (resnet50.sh or resnet50_pruned.sh) will require around 40GB of storage.

  All experiments require Anaconda torchfi environment. Activate it with *conda activate torchfi* before runing any command below. Check [this tutorial](https://github.com/bfgoldstein/torchfi#install-dependencies) if you do not have torchfi env installed.

### Runing Experiments

#### Full Set

  [resnet50.sh](https://github.com/bfgoldstein/torchfi/util/resnet50.sh) and [resnet50_pruned.sh](https://github.com/bfgoldstein/torchfi/util/resnet50_pruned.sh) provide a full set (13640 jobs) of fault injection experiments. To run these experiments the pre-processed validation set folder and the batch size should be provided.

  ```bash
    ./resnet50.sh -d /home/dataset/imagenet/ -b 320 -g 0
  ```

  The example above run ResNet50 wihtout pruning on GPU id *0* using validation set under */home/dataset/imagenet/* folder with batch size equals to *320*.

#### Single run

**TODO**