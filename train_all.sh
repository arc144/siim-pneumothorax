#!/usr/bin/env bash

    python train.py -m PANetDilatedResNet34_768_Fold0
    python train.py -m PANetDilatedResNet34_768_Fold1
    python train.py -m PANetDilatedResNet34_768_Fold3
    python train.py -m PANetDilatedResNet34_768_Fold4
    python train.py -m PANetResNet50_768_Fold0
    python train.py -m PANetResNet50_768_Fold1
    python train.py -m PANetResNet50_768_Fold2
    python train.py -m PANetResNet50_768_Fold3
    python train.py -m EMANetResNet101_768_Fold7
    python train.py -m EMANetResNet101_768_Fold8