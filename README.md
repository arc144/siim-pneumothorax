I'd like to thank Kaggle and the organizers for hosting this competition and congratulate all the competitors! Also, a special thanks to my partner @seesee.  
  
  # Solution description
##  TL;DR  
Our solution is quite simple, it is an ensemble (simple average) of four different models (several folds each):  
- FPNetResNet50 (5 folds)  
- FPNetResNet101 (5 folds)  
- FPNetResNet101 (7 folds with different seeds)  
- PANetDilatedResNet34 (4 folds)  
- PANetResNet50 (4 folds)  
- EMANetResNet101 (2 folds)  
  
Models trained at 768x768 (or close to that) using  [AdamW](https://arxiv.org/abs/1711.05101) optimizer.  For the [FPN](https://arxiv.org/abs/1612.03144) models Flip TTA was used whilst Scale (1024) TTA was used for the rest. We used two thresholds, one for segmentation and another (of higher value) for classification.  
  
## See's Models  
See trained models based on the [Feature Pyramid Network](https://arxiv.org/abs/1612.03144) decoder and ResNet as encoders. The ResNet models were based on the modified version proposed in the [Bag-of-tricks paper](https://arxiv.org/abs/1812.01187).  See used [AdamW](https://arxiv.org/abs/1711.05101)  as optmizer and a linear decay with warm-up for the learning rate schedule. For loss he used a weighted combination for BCE and Dice.  Horizontal flip TTA was also used. 

In total he trained 17 models:
- FPNetResNet50 (5 folds)  
- FPNetResNet101 (5 folds)  
- FPNetResNet101 (7 folds with different seeds)  

See's code is available in his [GitHub repo](https://github.com/see--/pneumothorax-segmentation).

## Eduardo's Models
I trained models based on two different types of decoders: [EMANet](http://arxiv.org/abs/1907.13426) and [PANet](http://arxiv.org/abs/1805.10180). Both of my own re-implementation with small modifications:

 - For the PANet I just reduced the number of channels before going to the FPA module. I didn't notice any loss of performance and the network was  much lighter.
   
 - For EMANets I changed the steep bilinear upsampling of 8x for gradual 2x upsampling using the GAU module of  PANet. It considerably improved the performance.

For the encoders I also used the ResNet family with the modifications proposed by the [Bag-of-tricks paper](https://arxiv.org/abs/1812.01187). My models were  trained using [AdamW](https://arxiv.org/abs/1711.05101)  with [Cosine Annealing](https://arxiv.org/abs/1803.05407) learning rate schedule. For the loss a simple BCE worked out fine. In total I trained 10 models at 768x768 resolution:

- PANetDilatedResNet34 (4 folds)  
- PANetResNet50 (4 folds)  
- EMANetResNet101 (2 folds)  

For prediction Scale TTA (768, 1024) was used.

## Ensembling and tresholding
For the ensembling a simple average of the models' soft predictions (after sigmoid, before thresholding)  were used. Finnaly, we used two values for thresholding. One of smaller value (0.2) is used for segmentation and another (0.72) is used for classification of non-pneumothorax images.

## Things that did not work
- HRNet
- FishNet as encoder (not bad but no improvements either)
- A second-level classifier based on gradient boosting using segnets' outputs and metadata as features

## Hardware
- See had access to V100s
- Eduardo used a 1080ti

### Requirements:
-	 At least 11GB VRAM GPU
-	More than 32GB RAM (for ensembling)

# Code Instructions  
In order to generate the submission file, one is expected to use both See's and Eduardo's models. Therefore, the code is split in two parts. The first part is under the subdirectory `./See` and has its own README and   
instructions. The seconds, is under this directory.  
  
## Requirements  
The requirements for the first part are listed under `./See/README.md`. The requirements for the second part are   
listed in `requirements.txt`.  
   
## Instructions  
1. First, navigate into `See` folder and follow the instructions in `./See/README.md` to train and save predictions   
to disk (running  `ensemble_all.py` is not required).
2. Train all models from this directory by executing: `./train_all.sh`. Note that the `.csv` folds (located at `  
./Data/Folds`) have the full paths of images. Therefore, one should adjust it accordingly to its own directory tree (you can easily do that using `pandas`).  
3. Once the models are fully trained, the saved weights are located at `./Saves/<model_name>/<fold>/<date>/`. Adjust   
the path to the weights for each model's config file (located at `./Config/<model_name>.py`).  
4. Inference must be done by running `python inference.py`.  
5. Now all predictions are already saved to disk. Finally, the submission `.csv` can be created by running `python from_zip_to_sub.py`. The submission is store in the directory `./Output`.
