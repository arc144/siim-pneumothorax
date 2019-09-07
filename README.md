# Summary
The code is split in two parts. The first part is under the subdirectory `./See` and has its own README and 
instructions. The seconds, is under this directory.

# Requirements
The requirements for the first part are listed under `./See/README.md`. The requirements for the second part are 
listed in `requirements.txt`.
 
# Instructions
1. First, navigate into `See` folder and follow the instructions in `./See/README.md` to train and save predictions 
to disk.
2. Train all models from this directory by executing: `./train_all.sh`. Note that the `.csv` folds (located at `
./Data/Folds`) have the full paths of images. Therefore, one should adjust it accordingly to its own directory tree.
3. Once the models are fully trained, the saved weights are located at `./Saves/<model_name>/<fold>/<date>/`. Adjust 
the path to the weights for each model's config file (located at `./Config/<model_name>.py`).
4. Inference must be done by running `python inference.py`.
5. Now all predictions are already saved to disk. Finally, the submission `.csv` can be created by running `python 
from_zip_to_sub.py`. The submission is store in the directory `./Output`.


