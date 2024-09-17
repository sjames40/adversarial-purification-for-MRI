# Robust Physics-based Deep MRI Reconstruction Via Diffusion Purification

The paper can be found here: https://arxiv.org/pdf/2309.05794



### Brief explanation of the inference procedure
The primary workflow consists of two key stages: first, the purification process, implemented using the adv_purification code. Second is passing through the MoDL pre-trained model.  

### Setting up the code: 

Install the required dependencies:
mkdir weights
wget -O weights/checkpoint_95.pth https://www.dropbox.com/s/27gtxkmh2dlkho9/checkpoint_95.pth?dl=0

#### create env and activate
conda create -n name python=3.8
conda activate name

#### install dependencies
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch


Usage Download the dataset can be found from Dropbox: Data avaliable on https://www.dropbox.com/scl/fi/801dxovhbkp2bkl2krz5x/NEW_KSPACE.zip?rlkey=4u3b32f6c4pfujsv3kp7z5bdk&st=hwe9thrv&dl=0

Open and run the adv_purification.py to have the purification result for the initial stage and evaluate the model on image restoration tasks based on the pretrained MoDL model and run th test case result.

The pretraind MoDL is also on the dropbox.

The pretrained diffusion model can also be found in the dropbox link. Please note the pre-trained model was adopted from: https://arxiv.org/pdf/2110.05243


#### Directory Structure:
models/: Contains model architecture code which have the DIDN network and the score based MRI model network

autoattack/: Implementation of the auto attack algorithom.

utils/: Utility functions for the project.
