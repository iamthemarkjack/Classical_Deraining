# Classical_Deraining
After a comparison in both our codes, we finalised to use deraining using priors for final project evaluation. \
To run the codes, 
- A folder in the same directory named `Validation_set` must contain two folders: `Binary_mask` and `rain`. 
- Running the code saves the predicted rain mask of each image in the `prediction` and makes a csv file of the IOU scores. 

Specific instructions
1. Layers with priors
- Run `evaluate.py` file in `Layer_priors` folder
- Make sure libraries torch, numpy, matplotlib, pandas, cv2 are installed
- Preffered to run on hardware with GPU available and CUDA installed

2. Morphological component analysis
- Run `MCA.py` file in `Morphological_component_analysis` folder
- Make sure numpy, sklearn, skimage, cv2 libraries are installed
