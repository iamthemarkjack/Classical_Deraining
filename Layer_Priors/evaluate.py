# Script to run evaluations
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from Layer_Priors.layer_priors import RainStreakRemoval
from Calculate_IoU import calculate_iou

# Set the Validation Directories
val_rain_dir = r'../Validation Set/Rain'
val_mask_dir = r'../Validation Set/Binary_mask'

paths = os.listdir(val_rain_dir)

# Create the Validation Predictions directory if doesn't exists already
os.makedirs('Validation_Predictions', exist_ok=True)

# Array to store the IoU Scores
scores = []

for i, path in enumerate(paths):
    img = plt.imread(os.path.join(val_rain_dir, path)) # Rainy Input Image
    R_gt = plt.imread(os.path.join(val_mask_dir, path[:-4] + "masks.png")) # Ground Truth Rain Streak Layer

    # Initialize the Solver
    r = RainStreakRemoval(img)
    B, R_pred = r.optimize() # Solve for B and R_pred

    # Compute the score and store it
    iou_score = calculate_iou(R_pred, R_gt)
    scores.append(iou_score)

    # Saving the predicted Binary mask
    plt.imsave(f'Validation_Predictions/{path}', R_pred, cmap='gray')
    print(f"Done with {path}.")

# Saving the IoU Scores as a csv for reference
pd.DataFrame({'Image': paths, 'IoU Score': scores}).to_csv(r'IoU_Scores_Layer_Priors.csv', index=False)

print("Mean Score: ", np.mean(scores))
print("Min Score: ", np.min(scores))
print("Max Score: ", np.max(scores))
print("Standard Deviation: ", np.std(scores))