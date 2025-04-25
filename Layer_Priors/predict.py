# Script to predict Binary rain mask of Single Rain Images dataset
import os
import sys
import matplotlib.pyplot as plt

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from Layer_Priors.layer_priors import RainStreakRemoval

# Set the Data Directory
dir = r'../Single Rain Images'

filenames = os.listdir(dir)

# Create the Predictions directory if doesn't exists already
os.makedirs('Predictions', exist_ok=True)

for i, filename in enumerate(filenames):
    img = plt.imread(os.path.join(dir, filename)) # Rainy Input Image

    # Initialize the Solver
    r = RainStreakRemoval(img)
    B, R_pred = r.optimize() # Solve for B and R_pred

    # Saving the predicted Binary mask
    plt.imsave(f'Predictions/{filename}', R_pred, cmap='gray')
    print(f"Done with {filename}.")