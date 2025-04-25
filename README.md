# Classical Rain Streaks Removal

After comparing multiple methods, we finalized **Rain Streaks Removal using Layer Priors** for the final project evaluation.

The relevant code is located in the `Layer_Priors` directory. To run the model, create an instance of the `RainStreakRemoval` class from `layer_priors.py`, and pass in the rainy image as a NumPy array (ensure it's an RGB image).

You can refer to `evaluate.py` or `predict.py` for example usage, or use the snippet below:

```python
import matplotlib.pyplot as plt
from layer_priors import RainStreakRemoval

img = plt.imread(path_to_your_rgb_image)  # Rainy Input Image

# Initialize the Solver
r = RainStreakRemoval(img)

# Optimize to get clean background (B) and predicted rain layer (R_pred)
B, R_pred = r.optimize()
```

## Instructions to Run the Code on the Validation Set

Before running the code, make sure to install all dependencies listed in `requirements.txt`.

---

### 1. Rain Streak Removal using Layer Priors
- Navigate to the `Layer_Priors` folder and run:
  ```bash
  python evaluate.py
  ```
- You can adjust the number of iterations for better performance.
- By default, the code utilizes a CUDA-enabled GPU if available; otherwise, it falls back to the CPU.

---

### 2. Morphological Component Analysis (MCA)
- Navigate to the `Morphological_Component_Analysis` folder and run:
  ```bash
  python MCA.py
  ```
- You can experiment with:
  - Patch size
  - Number of atoms in the dictionary
  - Number of patches used in dictionary learning
