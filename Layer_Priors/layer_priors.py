# Implementation of Layer Prior Method for Rain Streak Removal
import torch
import torch.optim as optim
import numpy as np
import cv2

class RainStreakRemoval:
    def __init__(self, image, max_iter=1000, lr=0.01, device='cuda'):
        """
        Initializes the RainStreakRemoval class.

        Parameters:
        - image (np.array): Input RGB image with rain streaks, assumed to be normalized to [0, 1].
        - max_iter (int): Maximum number of optimization iterations.
        - lr (float): Learning rate for the optimizer.
        - device (str): 'cuda' or 'cpu' for running on GPU or CPU.
        """
        # Set device based on availability
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Convert the image to 8-bit format and convert from RGB to YUV
        image = (image * 255).astype(np.uint8)
        image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV).astype(np.float32) / 255.0

        # Extract Y, U, and V channels as PyTorch tensors
        self.Y_channel = torch.tensor(image_yuv[:, :, 0], dtype=torch.float32, device=self.device)
        self.U_channel = torch.tensor(image_yuv[:, :, 1], dtype=torch.float32, device=self.device)
        self.V_channel = torch.tensor(image_yuv[:, :, 2], dtype=torch.float32, device=self.device)

        # Estimate the rain angle and compute its perpendicular direction
        self.rain_angle90 = self.find_angle(image_yuv[:, :, 0])

        self.max_iter = max_iter
        self.lr = lr      

    def find_angle(self, image):
        """
        Estimates the dominant rain streak direction (perpendicular angle).

        Parameters:
        - image (np.array): Grayscale image (Y-channel of YUV).

        Returns:
        - float: Dominant rain angle in radians (perpendicular direction).
        """
        # Compute gradients using Sobel filters
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

        # Compute gradient magnitude and angle
        mag = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx)
        
        # Select top 1% strongest gradients
        threshold = np.percentile(mag, 99)
        mask = mag > threshold

        if np.sum(mask) > 0:
            # Histogram of angles to find the dominant direction
            hist, bins = np.histogram(angle[mask], bins=180, range=(-np.pi, np.pi))
            rain_angle90 = bins[np.argmax(hist)]
            rain_angle90 = min(rain_angle90, np.pi - rain_angle90)
            return rain_angle90
        else:
            return 0 
    
    def calculate_loss(self, O, B, R):
        """
        Computes the total loss for optimization.

        Parameters:
        - O (torch.Tensor): Original rainy image (Y-channel).
        - B (torch.Tensor): Estimated clean background.
        - R (torch.Tensor): Estimated rain streaks.

        Returns:
        - torch.Tensor: Total loss value.
        """
        # Data fidelity term (reconstruction loss)
        data_fidelity = torch.norm(O - B - R + B * R, p='fro')**2

        # Total variation regularization on the background
        B_dx = (B[1:, :-1] - B[:-1, :-1])
        B_dy = (B[:-1, 1:] - B[:-1, :-1])
        B_tv = 0.02 * torch.sqrt(B_dx**2 + B_dy**2 + 1e-16).sum()

        # Sparsity penalty on the rain streak layer
        R_sparse = 0.005 * torch.sum(torch.abs(R))

        # Directional sparsity of rain gradients (along estimated rain direction)
        R_dx = (R[1:, :-1] - R[:-1, :-1])
        R_dy = (R[:-1, 1:] - R[:-1, :-1])
        R_theta_grad = R_dx * np.cos(self.rain_angle90) + R_dy * np.sin(self.rain_angle90)
        R_grad_sparse = 0.01 * torch.sum(torch.abs(R_theta_grad))

        return data_fidelity + B_tv + R_sparse + R_grad_sparse  # Total combined loss
    
    def optimize(self):
        """
        Runs the optimization loop to separate rain streaks from the input image.

        Returns:
        - np.array: Derained RGB image.
        - torch.Tensor: Estimated rain streak layer (Y-channel).
        """
        O = self.Y_channel
        B = O.clone().detach().requires_grad_(True)
        R = torch.zeros_like(O, device=self.device).requires_grad_(True)
        
        # Set up optimizer for B and R
        optimizer = optim.Adam([B, R], lr=self.lr)

        print("Starting the optimization.")

        for iter in range(self.max_iter):
            optimizer.zero_grad()
            loss = self.calculate_loss(O, B, R)
            loss.backward()
            optimizer.step()
            
            # Clamp values to keep B and R in valid range [0, 1]
            with torch.no_grad():
                B.clamp_(0.0, 1.0)
                R.clamp_(0.0, 1.0)

        print("Optimization ended.")
        
        # Reconstruct the derained image in RGB color space
        B_y = B.detach().cpu().numpy()
        U = self.U_channel.cpu().numpy()
        V = self.V_channel.cpu().numpy()

        B_yuv = np.stack([B_y, U, V], axis=2)
        B_yuv = (B_yuv * 255).astype(np.uint8)
        B_rgb = cv2.cvtColor(B_yuv, cv2.COLOR_YUV2RGB)

        # Apply thresholding to convert into Binary mask
        R_binary = (R.detach().cpu().numpy() > 0.05)

        return B_rgb, R_binary