import os
import numpy as np
import torch
import torch.optim as optim
import cv2
import matplotlib.pyplot as plt

class RainStreakRemoval:
    def __init__(self, image, max_iter=100, lr=0.01, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        image = (image * 255).astype(np.uint8)
        image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV).astype(np.float32) / 255.0

        self.Y_channel = torch.tensor(image_yuv[:, :, 0], dtype=torch.float32, device=self.device)
        self.U_channel = torch.tensor(image_yuv[:, :, 1], dtype=torch.float32, device=self.device)
        self.V_channel = torch.tensor(image_yuv[:, :, 2], dtype=torch.float32, device=self.device)

        self.rain_angle90 = self.find_angle(image_yuv[:, :, 0])
        self.max_iter = max_iter
        self.lr = lr      

    def find_angle(self, image):
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

        mag = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx)
        
        threshold = np.percentile(mag, 99)
        mask = mag > threshold

        if np.sum(mask) > 0:
            hist, bins = np.histogram(angle[mask], bins=180, range=(-np.pi, np.pi))
            rain_angle = bins[np.argmax(hist)]
            rain_angle = min(rain_angle, np.pi - rain_angle)
            return rain_angle
        else:
            return 0 
    
    def calculate_loss(self, O, B, R):
        data_fidelity = torch.norm(O - B - R + B * R, p='fro')**2

        B_dx = (B[1:, :-1] - B[:-1, :-1])
        B_dy = (B[:-1, 1:] - B[:-1, :-1])
        B_tv = 0.02 * torch.sqrt(B_dx**2 + B_dy**2 + 1e-16).sum()

        R_sparse = 0.005 * torch.sum(torch.abs(R))

        R_dx = (R[1:, :-1] - R[:-1, :-1])
        R_dy = (R[:-1, 1:] - R[:-1, :-1])
        R_theta_grad = R_dx * np.cos(self.rain_angle90) + R_dy * np.sin(self.rain_angle90)
        R_grad_sparse = 0.01 * torch.sum(torch.abs(R_theta_grad))

        return data_fidelity + B_tv + R_sparse + R_grad_sparse
    
    def optimize(self):
        O = self.Y_channel
        B = O.clone().detach().requires_grad_(True)
        R = torch.zeros_like(O, device=self.device).requires_grad_(True)
        
        optimizer = optim.Adam([B, R], lr=self.lr)

        for iter in range(self.max_iter):
            optimizer.zero_grad()
            loss = self.calculate_loss(O, B, R)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                B.clamp_(0.0, 1.0)
                R.clamp_(0.0, 1.0)
        
        B_y = B.detach().cpu().numpy()
        U = self.U_channel.cpu().numpy()
        V = self.V_channel.cpu().numpy()

        B_yuv = np.stack([B_y, U, V], axis=2)
        B_yuv = (B_yuv * 255).astype(np.uint8)
        B_rgb = cv2.cvtColor(B_yuv, cv2.COLOR_YUV2RGB)

        return B_rgb, R.detach().cpu()

dir = r'../Validation Set/Rain'
paths = os.listdir(dir)

os.makedirs('Predictions/Background', exist_ok=True)
os.makedirs('Predictions/Rain', exist_ok=True)

for path in paths:
    img = plt.imread(os.path.join(dir, path))
    r = RainStreakRemoval(img)
    B, R = r.optimize()
    plt.imsave(f'Predictions/Background/{path}', B)
    plt.imsave(f'Predictions/Rain/{path}', R > 0.05, cmap='gray')
    print(f"Done with {path}.")