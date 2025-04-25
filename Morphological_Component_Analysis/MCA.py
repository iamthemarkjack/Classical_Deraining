import imageio
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import DictionaryLearning
from skimage.feature import hog
from skimage import exposure
from sklearn.cluster import KMeans
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.feature_extraction.image import extract_patches_2d
import os

def load_images(img_path, mask_path):
    img = imageio.imread(img_path)
    imgground = imageio.imread(mask_path)
    return img, imgground

def get_high_freq_image(img, blur_kernel=(25, 25)):
    img2 = cv2.GaussianBlur(img, blur_kernel, 0)
    imgHighFreq = cv2.subtract(img, img2)
    imgHighFreq = cv2.add(imgHighFreq, 127)
    return imgHighFreq

def extract_patches(img, patch_size=(4, 4), max_patches=500):
    patches = extract_patches_2d(img, patch_size=patch_size, max_patches=max_patches)
    return patches

def learn_dictionary(patches, n_components=32, max_iter=300):
    data = patches.reshape(patches.shape[0], -1)
    dlearner = DictionaryLearning(n_components=n_components, max_iter=max_iter, transform_algorithm='omp')
    dlearner.fit(data)
    atoms = dlearner.components_
    atoms = (atoms - atoms.min()) / (atoms.max() - atoms.min())
    atoms = (atoms * 255).astype(np.uint8)
    return atoms

def compute_hog_features(atoms, patch_size=(4, 4)):
    hogFeatures = []
    for i in range(atoms.shape[0]):
        atom_img = atoms[i].reshape((*patch_size, 3))
        features= hog(atom_img, 9, patch_size, channel_axis=-1, cells_per_block=(1, 1), visualize=False)
        hogFeatures.append(features)
    return hogFeatures

def cluster_atoms(hogFeatures, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10)
    kmeans.fit(hogFeatures)
    clusterLabels = kmeans.labels_
    return clusterLabels

def identify_rain_cluster(hogFeatures, clusterLabels):
    cluster_0_indices = [i for i in range(len(clusterLabels)) if clusterLabels[i] == 0]
    cluster_1_indices = [i for i in range(len(clusterLabels)) if clusterLabels[i] == 1]
    cluster_0_hog_features = [hogFeatures[i] for i in cluster_0_indices]
    cluster_1_hog_features = [hogFeatures[i] for i in cluster_1_indices]
    mean_0 = np.mean(cluster_0_hog_features, axis=0)
    mean_1 = np.mean(cluster_1_hog_features, axis=0)
    variance_0 = np.mean([np.sum((feature - mean_0) ** 2) for feature in cluster_0_hog_features])
    variance_1 = np.mean([np.sum((feature - mean_1) ** 2) for feature in cluster_1_hog_features])
    if variance_0 < variance_1:
        rainCluster = 0
    else:
        rainCluster = 1
    return rainCluster

def generate_rain_mask(imgHighFreq, atoms, clusterLabels, rainCluster, patch_size=(4,4), L=5):
    mask = np.zeros((imgHighFreq.shape[0], imgHighFreq.shape[1]))
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=L, tol=None, fit_intercept=False)
    for i in range(0, imgHighFreq.shape[0] - patch_size[0] + 1, patch_size[0]):
        for j in range(0, imgHighFreq.shape[1] - patch_size[1] + 1, patch_size[1]):
            patch = imgHighFreq[i:i + patch_size[0], j:j + patch_size[1]].reshape(-1)
            omp.fit(atoms.T, patch)
            coef = omp.coef_
            if np.argmax(coef) in [idx for idx, label in enumerate(clusterLabels) if label == rainCluster]:
                mask[i:i + patch_size[0], j:j + patch_size[1]] = 1
            else:
                mask[i:i + patch_size[0], j:j + patch_size[1]] = 0
    return mask

def calculate_iou(mask_pred, mask_gt):
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    iou = intersection / union if union != 0 else 0
    return iou

def show_mask_comparison(img, mask):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].axis('off')
    axes[0].set_title("Ground truth")
    axes[1].imshow(mask, cmap='gray')
    axes[1].axis('off')
    axes[1].set_title("Rain Mask")
    plt.tight_layout()
    plt.show()

def MCA(img):
    imgHighFreq = get_high_freq_image(img)

    patchSize = (4, 4)
    patches = extract_patches(imgHighFreq, patch_size=patchSize, max_patches=350)
    atoms = learn_dictionary(patches, n_components=32, max_iter=300)

    hogFeatures = compute_hog_features(atoms, patch_size=patchSize)

    clusterLabels = cluster_atoms(hogFeatures)
    rainCluster = identify_rain_cluster(hogFeatures, clusterLabels)

    mask = generate_rain_mask(imgHighFreq, atoms, clusterLabels, rainCluster, patch_size=patchSize, L=5)

    return mask

# Example usage (replace with your paths)
rain_folder = "Validation set/Rain/"
mask_folder = "Validation set/Binary_mask/"
predictions_folder = "predictions"

rain_images = sorted(os.listdir(rain_folder))
mask_images = sorted(os.listdir(mask_folder))

for rain_image, mask_image in zip(rain_images, mask_images):
    img_path = os.path.join(rain_folder, rain_image)
    mask_path = os.path.join(mask_folder, mask_image)

    img, imgground = load_images(img_path, mask_path)

    mask = MCA(img)

    prediction_path = os.path.join(predictions_folder, f"mask_{rain_image}")
    cv2.imwrite(prediction_path, (mask * 255).astype(np.uint8))

    iou_score = calculate_iou(mask, imgground)
    with open("iou_scores.csv", "a") as f:
        f.write(f"{rain_image},{iou_score}\n")
    #show_mask_comparison(imgground, mask)
    print(f"IOU for {rain_image}: {iou_score}")
