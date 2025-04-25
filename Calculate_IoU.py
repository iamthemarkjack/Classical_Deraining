import numpy as np

def calculate_iou(pred, gt):
    """
    Calculate the Intersection over Union (IoU) score between two binary masks.

    Parameters:
    -----------
    pred : np.ndarray
        Predicted binary mask. Must be a boolean or 0/1 numpy array of the same shape as `gt`.
    gt : np.ndarray
        Ground truth binary mask. Must be a boolean or 0/1 numpy array of the same shape as `pred`.

    Returns:
    --------
    iou_score: float
        The final intersection over union score
    """
    # Compute the pixel-wise intersection: True where both pred and gt are True
    intersection = np.logical_and(pred, gt).sum()

    # Compute the pixel-wise union: True where either pred or gt (or both) are True
    union = np.logical_or(pred, gt).sum()

    # Avoid division by zero; return IoU score
    iou_score = (intersection / union)

    return iou_score