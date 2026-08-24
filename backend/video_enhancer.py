import cv2
import numpy as np

def enhance_frame(frame):
    """
    Advanced Night-Vision Pipeline for AI Object Detection.
    Runs completely in the background.
    """
    # STEP 1: Gamma Correction (Lifts the darkest shadows naturally)
    gamma = 1.5
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    brightened = cv2.LUT(frame, table)
    
    # STEP 2: CLAHE (Sharpens the edges of the cars so the AI can see the shapes)
    lab = cv2.cvtColor(brightened, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    
    merged_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    high_contrast = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    
    # STEP 3: Mild Denoising (Blurs the static grain so the AI doesn't get confused)
    # 3x3 Gaussian Blur is extremely fast and removes sensor noise
    final_clean_frame = cv2.GaussianBlur(high_contrast, (3, 3), 0)
    
    return final_clean_frame