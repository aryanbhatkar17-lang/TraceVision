import cv2

def enhance_frame(frame):
    """Applies CLAHE enhancement to a single frame in LAB color space."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Clip limit 2.0 keeps headlights/streetlights from glaring
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    
    merged_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)