import cv2
import numpy as np
import os

current_folder = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(current_folder, 'dark_car.jpg')
output_path = os.path.join(current_folder, 'bright_car.jpg')

print("1. Creating a synthetic dark image with 'headlights'...")
# Create a very dark grey image (height 300, width 400)
dark_image = np.full((300, 400, 3), 30, dtype=np.uint8)
# Draw two bright circles (headlights)
cv2.circle(dark_image, (120, 150), 30, (255, 255, 200), -1)
cv2.circle(dark_image, (280, 150), 30, (255, 255, 200), -1)
cv2.imwrite(input_path, dark_image)

print("2. Reading the image back using OpenCV...")
image = cv2.imread(input_path)

if image is None:
    print("Error: Your OpenCV installation is broken.")
else:
    print("3. Applying CLAHE to brighten the dark areas...")
    # Convert to LAB and apply CLAHE to the lightness channel
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l)
    
    merged = cv2.merge((enhanced_l, a, b))
    final = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    cv2.imwrite(output_path, final)
    print("SUCCESS! Check your folder for 'bright_car.jpg'")