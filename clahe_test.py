import cv2
import numpy as np
import os

def enhance_frame(frame):
    """Applies CLAHE enhancement to a single frame in LAB color space."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Clip limit 2.0 keeps headlights/streetlights from glaring
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    
    merged_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

# Find folder path
current_folder = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(current_folder, 'dark_video.mp4')
output_video_path = os.path.join(current_folder, 'enhanced_video.mp4')

# Check if input video exists
if not os.path.exists(input_video_path):
    print(f"Error: Could not find '{input_video_path}'.")
    print("Please place a video file named 'dark_video.mp4' in your project folder.")
    exit()

# Open the video
cap = cv2.VideoCapture(input_video_path)

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

# Setup video writer to save enhanced output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print("Processing video... Press 'q' inside the window to stop early.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # 1. Enhance the frame
    bright_frame = enhance_frame(frame)
    
    # 2. Save frame to output video file
    out.write(bright_frame)

    # 3. Create a side-by-side comparison for live preview
    # Resize slightly so it fits on your screen if the video is huge
    preview_orig = cv2.resize(frame, (640, 360))
    preview_enh = cv2.resize(bright_frame, (640, 360))
    
    # Put labels on the frames
    cv2.putText(preview_orig, "ORIGINAL (DARK)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(preview_enh, "CLAHE ENHANCED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Stack side-by-side
    side_by_side = np.hstack((preview_orig, preview_enh))
    cv2.imshow('Night Enhancement Preview (Original vs CLAHE)', side_by_side)

    # Wait 25ms per frame; exit if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        print("Stopping preview early...")
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Done! Enhanced video saved as '{output_video_path}'.")