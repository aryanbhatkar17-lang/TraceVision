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


def test_synthetic_image():
    current_folder = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_folder, 'dark_car.jpg')
    output_path = os.path.join(current_folder, 'bright_car.jpg')

    print("1. Creating a synthetic dark image with 'headlights'...")
    dark_image = np.full((300, 400, 3), 30, dtype=np.uint8)
    cv2.circle(dark_image, (120, 150), 30, (255, 255, 200), -1)
    cv2.circle(dark_image, (280, 150), 30, (255, 255, 200), -1)
    cv2.imwrite(input_path, dark_image)

    print("2. Reading the image back using OpenCV...")
    image = cv2.imread(input_path)
    if image is None:
        print("Error: Your OpenCV installation is broken.")
    else:
        print("3. Applying CLAHE to brighten the dark areas...")
        final = enhance_frame(image)
        cv2.imwrite(output_path, final)
        print("SUCCESS! Check your folder for 'bright_car.jpg'")


def test_video():
    current_folder = os.path.dirname(os.path.abspath(__file__))
    input_video_path = os.path.join(current_folder, 'dark_video.mp4')
    output_video_path = os.path.join(current_folder, 'enhanced_video.mp4')

    if not os.path.exists(input_video_path):
        print(f"Notice: '{input_video_path}' not found. Running synthetic image test instead.")
        test_synthetic_image()
        return

    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print("Processing video... Press 'q' inside the window to stop early.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        bright_frame = enhance_frame(frame)
        out.write(bright_frame)

        preview_orig = cv2.resize(frame, (640, 360))
        preview_enh = cv2.resize(bright_frame, (640, 360))
        cv2.putText(preview_orig, "ORIGINAL (DARK)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(preview_enh, "CLAHE ENHANCED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        side_by_side = np.hstack((preview_orig, preview_enh))
        cv2.imshow('Night Enhancement Preview (Original vs CLAHE)', side_by_side)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            print("Stopping preview early...")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Done! Enhanced video saved as '{output_video_path}'.")


if __name__ == "__main__":
    current_folder = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(current_folder, 'dark_video.mp4')):
        test_video()
    else:
        test_synthetic_image()
