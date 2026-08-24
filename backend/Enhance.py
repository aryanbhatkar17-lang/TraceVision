import torch
import cv2
import numpy as np
from model import enhance_net_nopool

def load_zero_dce():
    """Loads and initializes the Zero-DCE model with pre-trained weights."""
    DCE_net = enhance_net_nopool()
    DCE_net.load_state_dict(torch.load('Epoch99.pth', map_location='cpu'))
    DCE_net.eval()
    return DCE_net

def apply_zero_dce(image_input, model=None):
    """Enhances a single image or video frame."""
    if model is None:
        model = load_zero_dce()
        
    # Accept either a file path (str) or an already loaded OpenCV numpy frame
    if isinstance(image_input, str):
        frame = cv2.imread(image_input)
    else:
        frame = image_input

    # Normalize pixel values (0 to 1) and prepare tensor
    data_lowlight = (np.asarray(frame) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float().permute(2, 0, 1).unsqueeze(0)
    
    # Execute model inference
    with torch.no_grad():
        _, enhanced_tensor, _ = model(data_lowlight)
        
    # Convert tensor output back to OpenCV numpy image format
    enhanced_np = enhanced_tensor.squeeze().permute(1, 2, 0).numpy()
    enhanced_np = np.clip(enhanced_np * 255.0, 0, 255).astype(np.uint8)
    
    return enhanced_np

def enhance_video(input_video_path, output_video_path):
    """Processes a video file frame-by-frame through Zero-DCE and outputs an enhanced video."""
    model = load_zero_dce()
    cap = cv2.VideoCapture(input_video_path)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print("Processing video frames through Zero-DCE...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        enhanced_frame = apply_zero_dce(frame, model)
        out.write(enhanced_frame)
        
    cap.release()
    out.release()
    print(f"Success! Enhanced video saved as {output_video_path}")

# --- Testing Section ---
if __name__ == '__main__':
    # To test a single image:
    # result = apply_zero_dce('dark_frame.jpg')
    # cv2.imwrite('brightened_frame.jpg', result)
    
    # To test a full video:
    enhance_video('dark_cctv.mp4', 'enhanced_cctv.mp4')