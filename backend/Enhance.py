import torch
import cv2
import numpy as np
from torchvision import transforms
from model import enhance_net_nopool # Imports the architecture from model.py

def load_zero_dce():
    # Initialize the model
    DCE_net = enhance_net_nopool()
    
    # Load the pre-trained weights (Epoch99.pth)
    # map_location='cpu' ensures it works even if you don't have a dedicated GPU
    DCE_net.load_state_dict(torch.load('Epoch99.pth', map_location='cpu'))
    
    # Set the model to evaluation mode (not training)
    DCE_net.eval()
    return DCE_net

def apply_zero_dce(image_path, model):
    # Load the dark image using OpenCV
    data_lowlight = cv2.imread(image_path)
    
    # Normalize pixel values to be between 0 and 1
    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    
    # Convert from Numpy array to PyTorch Tensor and rearrange dimensions (from HWC to CHW)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2,0,1)
    
    # Add a batch dimension (BCHW)
    data_lowlight = data_lowlight.unsqueeze(0)
    
    # Run the image through the network without calculating gradients (saves memory)
    with torch.no_grad():
        _, enhanced_image, _ = model(data_lowlight)
    
    # Convert the output tensor back to a format OpenCV can save
    enhanced_image = enhanced_image.squeeze().permute(1, 2, 0).numpy()
    enhanced_image = (enhanced_image * 255.0).astype(np.uint8)
    
    return enhanced_image

# --- Testing the Pipeline ---
if __name__ == '__main__':
    print("Loading Zero-DCE model...")
    dce_model = load_zero_dce()
    
    # Make sure you have a dark image named 'dark_frame.jpg' in the same folder
    print("Enhancing image...")
    result = apply_zero_dce('dark_frame.jpg', dce_model)
    
    # Save the brightened result
    cv2.imwrite('brightened_frame.jpg', result)
    print("Success! Check 'brightened_frame.jpg'")