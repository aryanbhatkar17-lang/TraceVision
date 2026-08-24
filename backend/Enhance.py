import os 
import torch
import cv2
import numpy as np
import logging
from typing import Optional
from model import EnhanceNetNoPool

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("zero-dce")

def load_zero_dce() -> Optional[torch.nn.Module]:
    """
    Loads and initializes the Zero-DCE model with pre-trained weights.
    
    Returns:
        Loaded model or None if loading fails
    """
    try:
        # 1. Updated class name to PascalCase
        DCE_net = EnhanceNetNoPool() 
        
        # 2. Dynamic path generation
        model_path = os.path.join(os.getcwd(), 'Epoch99.pth')
        
        # 3. Error handling with proper fallback
        if not os.path.exists(model_path):
            logger.error(f"CRITICAL ERROR: AI model weights not found at {model_path}")
            logger.info("Please download Epoch99.pth and place it in the working directory")
            return None
            
        DCE_net.load_state_dict(torch.load(model_path, map_location='cpu'))
        DCE_net.eval()
        logger.info("Zero-DCE model loaded successfully")
        return DCE_net
        
    except Exception as e:
        logger.error(f"Failed to load Zero-DCE model: {e}")
        return None

def apply_zero_dce(image_input: any, model: Optional[torch.nn.Module] = None) -> Optional[np.ndarray]:
    """
    Enhances a single image or video frame using Zero-DCE.
    
    Args:
        image_input: Either a file path (str) or loaded OpenCV frame (numpy array)
        model: Pre-loaded model. If None, will attempt to load.
    
    Returns:
        Enhanced frame as numpy array or None if enhancement fails
    """
    # Input validation
    if image_input is None:
        logger.warning("apply_zero_dce: Received None input")
        return None
    
    try:
        # Load model if not provided
        if model is None:
            model = load_zero_dce()
            if model is None:
                logger.warning("apply_zero_dce: Could not load model, returning None")
                return None
        
        # Accept either a file path (str) or an already loaded OpenCV numpy frame
        if isinstance(image_input, str):
            frame = cv2.imread(image_input)
            if frame is None:
                logger.error(f"Failed to read image from path: {image_input}")
                return None
        else:
            frame = image_input
        
        # Validate frame
        if frame is None or frame.size == 0:
            logger.warning("apply_zero_dce: Received empty or invalid frame")
            return None

        # Normalize pixel values (0 to 1) and prepare tensor
        try:
            data_lowlight = (np.asarray(frame, dtype=np.float32) / 255.0)
            data_lowlight = torch.from_numpy(data_lowlight).float().permute(2, 0, 1).unsqueeze(0)
            
            # Execute model inference
            with torch.no_grad():
                _, enhanced_tensor, _ = model(data_lowlight)
                
            # Convert tensor output back to OpenCV numpy image format
            enhanced_np = enhanced_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            enhanced_np = np.clip(enhanced_np * 255.0, 0, 255).astype(np.uint8)
            
            return enhanced_np
            
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            return None
    
    except Exception as e:
        logger.error(f"Unexpected error in apply_zero_dce: {e}")
        return None

def enhance_video(input_video_path: str, output_video_path: str) -> bool:
    """
    Processes a video file frame-by-frame through Zero-DCE and outputs an enhanced video.
    
    Args:
        input_video_path: Path to input video file
        output_video_path: Path to save enhanced video
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load model
        model = load_zero_dce()
        if model is None:
            logger.error("Cannot enhance video: model not loaded")
            return False
        
        # Open input video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open input video: {input_video_path}")
            return False
        
        # Extract video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Input video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            logger.error(f"Failed to create output video writer: {output_video_path}")
            cap.release()
            return False
        
        logger.info("Processing video frames through Zero-DCE...")
        frame_count = 0
        failed_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # --- INPUT VALIDATION ---
            if frame is None or frame.size == 0:
                logger.warning(f"Skipping empty/corrupted frame at index {frame_count}")
                failed_frames += 1
                frame_count += 1
                continue
            
            # --- PROGRESS FEEDBACK ---
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                logger.info(f"Processing frame {frame_count}/{total_frames} ({progress:.1f}%)")
            
            # --- ENHANCEMENT ---
            try:
                enhanced_frame = apply_zero_dce(frame, model)
                if enhanced_frame is None:
                    logger.warning(f"Enhancement failed for frame {frame_count}, using original")
                    enhanced_frame = frame
            except Exception as e:
                logger.warning(f"Exception during frame enhancement at {frame_count}: {e}")
                enhanced_frame = frame
            
            # Write enhanced frame
            out.write(enhanced_frame)
            frame_count += 1
        
        # Cleanup
        cap.release()
        out.release()
        
        logger.info(f"SUCCESS! Enhanced video saved to {output_video_path}")
        logger.info(f"Processed {frame_count} frames ({failed_frames} failed, {frame_count - failed_frames} successful)")
        return True
        
    except Exception as e:
        logger.error(f"Fatal error in enhance_video: {e}")
        return False

# --- Testing Section ---
if __name__ == '__main__':
    # To test a single image:
    # result = apply_zero_dce('dark_frame.jpg')
    # if result is not None:
    #     cv2.imwrite('brightened_frame.jpg', result)
    # else:
    #     print("Image enhancement failed")
    
    # To test a full video:
    success = enhance_video('dark_cctv.mp4', 'enhanced_cctv.mp4')
    if not success:
        print("Video enhancement failed")
