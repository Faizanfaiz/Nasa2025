"""SAM (Segment Anything Model) integration for interactive image segmentation."""

import base64
import io
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw
import torch
import cv2

# SAM imports (will be available once torch installation completes)
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("SAM not available yet - torch is still installing")

# Global SAM model instance
_sam_predictor: Optional[SamPredictor] = None
_sam_model = None

def initialize_sam_model():
    """Initialize the SAM model. Call this once when the app starts."""
    global _sam_predictor, _sam_model
    
    if not SAM_AVAILABLE:
        return False
    
    try:
        # SAM model configuration - change these to use different models
        # Available models: "vit_b", "vit_l", "vit_h"
        model_type = "vit_b"  # Use the smaller model for faster loading
        sam_checkpoint = "sam_vit_b_01ec64.pth"
        
        # Alternative model configurations (uncomment to use):
        # model_type = "vit_l"
        # sam_checkpoint = "sam_vit_l_0b3195.pth"
        
        # model_type = "vit_h" 
        # sam_checkpoint = "sam_vit_h_4b8939.pth"
        
        if not os.path.exists(sam_checkpoint):
            print(f"SAM checkpoint {sam_checkpoint} not found.")
            print(f"Please download it using:")
            if model_type == "vit_b":
                print("wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
            elif model_type == "vit_l":
                print("wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth")
            elif model_type == "vit_h":
                print("wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
            return False
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        _sam_model.to(device=device)
        _sam_predictor = SamPredictor(_sam_model)
        
        print(f"SAM model initialized on {device}")
        return True
        
    except Exception as e:
        print(f"Failed to initialize SAM model: {e}")
        return False

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 data URI."""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def base64_to_image(data_uri: str) -> Image.Image:
    """Convert base64 data URI to PIL Image."""
    if data_uri.startswith('data:image'):
        data_uri = data_uri.split(',')[1]
    image_data = base64.b64decode(data_uri)
    return Image.open(io.BytesIO(image_data))

def preprocess_image_for_sam(image: Image.Image) -> np.ndarray:
    """Preprocess image for SAM input with enhanced color differentiation."""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Enhance contrast and color differentiation for better segmentation
    # Apply slight contrast enhancement to make color differences more pronounced
    image_array = image_array.astype(np.float32)
    
    # Enhance contrast (simple linear stretch)
    min_val = np.percentile(image_array, 2)
    max_val = np.percentile(image_array, 98)
    image_array = np.clip((image_array - min_val) / (max_val - min_val) * 255, 0, 255)
    
    # Convert back to uint8
    image_array = image_array.astype(np.uint8)
    
    # Set image for SAM predictor
    if _sam_predictor is not None:
        _sam_predictor.set_image(image_array)
    
    return image_array

def get_segment_at_point(image_array: np.ndarray, x: int, y: int) -> Optional[np.ndarray]:
    """Get segmentation mask for a point click with improved sensitivity."""
    if _sam_predictor is None:
        return None
    
    try:
        # Create input point and label
        input_point = np.array([[x, y]])
        input_label = np.array([1])  # 1 for foreground
        
        # Predict mask with improved parameters for better sensitivity
        masks, scores, logits = _sam_predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        
        # Use more restrictive criteria for better segmentation
        # Filter masks by score threshold and select the most precise one
        score_threshold = 0.7  # Higher threshold for better quality
        valid_masks = [(mask, score) for mask, score in zip(masks, scores) if score > score_threshold]
        
        if valid_masks:
            # Select the mask with the best score among valid ones
            best_mask, best_score = max(valid_masks, key=lambda x: x[1])
            return best_mask
        else:
            # If no mask meets the threshold, return the best available
            best_mask_idx = np.argmax(scores)
            return masks[best_mask_idx]
        
    except Exception as e:
        print(f"Error getting segment: {e}")
        return None

def create_highlighted_image(image: Image.Image, mask: np.ndarray, 
                           highlight_color: Tuple[int, int, int] = (255, 255, 0),
                           alpha: float = 0.5) -> Image.Image:
    """Create an image with highlighted segment."""
    # Convert to numpy array
    img_array = np.array(image)
    
    # Ensure we're working with RGB
    if img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]  # Convert to RGB
    
    # Create highlight overlay with same shape as image
    highlight = np.zeros_like(img_array)
    highlight[mask] = highlight_color
    
    # Blend with original image
    highlighted = img_array.copy().astype(np.float32)
    highlight = highlight.astype(np.float32)
    
    # Apply alpha blending
    highlighted[mask] = (1 - alpha) * highlighted[mask] + alpha * highlight[mask]
    highlighted = np.clip(highlighted, 0, 255).astype(np.uint8)
    
    return Image.fromarray(highlighted)

def extract_segment_image(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Extract just the segmented region as a new image."""
    # Convert to numpy array
    img_array = np.array(image)
    
    # Convert to RGBA if needed
    if img_array.shape[2] == 3:
        rgba_array = np.zeros((img_array.shape[0], img_array.shape[1], 4), dtype=np.uint8)
        rgba_array[:, :, :3] = img_array
        rgba_array[:, :, 3] = 255  # Full opacity
    else:
        rgba_array = img_array.copy()
    
    # Set non-masked pixels to transparent
    rgba_array[~mask] = [0, 0, 0, 0]
    
    # Convert to RGBA
    segment_image = Image.fromarray(rgba_array, 'RGBA')
    
    return segment_image

def get_segment_bounds(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Get bounding box of the segment."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return 0, 0, 0, 0
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    return x_min, y_min, x_max, y_max

def crop_segment(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Crop image to segment bounds."""
    x_min, y_min, x_max, y_max = get_segment_bounds(mask)
    
    # Crop the image
    cropped = image.crop((x_min, y_min, x_max + 1, y_max + 1))
    
    # Crop the mask
    mask_cropped = mask[y_min:y_max+1, x_min:x_max+1]
    
    # Apply mask to cropped image
    cropped_array = np.array(cropped)
    
    # Convert to RGBA if needed
    if cropped_array.shape[2] == 3:
        rgba_array = np.zeros((cropped_array.shape[0], cropped_array.shape[1], 4), dtype=np.uint8)
        rgba_array[:, :, :3] = cropped_array
        rgba_array[:, :, 3] = 255  # Full opacity
    else:
        rgba_array = cropped_array.copy()
    
    # Set non-masked pixels to transparent
    rgba_array[~mask_cropped] = [0, 0, 0, 0]
    
    return Image.fromarray(rgba_array, 'RGBA')

class SAMImageProcessor:
    """Main class for handling SAM-based image processing."""
    
    def __init__(self):
        self.current_image: Optional[Image.Image] = None
        self.current_image_array: Optional[np.ndarray] = None
        self.current_mask: Optional[np.ndarray] = None
        
    def load_image(self, image_data: str) -> bool:
        """Load image from base64 data URI."""
        try:
            self.current_image = base64_to_image(image_data)
            self.current_image_array = preprocess_image_for_sam(self.current_image)
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def get_segment_at_coordinates(self, x: int, y: int) -> Optional[Dict[str, Any]]:
        """Get segment at given coordinates and return processed results."""
        if self.current_image_array is None:
            return None
        
        mask = get_segment_at_point(self.current_image_array, x, y)
        if mask is None:
            return None
        
        self.current_mask = mask
        
        # Create highlighted image
        highlighted = create_highlighted_image(self.current_image, mask)
        
        # Extract segment
        segment = extract_segment_image(self.current_image, mask)
        
        # Crop segment
        cropped_segment = crop_segment(self.current_image, mask)
        
        return {
            'highlighted_image': image_to_base64(highlighted),
            'segment_image': image_to_base64(segment),
            'cropped_segment': image_to_base64(cropped_segment),
            'mask_bounds': get_segment_bounds(mask),
            'mask_area': int(np.sum(mask))
        }
    
    def save_segment(self, filename: str = None) -> Optional[str]:
        """Save the current segment to file."""
        if self.current_image is None or self.current_mask is None:
            return None
        
        if filename is None:
            # Create a new unique filename
            import datetime
            import uuid
            timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            filename = f"segment_{timestamp}_{uuid.uuid4().hex[:8]}.png"
        
        # Ensure the filename has the correct path
        from pathlib import Path
        from v1 import FEATURE_SNAPSHOT_DIR
        
        # If filename doesn't have a path, save it in the feature snapshots directory
        if not Path(filename).is_absolute():
            FEATURE_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
            filename = str(FEATURE_SNAPSHOT_DIR / filename)
        
        segment = extract_segment_image(self.current_image, self.current_mask)
        segment.save(filename)
        return filename

# Global processor instance
sam_processor = SAMImageProcessor()