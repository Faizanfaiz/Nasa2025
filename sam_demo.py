#!/usr/bin/env python3
"""Demo script showing how SAM integration will work."""

import base64
import io
from PIL import Image, ImageDraw
import numpy as np

def create_demo_image():
    """Create a demo Mars surface image for testing."""
    # Create a 400x400 image with Mars-like colors
    img = Image.new('RGB', (400, 400), color=(139, 69, 19))  # Mars brown
    
    # Add some features
    draw = ImageDraw.Draw(img)
    
    # Draw a crater
    draw.ellipse([100, 100, 200, 200], fill=(101, 67, 33), outline=(80, 50, 20))
    draw.ellipse([120, 120, 180, 180], fill=(120, 80, 40))
    
    # Draw some rocks
    draw.ellipse([250, 150, 280, 180], fill=(90, 60, 30))
    draw.ellipse([300, 200, 320, 220], fill=(95, 65, 35))
    draw.ellipse([180, 300, 200, 320], fill=(85, 55, 25))
    
    # Draw a dust storm
    for i in range(50):
        x = 50 + i * 6
        y = 50 + (i % 3) * 10
        draw.ellipse([x, y, x+20, y+20], fill=(160, 120, 80, 100))
    
    return img

def image_to_base64(image):
    """Convert PIL Image to base64 data URI."""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def simulate_sam_segment(image, x, y, radius=30):
    """Simulate SAM segmentation by creating a circular mask around the click point."""
    # Convert to numpy array
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # Create a circular mask around the click point
    y_coords, x_coords = np.ogrid[:height, :width]
    mask = (x_coords - x) ** 2 + (y_coords - y) ** 2 <= radius ** 2
    
    return mask

def create_highlighted_image(image, mask, highlight_color=(255, 255, 0), alpha=0.5):
    """Create an image with highlighted segment."""
    img_array = np.array(image)
    
    # Create highlight overlay
    highlight = np.zeros_like(img_array)
    highlight[mask] = highlight_color
    
    # Blend with original image
    highlighted = img_array.copy().astype(np.float32)
    highlight = highlight.astype(np.float32)
    
    # Apply alpha blending
    highlighted[mask] = (1 - alpha) * highlighted[mask] + alpha * highlight[mask]
    highlighted = np.clip(highlighted, 0, 255).astype(np.uint8)
    
    return Image.fromarray(highlighted)

def extract_segment(image, mask):
    """Extract just the segmented region."""
    img_array = np.array(image)
    
    # Convert to RGBA
    if img_array.shape[2] == 3:
        rgba_array = np.zeros((img_array.shape[0], img_array.shape[1], 4), dtype=np.uint8)
        rgba_array[:, :, :3] = img_array
        rgba_array[:, :, 3] = 255  # Full opacity
    else:
        rgba_array = img_array.copy()
    
    # Set non-masked pixels to transparent
    rgba_array[~mask] = [0, 0, 0, 0]
    
    return Image.fromarray(rgba_array, 'RGBA')

def demo_sam_workflow():
    """Demonstrate the SAM workflow."""
    print("🎯 SAM Integration Demo")
    print("=" * 50)
    
    # Create demo image
    print("1. Creating demo Mars surface image...")
    demo_img = create_demo_image()
    print(f"   ✅ Created {demo_img.size[0]}x{demo_img.size[1]} image")
    
    # Convert to base64 for web display
    print("2. Converting to base64 data URI...")
    data_uri = image_to_base64(demo_img)
    print(f"   ✅ Data URI length: {len(data_uri)} characters")
    
    # Simulate user interactions
    test_points = [
        (150, 150, "Crater center"),
        (265, 165, "Rock formation"),
        (80, 80, "Dust storm area")
    ]
    
    print("\n3. Simulating SAM segmentation at different points:")
    
    for i, (x, y, description) in enumerate(test_points):
        print(f"\n   Point {i+1}: {description} at ({x}, {y})")
        
        # Simulate SAM segmentation
        mask = simulate_sam_segment(demo_img, x, y)
        segment_area = np.sum(mask)
        print(f"   ✅ Segment area: {segment_area} pixels")
        
        # Create highlighted image
        highlighted = create_highlighted_image(demo_img, mask)
        highlighted_uri = image_to_base64(highlighted)
        print(f"   ✅ Highlighted image created")
        
        # Extract segment
        segment = extract_segment(demo_img, mask)
        segment_uri = image_to_base64(segment)
        print(f"   ✅ Segment extracted")
        
        # Save demo files
        highlighted.save(f"demo_highlighted_{i+1}.png")
        segment.save(f"demo_segment_{i+1}.png")
        print(f"   💾 Saved demo files: demo_highlighted_{i+1}.png, demo_segment_{i+1}.png")
    
    print("\n4. Integration with Dash app:")
    print("   ✅ SAM viewer modal will show interactive image")
    print("   ✅ Hover events will trigger segment preview")
    print("   ✅ Click events will select segments for saving")
    print("   ✅ Save button will export highlighted segments")
    
    print("\n🎉 Demo completed! Once PyTorch is installed, the real SAM model will:")
    print("   • Use actual AI segmentation instead of circular masks")
    print("   • Provide more accurate object boundaries")
    print("   • Handle complex shapes and textures")
    print("   • Work with real Mars surface imagery")

if __name__ == "__main__":
    demo_sam_workflow()
