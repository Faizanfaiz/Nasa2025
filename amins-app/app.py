from flask import Flask, render_template, request, Response, send_file
import requests
from urllib.parse import urlparse
import os
import base64
from PIL import Image
import io
import struct
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/proxy/<path:url>')
def proxy_image(url):
    """Proxy NASA images to bypass CORS restrictions"""
    try:
        # Reconstruct the full URL
        full_url = f"https://{url}"
        print(f"Fetching image from: {full_url}")  # Debug log
        
        # Fetch the image
        response = requests.get(full_url, stream=True, timeout=30)
        
        if response.status_code == 200:
            # Return the image with proper headers
            return Response(
                response.content,
                status=200,
                headers={
                    'Content-Type': response.headers.get('content-type', 'image/jpeg'),
                    'Access-Control-Allow-Origin': '*',
                    'Cache-Control': 'public, max-age=3600'
                }
            )
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return f"Image not found: {response.status_code}", response.status_code
            
    except Exception as e:
        print(f"Exception: {str(e)}")
        return f"Error loading image: {str(e)}", 500

def parse_nasa_img_file(img_data, filename):
    """Parse NASA .img files with various format detection"""
    try:
        data_len = len(img_data)
        print(f"File size: {data_len} bytes")
        
        # Method 1: Analyze file structure
        # Check for common file signatures
        file_signature = img_data[:16]
        print(f"File signature (hex): {file_signature.hex()}")
        
        # Check for text headers
        header_text = img_data[:2000].decode('ascii', errors='ignore')
        print(f"Header text preview: {header_text[:200]}...")
        
        # Method 2: Try as raw binary data with more dimensions
        common_dims = [
            (1024, 1024), (512, 512), (256, 256), (2048, 2048),
            (1024, 512), (512, 1024), (2048, 1024), (1024, 2048),
            (4096, 4096), (8192, 8192), (128, 128), (64, 64),
            (1024, 256), (256, 1024), (2048, 512), (512, 2048),
            (4096, 2048), (2048, 4096)
        ]
        
        for width, height in common_dims:
            expected_size = width * height
            if data_len == expected_size or data_len == expected_size * 2 or data_len == expected_size * 4:
                try:
                    print(f"Trying dimensions {width}x{height} (expected: {expected_size}, actual: {data_len})")
                    
                    # Try as 8-bit grayscale
                    if data_len == expected_size:
                        img_array = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width))
                    # Try as 16-bit grayscale
                    elif data_len == expected_size * 2:
                        img_array = np.frombuffer(img_data, dtype=np.uint16).reshape((height, width))
                        # Scale 16-bit to 8-bit for display
                        img_array = (img_array / 256).astype(np.uint8)
                    # Try as 32-bit grayscale
                    elif data_len == expected_size * 4:
                        img_array = np.frombuffer(img_data, dtype=np.uint32).reshape((height, width))
                        # Scale 32-bit to 8-bit for display
                        img_array = (img_array / 16777216).astype(np.uint8)
                    
                    # Check if the data looks like an image (not all zeros or all same value)
                    unique_values = len(np.unique(img_array))
                    if unique_values > 10:  # Has some variation
                        # Convert to PIL Image
                        img = Image.fromarray(img_array, mode='L')
                        
                        # Apply contrast enhancement for better visibility
                        img_array = np.array(img)
                        img_array = np.clip(img_array * 1.5, 0, 255).astype(np.uint8)
                        img = Image.fromarray(img_array, mode='L')
                        
                        return img, f"Raw binary data ({width}x{height}, {data_len} bytes, {unique_values} unique values)"
                    
                except Exception as e:
                    print(f"Failed with {width}x{height}: {e}")
                    continue
        
        # Method 3: Try as PDS3 (Planetary Data System) format
        if 'pds_version_id' in header_text.lower() or 'pds3' in header_text.lower():
            print("Detected PDS3 format")
            lines = header_text.split('\n')
            width, height = None, None
            record_bytes = None
            header_end = None
            
            # Parse PDS3 header
            for i, line in enumerate(lines):
                line = line.strip()
                if 'lines' in line.lower() and '=' in line:
                    try:
                        # Handle different PDS3 formats
                        value = line.split('=')[1].strip().strip('"')
                        if '(' in value and ')' in value:
                            # Format like LINES = (1024) or LINES = (1024, 1)
                            height = int(value.split('(')[1].split(',')[0].strip())
                        else:
                            height = int(value)
                        print(f"Found LINES: {height}")
                    except:
                        pass
                if 'samples' in line.lower() and '=' in line:
                    try:
                        # Handle different PDS3 formats
                        value = line.split('=')[1].strip().strip('"')
                        if '(' in value and ')' in value:
                            # Format like SAMPLES = (1024) or SAMPLES = (1024, 1)
                            width = int(value.split('(')[1].split(',')[0].strip())
                        else:
                            width = int(value)
                        print(f"Found SAMPLES: {width}")
                    except:
                        pass
                if 'record_bytes' in line.lower() and '=' in line:
                    try:
                        record_bytes = int(line.split('=')[1].strip().strip('"'))
                        print(f"Found RECORD_BYTES: {record_bytes}")
                    except:
                        pass
                if 'end' in line.lower() and 'object' in line.lower():
                    header_end = i
                    break
            
            # Find the actual end of header (look for END statement)
            if header_end is None:
                for i, line in enumerate(lines):
                    if line.strip().upper() == 'END':
                        header_end = i
                        break
            
            if header_end is not None:
                # Calculate header size
                header_text_actual = '\n'.join(lines[:header_end + 1])
                header_size = len(header_text_actual.encode('ascii'))
                print(f"Header size: {header_size} bytes")
                
                # Try to find the actual data start (might need padding)
                data_start = header_size
                if record_bytes:
                    # Align to record boundary
                    data_start = ((header_size + record_bytes - 1) // record_bytes) * record_bytes
                    print(f"Aligned to record boundary: {data_start}")
                
                img_data_clean = img_data[data_start:]
                print(f"Data size after header: {len(img_data_clean)} bytes")
                
                if width and height:
                    expected_size = width * height
                    print(f"Expected data size: {expected_size} bytes")
                    
                    if len(img_data_clean) >= expected_size:
                        # Try different data types
                        for dtype in [np.uint8, np.uint16, np.uint32]:
                            try:
                                if len(img_data_clean) >= expected_size * np.dtype(dtype).itemsize:
                                    img_array = np.frombuffer(img_data_clean[:expected_size * np.dtype(dtype).itemsize], dtype=dtype).reshape((height, width))
                                    
                                    # Normalize to 8-bit for display
                                    if dtype != np.uint8:
                                        img_array = ((img_array - img_array.min()) * 255 / (img_array.max() - img_array.min())).astype(np.uint8)
                                    
                                    unique_values = len(np.unique(img_array))
                                    if unique_values > 10:
                                        img = Image.fromarray(img_array, mode='L')
                                        return img, f"PDS3 format ({width}x{height}, {dtype})"
                            except Exception as e:
                                print(f"Failed with {dtype}: {e}")
                                continue
                
                # If specific dimensions not found, try common MRO MARCI dimensions
                print("Trying common MRO MARCI dimensions...")
                # MRO MARCI files are typically square or rectangular
                mro_dims = [(1024, 1024), (512, 512), (2048, 1024), (1024, 2048), (2048, 2048), (1024, 512), (512, 1024)]
                for w, h in mro_dims:
                    expected_size = w * h
                    if len(img_data_clean) >= expected_size:
                        try:
                            img_array = np.frombuffer(img_data_clean[:expected_size], dtype=np.uint8).reshape((h, w))
                            unique_values = len(np.unique(img_array))
                            if unique_values > 10:
                                img = Image.fromarray(img_array, mode='L')
                                return img, f"PDS3 MRO MARCI format ({w}x{h})"
                        except:
                            continue
                
                # If still no luck, try swapping width/height
                print("Trying swapped dimensions...")
                for h, w in mro_dims:
                    expected_size = w * h
                    if len(img_data_clean) >= expected_size:
                        try:
                            img_array = np.frombuffer(img_data_clean[:expected_size], dtype=np.uint8).reshape((h, w))
                            unique_values = len(np.unique(img_array))
                            if unique_values > 10:
                                img = Image.fromarray(img_array, mode='L')
                                return img, f"PDS3 MRO MARCI format ({w}x{h}) - swapped"
                        except:
                            continue
        
        # Method 4: Try different data types and orientations
        print("Trying different data types...")
        for dtype in [np.uint8, np.uint16, np.uint32, np.int16, np.int32]:
            try:
                if data_len % np.dtype(dtype).itemsize == 0:
                    elements = data_len // np.dtype(dtype).itemsize
                    sqrt_elements = int(np.sqrt(elements))
                    if sqrt_elements * sqrt_elements == elements:
                        img_array = np.frombuffer(img_data, dtype=dtype).reshape((sqrt_elements, sqrt_elements))
                        # Normalize to 8-bit
                        if dtype != np.uint8:
                            img_array = ((img_array - img_array.min()) * 255 / (img_array.max() - img_array.min())).astype(np.uint8)
                        
                        unique_values = len(np.unique(img_array))
                        if unique_values > 10:
                            img = Image.fromarray(img_array, mode='L')
                            return img, f"Auto-detected {dtype} format ({sqrt_elements}x{sqrt_elements})"
            except:
                continue
        
        # Method 5: Try as MRO MARCI specific format
        print("Trying MRO MARCI specific parsing...")
        # MRO MARCI files are typically 1024x1024 or 512x512
        # File size is 73730048 bytes, which is 73730048 / 1024 / 1024 = ~70MB
        # This suggests it might be a multi-band or multi-frame image
        
        # Try to find header end by looking for specific patterns
        header_end_markers = [b'END', b'END_OBJECT', b'END_GROUP']
        header_end_pos = 0
        
        for marker in header_end_markers:
            pos = img_data.find(marker)
            if pos != -1:
                header_end_pos = pos + len(marker)
                break
        
        if header_end_pos > 0:
            print(f"Found header end at position: {header_end_pos}")
            img_data_clean = img_data[header_end_pos:]
            
            # Try common MRO MARCI dimensions
            mro_dims = [(1024, 1024), (512, 512), (2048, 1024), (1024, 2048), (2048, 2048)]
            for w, h in mro_dims:
                expected_size = w * h
                if len(img_data_clean) >= expected_size:
                    try:
                        img_array = np.frombuffer(img_data_clean[:expected_size], dtype=np.uint8).reshape((h, w))
                        unique_values = len(np.unique(img_array))
                        if unique_values > 10:
                            img = Image.fromarray(img_array, mode='L')
                            return img, f"MRO MARCI format ({w}x{h})"
                    except:
                        continue
                
                # Try 16-bit
                expected_size_16 = w * h * 2
                if len(img_data_clean) >= expected_size_16:
                    try:
                        img_array = np.frombuffer(img_data_clean[:expected_size_16], dtype=np.uint16).reshape((h, w))
                        # Scale to 8-bit
                        img_array = ((img_array - img_array.min()) * 255 / (img_array.max() - img_array.min())).astype(np.uint8)
                        unique_values = len(np.unique(img_array))
                        if unique_values > 10:
                            img = Image.fromarray(img_array, mode='L')
                            return img, f"MRO MARCI 16-bit format ({w}x{h})"
                    except:
                        continue
        
        # Method 6: Try as rectangular images
        print("Trying rectangular dimensions...")
        for width in range(100, min(5000, data_len), 100):
            if data_len % width == 0:
                height = data_len // width
                if 100 <= height <= 5000:
                    try:
                        img_array = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width))
                        unique_values = len(np.unique(img_array))
                        if unique_values > 10:
                            img = Image.fromarray(img_array, mode='L')
                            return img, f"Rectangular format ({width}x{height})"
                    except:
                        continue
        
        # If all methods fail, return detailed analysis
        analysis = f"""
        File Analysis:
        - Size: {data_len} bytes
        - Signature: {file_signature.hex()}
        - Header: {header_text[:100]}...
        - Possible factors: {[i for i in range(1, min(100, data_len)) if data_len % i == 0][:10]}
        """
        
        return None, f"Unable to parse as known NASA .img format. {analysis}"
        
    except Exception as e:
        return None, f"Error parsing file: {str(e)}"

@app.route('/upload_img', methods=['POST'])
def upload_img_file():
    """Handle .img file uploads and convert to web-viewable format"""
    try:
        if 'file' not in request.files:
            return {'error': 'No file uploaded'}, 400
        
        file = request.files['file']
        if file.filename == '':
            return {'error': 'No file selected'}, 400
        
        if file and file.filename.lower().endswith('.img'):
            # Read the .img file
            img_data = file.read()
            
            # First try PIL (for standard image formats)
            try:
                img = Image.open(io.BytesIO(img_data))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if too large
                max_size = 2048
                if img.width > max_size or img.height > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                # Convert to base64 for web display
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                
                return {
                    'success': True,
                    'image_data': f'data:image/png;base64,{img_base64}',
                    'width': img.width,
                    'height': img.height,
                    'filename': file.filename,
                    'format': 'Standard image format'
                }
                
            except Exception:
                # Try NASA-specific parsers
                img, format_info = parse_nasa_img_file(img_data, file.filename)
                
                if img:
                    # Resize if too large
                    max_size = 2048
                    if img.width > max_size or img.height > max_size:
                        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    
                    # Convert to base64 for web display
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format='PNG')
                    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                    
                    return {
                        'success': True,
                        'image_data': f'data:image/png;base64,{img_base64}',
                        'width': img.width,
                        'height': img.height,
                        'filename': file.filename,
                        'format': format_info
                    }
                else:
                    return {
                        'success': True,
                        'message': f'IMG file uploaded but {format_info}',
                        'filename': file.filename,
                        'size': len(img_data),
                        'format': 'Unknown NASA format'
                    }
        else:
            return {'error': 'Please upload a .img file'}, 400
            
    except Exception as e:
        return {'error': f'Error processing file: {str(e)}'}, 500

@app.route('/img_viewer')
def img_viewer():
    """Page for viewing .img files"""
    return render_template('img_viewer.html')

@app.route('/openseadragon_viewer')
def openseadragon_viewer():
    """Page for viewing .png files with OpenSeadragon"""
    return render_template('openseadragon_viewer.html')

@app.route('/process_large_image', methods=['POST'])
def process_large_image():
    """Process large images for OpenSeadragon viewing"""
    try:
        if 'file' not in request.files:
            return {'error': 'No file uploaded'}, 400
        
        file = request.files['file']
        if file.filename == '':
            return {'error': 'No file selected'}, 400
        
        print(f"Processing file: {file.filename}")
        print(f"File type: {file.content_type}")
        
        # Check both filename extension and MIME type
        is_image = False
        if file.filename:
            is_image = file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))
        
        if not is_image and file.content_type:
            is_image = file.content_type.startswith('image/')
        
        if file and is_image:
            try:
                # Read the image
                img_data = file.read()
                img = Image.open(io.BytesIO(img_data))
            except Exception as e:
                return {'error': f'Invalid image file: {str(e)}'}, 400
            
            # Get original dimensions
            original_width, original_height = img.size
            print(f"Original image size: {original_width}x{original_height}")
            
            # If image is too large, create a scaled version
            max_size = 4096  # Conservative limit to avoid canvas issues
            if original_width > max_size or original_height > max_size:
                # Calculate scaling factor
                scale_factor = min(max_size / original_width, max_size / original_height)
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)
                
                print(f"Scaling image to: {new_width}x{new_height} (factor: {scale_factor:.3f})")
                
                # Resize the image
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                # Even for smaller images, ensure they're not too large for canvas
                max_canvas_size = 16384  # 16K limit
                if original_width > max_canvas_size or original_height > max_canvas_size:
                    scale_factor = min(max_canvas_size / original_width, max_canvas_size / original_height)
                    new_width = int(original_width * scale_factor)
                    new_height = int(original_height * scale_factor)
                    
                    print(f"Preventive scaling to: {new_width}x{new_height} (factor: {scale_factor:.3f})")
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save as optimized PNG to a temporary file
            import tempfile
            import os
            
            # Create a unique filename
            import uuid
            unique_id = str(uuid.uuid4())
            filename = f"processed_{unique_id}.png"
            filepath = os.path.join(tempfile.gettempdir(), filename)
            
            # Save the image
            img.save(filepath, format='PNG', optimize=True)
            
            return {
                'success': True,
                'image_url': f'/processed_image/{filename}',
                'width': img.width,
                'height': img.height,
                'original_width': original_width,
                'original_height': original_height,
                'scaled': original_width > max_size or original_height > max_size,
                'filename': file.filename
            }
        else:
            # Try to process anyway, in case it's a valid image with wrong extension
            try:
                print(f"Trying to process file as image despite extension: {file.filename}")
                img_data = file.read()
                img = Image.open(io.BytesIO(img_data))
                
                # If we get here, it's a valid image
                print(f"Successfully opened as image: {file.filename}")
                
                # Get original dimensions
                original_width, original_height = img.size
                print(f"Original image size: {original_width}x{original_height}")
                
                # Continue with normal processing...
                # If image is too large, create a scaled version
                max_size = 4096  # Conservative limit to avoid canvas issues
                if original_width > max_size or original_height > max_size:
                    # Calculate scaling factor
                    scale_factor = min(max_size / original_width, max_size / original_height)
                    new_width = int(original_width * scale_factor)
                    new_height = int(original_height * scale_factor)
                    
                    print(f"Scaling image to: {new_width}x{new_height} (factor: {scale_factor:.3f})")
                    
                    # Resize the image
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                else:
                    # Even for smaller images, ensure they're not too large for canvas
                    max_canvas_size = 16384  # 16K limit
                    if original_width > max_canvas_size or original_height > max_canvas_size:
                        scale_factor = min(max_canvas_size / original_width, max_canvas_size / original_height)
                        new_width = int(original_width * scale_factor)
                        new_height = int(original_height * scale_factor)
                        
                        print(f"Preventive scaling to: {new_width}x{new_height} (factor: {scale_factor:.3f})")
                        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save as optimized PNG to a temporary file
                import tempfile
                import os
                
                # Create a unique filename
                import uuid
                unique_id = str(uuid.uuid4())
                filename = f"processed_{unique_id}.png"
                filepath = os.path.join(tempfile.gettempdir(), filename)
                
                # Save the image
                img.save(filepath, format='PNG', optimize=True)
                
                return {
                    'success': True,
                    'image_url': f'/processed_image/{filename}',
                    'width': img.width,
                    'height': img.height,
                    'original_width': original_width,
                    'original_height': original_height,
                    'scaled': original_width > max_size or original_height > max_size,
                    'filename': file.filename
                }
                
            except Exception as e:
                return {'error': f'Please upload an image file (PNG, JPG, TIFF). Received: {file.filename} (type: {file.content_type}). Error: {str(e)}'}, 400
            
    except Exception as e:
        return {'error': f'Error processing image: {str(e)}'}, 500

@app.route('/processed_image/<filename>')
def serve_processed_image(filename):
    """Serve processed images"""
    try:
        import tempfile
        import os
        
        filepath = os.path.join(tempfile.gettempdir(), filename)
        print(f"Serving image: {filepath}")
        
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            print(f"Image file size: {file_size} bytes")
            
            return send_file(
                filepath, 
                mimetype='image/png',
                as_attachment=False,
                download_name=filename
            )
        else:
            print(f"Image file not found: {filepath}")
            return 'Image not found', 404
            
    except Exception as e:
        print(f"Error serving image {filename}: {str(e)}")
        return f'Error serving image: {str(e)}', 500

@app.route('/test_image/<filename>')
def test_image(filename):
    """Test if image can be loaded"""
    try:
        import tempfile
        import os
        
        filepath = os.path.join(tempfile.gettempdir(), filename)
        
        if os.path.exists(filepath):
            # Try to open with PIL to verify it's a valid image
            img = Image.open(filepath)
            return {
                'success': True,
                'filename': filename,
                'size': os.path.getsize(filepath),
                'dimensions': f"{img.width}x{img.height}",
                'format': img.format,
                'mode': img.mode
            }
        else:
            return {'success': False, 'error': 'File not found'}, 404
            
    except Exception as e:
        return {'success': False, 'error': str(e)}, 500

@app.route('/try_dimensions', methods=['POST'])
def try_dimensions():
    """Try parsing .img file with manual dimensions"""
    try:
        if 'file' not in request.files:
            return {'error': 'No file uploaded'}, 400
        
        file = request.files['file']
        width = int(request.form.get('width', 1024))
        height = int(request.form.get('height', 1024))
        
        if file.filename == '':
            return {'error': 'No file selected'}, 400
        
        if file and file.filename.lower().endswith('.img'):
            img_data = file.read()
            
            # Enhanced header parsing for PDS3 files
            header_text = img_data[:5000].decode('ascii', errors='ignore')
            print(f"Header analysis for {file.filename}:")
            print(f"File size: {len(img_data)} bytes")
            
            # Look for specific PDS3 keywords that might give us clues
            lines = header_text.split('\n')
            record_bytes = None
            header_end_pos = 0
            
            for i, line in enumerate(lines):
                line = line.strip()
                if 'record_bytes' in line.lower() and '=' in line:
                    try:
                        record_bytes = int(line.split('=')[1].strip().strip('"'))
                        print(f"Found RECORD_BYTES: {record_bytes}")
                    except:
                        pass
                if line.upper() == 'END' or 'end_object' in line.lower():
                    header_end_pos = i
                    break
            
            # Calculate actual header end position
            if header_end_pos > 0:
                header_text_actual = '\n'.join(lines[:header_end_pos + 1])
                header_size = len(header_text_actual.encode('ascii'))
                print(f"Calculated header size: {header_size} bytes")
                
                # Align to record boundary if needed
                if record_bytes:
                    header_size = ((header_size + record_bytes - 1) // record_bytes) * record_bytes
                    print(f"Aligned to record boundary: {header_size} bytes")
            else:
                # Fallback: try to find END markers in binary
                for marker in [b'END', b'END_OBJECT', b'END_GROUP']:
                    pos = img_data.find(marker)
                    if pos != -1:
                        header_size = pos + len(marker)
                        print(f"Found END marker at: {header_size} bytes")
                        break
                else:
                    header_size = 2000  # Fallback
                    print(f"Using fallback header size: {header_size} bytes")
            
            img_data_clean = img_data[header_size:]
            print(f"Data size after header: {len(img_data_clean)} bytes")
            print(f"Trying dimensions: {width}x{height}")
            
            # Try different data types and byte orders
            for dtype in [np.uint8, np.uint16, np.uint32]:
                for byte_order in ['<', '>']:  # little-endian, big-endian
                    try:
                        expected_size = width * height * np.dtype(dtype).itemsize
                        if len(img_data_clean) >= expected_size:
                            # Try with specific byte order
                            dt = np.dtype(dtype).newbyteorder(byte_order)
                            img_array = np.frombuffer(img_data_clean[:expected_size], dtype=dt).reshape((height, width))
                            
                            # Check if this looks like a real image
                            unique_values = len(np.unique(img_array))
                            if unique_values > 50:  # More strict threshold
                                # Normalize to 8-bit for display
                                if dtype != np.uint8:
                                    img_array = ((img_array - img_array.min()) * 255 / (img_array.max() - img_array.min())).astype(np.uint8)
                                
                                img = Image.fromarray(img_array, mode='L')
                                
                                # Resize if too large
                                max_size = 2048
                                if img.width > max_size or img.height > max_size:
                                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                                
                                # Convert to base64 for web display
                                img_buffer = io.BytesIO()
                                img.save(img_buffer, format='PNG')
                                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                                
                                print(f"Success with {dtype} {byte_order} - {unique_values} unique values")
                                
                                return {
                                    'success': True,
                                    'image_data': f'data:image/png;base64,{img_base64}',
                                    'width': img.width,
                                    'height': img.height,
                                    'filename': file.filename,
                                    'format': f'Manual dimensions ({width}x{height}, {dtype} {byte_order})'
                                }
                    except Exception as e:
                        print(f"Failed with {dtype} {byte_order}: {e}")
                        continue
            
            # If still no luck, try some common NASA image dimensions
            print("Trying common NASA dimensions...")
            nasa_dims = [(1024, 1024), (512, 512), (2048, 1024), (1024, 2048), (256, 256), (128, 128)]
            for w, h in nasa_dims:
                for dtype in [np.uint8, np.uint16]:
                    try:
                        expected_size = w * h * np.dtype(dtype).itemsize
                        if len(img_data_clean) >= expected_size:
                            img_array = np.frombuffer(img_data_clean[:expected_size], dtype=dtype).reshape((h, w))
                            unique_values = len(np.unique(img_array))
                            if unique_values > 50:
                                if dtype != np.uint8:
                                    img_array = ((img_array - img_array.min()) * 255 / (img_array.max() - img_array.min())).astype(np.uint8)
                                
                                img = Image.fromarray(img_array, mode='L')
                                max_size = 2048
                                if img.width > max_size or img.height > max_size:
                                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                                
                                img_buffer = io.BytesIO()
                                img.save(img_buffer, format='PNG')
                                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                                
                                print(f"Success with NASA dims {w}x{h} {dtype}")
                                
                                return {
                                    'success': True,
                                    'image_data': f'data:image/png;base64,{img_base64}',
                                    'width': img.width,
                                    'height': img.height,
                                    'filename': file.filename,
                                    'format': f'NASA dimensions ({w}x{h}, {dtype})'
                                }
                    except:
                        continue
            
            return {'error': f'Could not parse with any dimensions. File may need specialized tools like GDAL with specific parameters.'}, 400
        else:
            return {'error': 'Please upload a .img file'}, 400
            
    except Exception as e:
        return {'error': f'Error trying dimensions: {str(e)}'}, 500

@app.route('/gdal_help', methods=['POST'])
def gdal_help():
    """Provide GDAL command suggestions for .img file"""
    try:
        if 'file' not in request.files:
            return {'error': 'No file uploaded'}, 400
        
        file = request.files['file']
        if file.filename == '':
            return {'error': 'No file selected'}, 400
        
        if file and file.filename.lower().endswith('.img'):
            img_data = file.read()
            header_text = img_data[:5000].decode('ascii', errors='ignore')
            
            # Analyze header for GDAL parameters
            lines = header_text.split('\n')
            gdal_params = []
            
            # Look for specific PDS3 parameters
            for line in lines:
                line = line.strip()
                if 'samples' in line.lower() and '=' in line:
                    try:
                        samples = int(line.split('=')[1].strip().strip('"').split('(')[1].split(',')[0].strip())
                        gdal_params.append(f"-co WIDTH={samples}")
                    except:
                        pass
                if 'lines' in line.lower() and '=' in line:
                    try:
                        lines_val = int(line.split('=')[1].strip().strip('"').split('(')[1].split(',')[0].strip())
                        gdal_params.append(f"-co HEIGHT={lines_val}")
                    except:
                        pass
                if 'sample_bits' in line.lower() and '=' in line:
                    try:
                        bits = int(line.split('=')[1].strip().strip('"'))
                        gdal_params.append(f"-co NBITS={bits}")
                    except:
                        pass
                if 'sample_type' in line.lower() and '=' in line:
                    sample_type = line.split('=')[1].strip().strip('"').lower()
                    if 'msb' in sample_type:
                        gdal_params.append("-co BYTEORDER=MSB")
                    elif 'lsb' in sample_type:
                        gdal_params.append("-co BYTEORDER=LSB")
            
            # Generate GDAL command suggestions
            gdal_commands = [
                f"gdal_translate {file.filename} output.png",
                f"gdal_translate -of PNG {file.filename} output.png",
                f"gdal_translate -of PNG -co COMPRESS=LZW {file.filename} output.png"
            ]
            
            if gdal_params:
                params_str = ' '.join(gdal_params)
                gdal_commands.append(f"gdal_translate -of PNG {params_str} {file.filename} output.png")
            
            # Add specific MRO MARCI suggestions
            gdal_commands.extend([
                f"gdal_translate -of PNG -co WIDTH=1024 -co HEIGHT=1024 {file.filename} output.png",
                f"gdal_translate -of PNG -co WIDTH=512 -co HEIGHT=512 {file.filename} output.png",
                f"gdal_translate -of PNG -co WIDTH=2048 -co HEIGHT=1024 {file.filename} output.png"
            ])
            
            return {
                'success': True,
                'filename': file.filename,
                'gdal_commands': gdal_commands,
                'detected_params': gdal_params,
                'header_analysis': header_text[:1000]
            }
        else:
            return {'error': 'Please upload a .img file'}, 400
            
    except Exception as e:
        return {'error': f'Error analyzing file: {str(e)}'}, 500

@app.route('/analyze_img', methods=['POST'])
def analyze_img_file():
    """Analyze .img file structure without parsing"""
    try:
        if 'file' not in request.files:
            return {'error': 'No file uploaded'}, 400
        
        file = request.files['file']
        if file.filename == '':
            return {'error': 'No file selected'}, 400
        
        if file and file.filename.lower().endswith('.img'):
            img_data = file.read()
            data_len = len(img_data)
            
            # Basic analysis
            file_signature = img_data[:16]
            header_text = img_data[:2000].decode('ascii', errors='ignore')
            
            # Find possible dimensions
            possible_dims = []
            for width in range(100, min(5000, data_len), 100):
                if data_len % width == 0:
                    height = data_len // width
                    if 100 <= height <= 5000:
                        possible_dims.append((width, height))
            
            # Check for common NASA formats
            is_pds = 'pds' in header_text.lower() or 'planetary' in header_text.lower()
            is_vicar = 'vicar' in header_text.lower()
            is_fits = 'fits' in header_text.lower()
            
            return {
                'success': True,
                'filename': file.filename,
                'size': data_len,
                'signature_hex': file_signature.hex(),
                'header_preview': header_text[:500],
                'possible_dimensions': possible_dims[:10],
                'is_pds': is_pds,
                'is_vicar': is_vicar,
                'is_fits': is_fits,
                'analysis': f"File size: {data_len} bytes, Signature: {file_signature.hex()}, Possible dims: {possible_dims[:5]}"
            }
        else:
            return {'error': 'Please upload a .img file'}, 400
            
    except Exception as e:
        return {'error': f'Error analyzing file: {str(e)}'}, 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

