# Mars Surface Explorer with SAM Integration

A comprehensive Mars surface exploration tool that combines NASA's Mars Global Surveyor (MGS) MOLA elevation data with Mars Reconnaissance Orbiter (MRO) imagery, enhanced with AI-powered segmentation using Meta's Segment Anything Model (SAM).

## Features

- **Interactive Mars Map**: Explore Mars using NASA's official MOLA elevation tiles
- **3D Terrain Visualization**: Real-time 3D terrain rendering with customizable views
- **MRO Data Overlays**: High-resolution imagery from Mars Reconnaissance Orbiter instruments (CTX, HiRISE)
- **AI-Powered Segmentation**: Interactive image analysis using SAM for terrain feature detection
- **Feature Annotation**: Save and manage points of interest with custom notes
- **Terrain Snapshots**: Capture 3D terrain views from any angle for analysis

## Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Nasa2025
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements
   ```
   
   **Note**: The requirements file is configured for CPU-only PyTorch installation. This ensures compatibility across different systems without requiring CUDA/GPU setup.

4. **Download SAM Model Checkpoint**:
   
   The application supports three SAM model variants. Choose one based on your needs:

   **Option A: ViT-B (Recommended - Fastest, Good Quality)**
   ```bash
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
   ```

   **Option B: ViT-L (Balanced Performance)**
   ```bash
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
   ```

   **Option C: ViT-H (Best Quality, Slower)**
   ```bash
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   ```

   **Note**: The current code is configured for ViT-B (`sam_vit_b_01ec64.pth`). To use other models, modify the `model_type` and `sam_checkpoint` variables in `sam_integration.py`.

## Usage

### Starting the Application

```bash
python v1.py
```

The application will start on `http://localhost:8050`

### Basic Navigation

1. **Map Exploration**: 
   - Use mouse to pan and zoom on the Mars map
   - Click anywhere to sample elevation data
   - Use the instrument dropdown to search for MRO overlays

2. **3D Terrain Analysis**:
   - Click "Load terrain patch" to generate 3D terrain
   - Rotate and zoom the 3D view as desired
   - Click "Capture terrain snapshot" to save the current view

3. **AI-Powered Segmentation**:
   - Click "View" on any saved feature
   - Choose "🔍 Interactive SAM Analysis" for AI segmentation
   - Click anywhere on the image to segment that area
   - Use "Save Highlighted Segment" to export results

### SAM Model Compatibility

The application is designed to work with all three SAM model variants:

- **ViT-B** (`sam_vit_b_01ec64.pth`): 91MB, Fastest, Good for most use cases
- **ViT-L** (`sam_vit_l_0b3195.pth`): 375MB, Balanced performance
- **ViT-H** (`sam_vit_h_4b8939.pth`): 2.4GB, Best quality, requires more resources

To switch models, edit `sam_integration.py`:
```python
model_type = "vit_b"  # Change to "vit_l" or "vit_h"
sam_checkpoint = "sam_vit_b_01ec64.pth"  # Change to corresponding checkpoint
```

## Data Sources

- **MOLA Elevation**: NASA Mars Global Surveyor MOLA global hillshade (463m/pixel)
- **MRO Imagery**: Mars Reconnaissance Orbiter Context Camera (CTX) and HiRISE data
- **Tile Services**: NASA Mars Trek WMTS services
- **Elevation Data**: NASA MOLA 128/64 merge elevation model

## Technical Details

### Architecture
- **Frontend**: Dash (Python web framework)
- **Mapping**: Dash Leaflet for interactive maps
- **3D Visualization**: Plotly for terrain rendering
- **AI Model**: Meta's Segment Anything Model (SAM)
- **Image Processing**: OpenCV and PIL

### Performance Notes
- SAM model loads on first use (may take 10-30 seconds)
- ViT-B model recommended for most users
- **CPU-only installation** (PyTorch CPU version included in requirements)
- Terrain patches are cached for better performance

## Troubleshooting

### Common Issues

1. **SAM Model Not Loading**:
   - Ensure checkpoint file is downloaded and in the project directory
   - Check file permissions and path
   - Verify PyTorch installation

2. **Memory Issues**:
   - Use ViT-B model instead of ViT-H for lower memory usage
   - Reduce terrain patch size
   - Close other applications

3. **Slow Performance**:
   - Use CPU-only PyTorch if GPU drivers are problematic
   - Reduce image resolution in SAM processing
   - Use smaller terrain patches

### Getting Help

- Check browser console for JavaScript errors
- Verify all dependencies are installed correctly
- Ensure Python version compatibility (3.8+)

## License

This project is part of NASA Space Apps Challenge 2025.

## Acknowledgments

- NASA for providing Mars data and imagery
- Meta AI for the Segment Anything Model
- The open-source community for the various libraries used