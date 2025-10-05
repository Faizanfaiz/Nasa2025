#!/usr/bin/env python3
"""Test script for SAM integration."""

import sys
import os

def test_sam_imports():
    """Test if SAM can be imported."""
    print("Testing SAM imports...")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"❌ PyTorch not available: {e}")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV not available: {e}")
        return False
    
    try:
        from segment_anything import sam_model_registry, SamPredictor
        print("✅ Segment Anything Model imported successfully")
    except ImportError as e:
        print(f"❌ SAM not available: {e}")
        return False
    
    return True

def test_sam_integration():
    """Test the SAM integration module."""
    print("\nTesting SAM integration module...")
    
    try:
        from sam_integration import SAM_AVAILABLE, initialize_sam_model
        print(f"✅ SAM integration module imported")
        print(f"✅ SAM_AVAILABLE: {SAM_AVAILABLE}")
        
        if SAM_AVAILABLE:
            print("Attempting to initialize SAM model...")
            # Note: This will fail without the checkpoint file, but that's expected
            result = initialize_sam_model()
            print(f"SAM initialization result: {result}")
        
    except ImportError as e:
        print(f"❌ SAM integration module not available: {e}")
        return False
    
    return True

def test_sam_viewer():
    """Test the SAM viewer module."""
    print("\nTesting SAM viewer module...")
    
    try:
        from sam_viewer import SAM_VIEWER_AVAILABLE
        print(f"✅ SAM viewer module imported")
        print(f"✅ SAM_VIEWER_AVAILABLE: {SAM_VIEWER_AVAILABLE}")
        
    except ImportError as e:
        print(f"❌ SAM viewer module not available: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🧪 Testing SAM Integration")
    print("=" * 50)
    
    tests = [
        test_sam_imports,
        test_sam_integration,
        test_sam_viewer
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"✅ Passed: {sum(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}")
    
    if all(results):
        print("🎉 All tests passed! SAM integration is ready.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

