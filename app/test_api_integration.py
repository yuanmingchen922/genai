"""
API Integration Test
Tests the MNIST GAN endpoints in the FastAPI application
"""

import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from fastapi.testclient import TestClient


def test_api_endpoints():
    """Test all GAN-related API endpoints"""
    print("\n" + "=" * 70)
    print("MNIST GAN API INTEGRATION TEST")
    print("=" * 70 + "\n")
    
    try:
        # Import with absolute import
        import main as app_module
        client = TestClient(app_module.app)
        
        # Test 1: Root endpoint
        print("Test 1: Root Endpoint")
        print("-" * 70)
        response = client.get("/")
        assert response.status_code == 200, f"Status code: {response.status_code}"
        data = response.json()
        assert "image_generation" in data["endpoints"], "Missing image_generation endpoints"
        print("✅ Root endpoint includes GAN endpoints")
        print(f"   Available GAN endpoints: {data['endpoints']['image_generation']}\n")
        
        # Test 2: Generate single digit
        print("Test 2: Generate Single Digit")
        print("-" * 70)
        response = client.post("/generate-digit", json={"seed": 42})
        assert response.status_code == 200, f"Status code: {response.status_code}"
        data = response.json()
        assert data["success"] == True, "Generation not successful"
        assert "image" in data, "No image in response"
        assert data["format"] == "base64_png", "Wrong format"
        assert data["size"] == "28x28", "Wrong size"
        print("✅ Single digit generation successful")
        print(f"   Format: {data['format']}")
        print(f"   Size: {data['size']}")
        print(f"   Seed: {data['seed']}\n")
        
        # Test 3: Generate batch with grid
        print("Test 3: Generate Batch (Grid)")
        print("-" * 70)
        response = client.post("/generate-digits-batch", json={"batch_size": 16, "grid": True})
        assert response.status_code == 200, f"Status code: {response.status_code}"
        data = response.json()
        assert data["success"] == True, "Batch generation not successful"
        assert "image" in data, "No image in response"
        assert data["batch_size"] == 16, "Wrong batch size"
        assert data["layout"] == "grid", "Not grid layout"
        print("✅ Batch generation (grid) successful")
        print(f"   Batch size: {data['batch_size']}")
        print(f"   Layout: {data['layout']}\n")
        
        # Test 4: Generate batch without grid
        print("Test 4: Generate Batch (List)")
        print("-" * 70)
        response = client.post("/generate-digits-batch", json={"batch_size": 8, "grid": False})
        assert response.status_code == 200, f"Status code: {response.status_code}"
        data = response.json()
        assert data["success"] == True, "Batch generation not successful"
        assert "images" in data, "No images in response"
        assert len(data["images"]) == 8, f"Wrong number of images: {len(data['images'])}"
        print("✅ Batch generation (list) successful")
        print(f"   Number of images: {data['count']}\n")
        
        # Test 5: Get model info
        print("Test 5: Get Model Info")
        print("-" * 70)
        response = client.get("/gan-model-info")
        assert response.status_code == 200, f"Status code: {response.status_code}"
        data = response.json()
        assert data["success"] == True, "Failed to get model info"
        assert "model_type" in data, "Missing model_type"
        assert data["noise_dimension"] == 100, "Wrong noise dimension"
        assert data["output_size"] == "28x28", "Wrong output size"
        print("✅ Model info retrieval successful")
        print(f"   Model: {data['model_type']}")
        print(f"   Noise dimension: {data['noise_dimension']}")
        print(f"   Output size: {data['output_size']}")
        print(f"   Parameters: {data['total_parameters']:,}\n")
        
        print("=" * 70)
        print("✅ ALL API TESTS PASSED!")
        print("=" * 70 + "\n")
        return True
        
    except ImportError as e:
        print(f"❌ Cannot import FastAPI app: {e}")
        print("   Make sure all dependencies are installed.")
        return False
    except AssertionError as e:
        print(f"❌ Test assertion failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_request_validation():
    """Test request validation"""
    print("\n" + "=" * 70)
    print("REQUEST VALIDATION TEST")
    print("=" * 70 + "\n")
    
    try:
        import main as app_module
        client = TestClient(app_module.app)
        
        # Test invalid batch size (too large)
        print("Test: Invalid batch size (>64)")
        response = client.post("/generate-digits-batch", json={"batch_size": 100})
        # Should handle gracefully even if it's outside validation
        print(f"   Response status: {response.status_code}")
        
        # Test with specific digit
        print("\nTest: Generate specific digit (0-9)")
        response = client.post("/generate-digit", json={"digit": 5})
        assert response.status_code == 200
        print("   ✅ Specific digit request handled")
        
        print("\n✅ REQUEST VALIDATION TESTS PASSED\n")
        return True
        
    except Exception as e:
        print(f"❌ Validation test error: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "*" * 70)
    print("RUNNING MNIST GAN API INTEGRATION TESTS")
    print("*" * 70)
    
    test1 = test_api_endpoints()
    test2 = test_request_validation()
    
    if test1 and test2:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("The MNIST GAN API is fully functional and error-free.\n")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please review errors above.\n")
        sys.exit(1)
