"""
Test script for Assignment 4 API endpoints
Tests Diffusion and Energy Model endpoints
"""

import requests
import json
import base64
from PIL import Image
from io import BytesIO


API_BASE_URL = "http://localhost:8000"


def test_diffusion_generation():
    """Test Diffusion Model image generation."""
    print("\n" + "=" * 70)
    print("Testing Diffusion Model Generation")
    print("=" * 70)
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate-diffusion",
            json={"num_samples": 8, "seed": 42},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Diffusion generation successful")
            print(f"  - Model: {data.get('model')}")
            print(f"  - Samples: {data.get('num_samples')}")
            print(f"  - Image size: {data.get('image_size')}")
            print(f"  - Image data length: {len(data.get('image', ''))} characters")
            
            # Save image
            img_data = base64.b64decode(data['image'])
            img = Image.open(BytesIO(img_data))
            img.save('diffusion_test_output.png')
            print(f"  - Saved to: diffusion_test_output.png")
            
            return True
        else:
            print(f"✗ Diffusion generation failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Diffusion generation error: {str(e)}")
        return False


def test_energy_generation():
    """Test Energy Model image generation."""
    print("\n" + "=" * 70)
    print("Testing Energy-Based Model Generation")
    print("=" * 70)
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate-energy",
            json={"num_samples": 8, "langevin_steps": 60, "seed": 42},
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Energy generation successful")
            print(f"  - Model: {data.get('model')}")
            print(f"  - Samples: {data.get('num_samples')}")
            print(f"  - Langevin steps: {data.get('langevin_steps')}")
            print(f"  - Image size: {data.get('image_size')}")
            print(f"  - Image data length: {len(data.get('image', ''))} characters")
            
            # Save image
            img_data = base64.b64decode(data['image'])
            img = Image.open(BytesIO(img_data))
            img.save('energy_test_output.png')
            print(f"  - Saved to: energy_test_output.png")
            
            return True
        else:
            print(f"✗ Energy generation failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Energy generation error: {str(e)}")
        return False


def test_diffusion_model_info():
    """Test Diffusion Model info endpoint."""
    print("\n" + "=" * 70)
    print("Testing Diffusion Model Info")
    print("=" * 70)
    
    try:
        response = requests.get(f"{API_BASE_URL}/diffusion-model-info", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Diffusion model info retrieved")
            print(f"  - Model type: {data.get('model_type')}")
            print(f"  - Architecture: {data.get('architecture')}")
            print(f"  - Parameters: {data.get('parameters'):,}")
            print(f"  - Timesteps: {data.get('timesteps')}")
            print(f"  - Device: {data.get('device')}")
            return True
        else:
            print(f"✗ Failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


def test_energy_model_info():
    """Test Energy Model info endpoint."""
    print("\n" + "=" * 70)
    print("Testing Energy-Based Model Info")
    print("=" * 70)
    
    try:
        response = requests.get(f"{API_BASE_URL}/energy-model-info", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Energy model info retrieved")
            print(f"  - Model type: {data.get('model_type')}")
            print(f"  - Architecture: {data.get('architecture')}")
            print(f"  - Parameters: {data.get('parameters'):,}")
            print(f"  - Sampling method: {data.get('sampling_method')}")
            print(f"  - Langevin steps: {data.get('langevin_steps')}")
            print(f"  - Device: {data.get('device')}")
            return True
        else:
            print(f"✗ Failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


def test_root_endpoint():
    """Test root endpoint to verify new endpoints are listed."""
    print("\n" + "=" * 70)
    print("Testing Root Endpoint")
    print("=" * 70)
    
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Root endpoint accessible")
            print(f"  - API version: {data.get('version', 'N/A')}")
            
            if 'endpoints' in data and 'image_generation' in data['endpoints']:
                endpoints = data['endpoints']['image_generation']
                print(f"  - Image generation endpoints:")
                for ep in endpoints:
                    print(f"    - {ep}")
                
                # Check for new endpoints
                required = ['/generate-diffusion', '/generate-energy', 
                           '/diffusion-model-info', '/energy-model-info']
                missing = [ep for ep in required if ep not in endpoints]
                
                if missing:
                    print(f"  ⚠ Missing endpoints: {missing}")
                    return False
                else:
                    print(f"  ✓ All Assignment 4 endpoints present")
                    return True
            else:
                print(f"  ✗ Endpoint structure unexpected")
                return False
        else:
            print(f"✗ Failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Assignment 4 API Test Suite")
    print("Testing Diffusion and Energy Model Endpoints")
    print("=" * 70)
    print(f"\nAPI Base URL: {API_BASE_URL}")
    print("Make sure the API server is running!")
    print()
    
    tests = [
        ("Root Endpoint", test_root_endpoint),
        ("Diffusion Model Info", test_diffusion_model_info),
        ("Energy Model Info", test_energy_model_info),
        ("Diffusion Generation", test_diffusion_generation),
        ("Energy Generation", test_energy_generation),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {str(e)}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Assignment 4 API integration successful.")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

