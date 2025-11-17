#!/usr/bin/env python3
"""
Test script for Docker deployment verification.
Tests all API endpoints to ensure proper functionality.
"""

import requests
import json
import time
import sys
from typing import Dict, Any


API_BASE_URL = "http://localhost:8000"


def test_health_endpoint() -> bool:
    """Test the health endpoint."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"Health Check: {data['status']}")
            return True
        else:
            print(f"Health check failed with status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Health check failed: {str(e)}")
        return False


def test_root_endpoint() -> bool:
    """Test the root endpoint."""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"Root endpoint: {data['message']}")
            print(f"Available endpoint categories: {list(data['endpoints'].keys())}")
            return True
        else:
            print(f"Root endpoint failed with status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Root endpoint failed: {str(e)}")
        return False


def test_gan_model_info() -> bool:
    """Test the GAN model info endpoint."""
    try:
        response = requests.get(f"{API_BASE_URL}/gan-model-info", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"GAN Model Info:")
            print(f"  - Generator parameters: {data.get('generator_params', 'N/A')}")
            print(f"  - Noise dimension: {data.get('noise_dim', 'N/A')}")
            return True
        else:
            print(f"GAN model info failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"GAN model info failed: {str(e)}")
        return False


def test_generate_digit() -> bool:
    """Test single digit generation."""
    try:
        payload = {
            "digit": 5,
            "seed": 42
        }
        response = requests.post(
            f"{API_BASE_URL}/generate-digit",
            json=payload,
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            print(f"Single digit generation successful:")
            print(f"  - Format: {data.get('format', 'N/A')}")
            print(f"  - Size: {data.get('size', 'N/A')}")
            print(f"  - Image data length: {len(data.get('image', ''))} characters")
            return True
        else:
            print(f"Generate digit failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"Generate digit failed: {str(e)}")
        return False


def test_generate_batch() -> bool:
    """Test batch digit generation."""
    try:
        payload = {
            "batch_size": 9,
            "grid": True
        }
        response = requests.post(
            f"{API_BASE_URL}/generate-digits-batch",
            json=payload,
            timeout=15
        )
        if response.status_code == 200:
            data = response.json()
            print(f"Batch generation successful:")
            print(f"  - Batch size: {data.get('batch_size', 'N/A')}")
            print(f"  - Layout: {data.get('layout', 'N/A')}")
            print(f"  - Image data length: {len(data.get('image', ''))} characters")
            return True
        else:
            print(f"Batch generation failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"Batch generation failed: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Docker Deployment Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        ("Health Check", test_health_endpoint),
        ("Root Endpoint", test_root_endpoint),
        ("GAN Model Info", test_gan_model_info),
        ("Generate Single Digit", test_generate_digit),
        ("Generate Batch", test_generate_batch),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\nTesting: {test_name}")
        print("-" * 60)
        try:
            result = test_func()
            results[test_name] = result
            status = "PASSED" if result else "FAILED"
            print(f"Status: {status}")
        except Exception as e:
            print(f"Status: FAILED (Exception: {str(e)})")
            results[test_name] = False
        print()
    
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nAll tests passed! Docker deployment is successful.")
        sys.exit(0)
    else:
        print(f"\n{total - passed} test(s) failed. Please check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    # Wait a moment for the server to be ready
    print("Waiting for API to be ready...")
    time.sleep(2)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        sys.exit(1)

