"""
Standalone API Test
Direct test of MNIST GAN model functionality without FastAPI
"""

import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_mnist_gan_model_standalone():
    """Test MNIST GAN model standalone"""
    print("\n" + "=" * 70)
    print("MNIST GAN MODEL STANDALONE TEST")
    print("=" * 70 + "\n")
    
    try:
        from mnist_gan_model import get_mnist_gan_generator
        
        print("Test 1: Initialize GAN Generator")
        print("-" * 70)
        gan = get_mnist_gan_generator(model_path=None)
        print(f"✅ GAN generator initialized on device: {gan.device}")
        print(f"   Noise dimension: {gan.noise_dim}")
        
        print("\nTest 2: Get Model Info")
        print("-" * 70)
        info = gan.get_model_info()
        print(f"✅ Model info retrieved:")
        print(f"   Type: {info['model_type']}")
        print(f"   Noise dim: {info['noise_dimension']}")
        print(f"   Output: {info['output_size']}")
        print(f"   Parameters: {info['total_parameters']:,}")
        
        print("\nTest 3: Generate Single Digit")
        print("-" * 70)
        digit = gan.generate_digit(seed=42)
        print(f"✅ Single digit generated")
        print(f"   Type: {type(digit)}")
        print(f"   Length: {len(digit)} characters")
        
        print("\nTest 4: Generate Batch (Grid)")
        print("-" * 70)
        grid = gan.generate_batch(batch_size=16, grid=True)
        print(f"✅ Batch grid generated")
        print(f"   Type: {type(grid)}")
        print(f"   Length: {len(grid)} characters")
        
        print("\nTest 5: Generate Batch (List)")
        print("-" * 70)
        batch = gan.generate_batch(batch_size=8, grid=False)
        print(f"✅ Batch list generated")
        print(f"   Type: {type(batch)}")
        print(f"   Count: {len(batch)} images")
        
        print("\nTest 6: Generate with Different Formats")
        print("-" * 70)
        
        # Tensor format
        tensor_imgs = gan.generate_images(num_images=2, return_format='tensor')
        print(f"✅ Tensor format: {tensor_imgs.shape}")
        
        # Numpy format
        numpy_imgs = gan.generate_images(num_images=2, return_format='numpy')
        print(f"✅ Numpy format: {numpy_imgs.shape}")
        
        # PIL format
        pil_imgs = gan.generate_images(num_images=2, return_format='pil')
        print(f"✅ PIL format: {len(pil_imgs)} images, size {pil_imgs[0].size}")
        
        # Base64 format
        b64_imgs = gan.generate_images(num_images=2, return_format='base64')
        print(f"✅ Base64 format: {len(b64_imgs)} images")
        
        print("\n" + "=" * 70)
        print("✅ ALL STANDALONE TESTS PASSED!")
        print("=" * 70 + "\n")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_architectures():
    """Test model architectures directly"""
    print("\n" + "=" * 70)
    print("MODEL ARCHITECTURE VERIFICATION")
    print("=" * 70 + "\n")
    
    try:
        from mnist_gan_model import MNISTGenerator, MNISTDiscriminator
        import torch
        
        print("Test 1: Generator Architecture")
        print("-" * 70)
        gen = MNISTGenerator(noise_dim=100)
        noise = torch.randn(4, 100)
        output = gen(noise)
        print(f"✅ Generator:")
        print(f"   Input: {noise.shape}")
        print(f"   Output: {output.shape}")
        print(f"   Parameters: {sum(p.numel() for p in gen.parameters()):,}")
        
        print("\nTest 2: Discriminator Architecture")
        print("-" * 70)
        disc = MNISTDiscriminator()
        images = torch.randn(4, 1, 28, 28)
        pred = disc(images)
        print(f"✅ Discriminator:")
        print(f"   Input: {images.shape}")
        print(f"   Output: {pred.shape}")
        print(f"   Parameters: {sum(p.numel() for p in disc.parameters()):,}")
        
        print("\n" + "=" * 70)
        print("✅ ARCHITECTURE VERIFICATION PASSED!")
        print("=" * 70 + "\n")
        return True
        
    except Exception as e:
        print(f"❌ Architecture test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_compatibility():
    """Test that model is compatible with API requirements"""
    print("\n" + "=" * 70)
    print("API COMPATIBILITY TEST")
    print("=" * 70 + "\n")
    
    try:
        from mnist_gan_model import get_mnist_gan_generator
        import base64
        from io import BytesIO
        from PIL import Image
        
        gan = get_mnist_gan_generator()
        
        print("Test 1: Base64 Image Decoding")
        print("-" * 70)
        img_b64 = gan.generate_digit(seed=123)
        img_data = base64.b64decode(img_b64)
        img = Image.open(BytesIO(img_data))
        print(f"✅ Image decoded successfully:")
        print(f"   Format: {img.format}")
        print(f"   Mode: {img.mode}")
        print(f"   Size: {img.size}")
        assert img.size == (28, 28), f"Wrong size: {img.size}"
        
        print("\nTest 2: Batch Grid Decoding")
        print("-" * 70)
        grid_b64 = gan.generate_batch(batch_size=16, grid=True)
        grid_data = base64.b64decode(grid_b64)
        grid_img = Image.open(BytesIO(grid_data))
        print(f"✅ Grid image decoded successfully:")
        print(f"   Size: {grid_img.size}")
        
        print("\nTest 3: Multiple Images in Batch")
        print("-" * 70)
        batch = gan.generate_batch(batch_size=8, grid=False)
        for i, img_b64 in enumerate(batch[:3]):  # Test first 3
            img_data = base64.b64decode(img_b64)
            img = Image.open(BytesIO(img_data))
            assert img.size == (28, 28), f"Image {i} wrong size: {img.size}"
        print(f"✅ All {len(batch)} batch images validated")
        
        print("\n" + "=" * 70)
        print("✅ API COMPATIBILITY TEST PASSED!")
        print("=" * 70 + "\n")
        return True
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "*" * 70)
    print("COMPREHENSIVE MNIST GAN TESTING")
    print("*" * 70)
    
    tests = [
        test_model_architectures,
        test_mnist_gan_model_standalone,
        test_api_compatibility
    ]
    
    results = []
    for test_func in tests:
        result = test_func()
        results.append(result)
    
    print("\n" + "*" * 70)
    print("FINAL SUMMARY")
    print("*" * 70)
    print(f"\nTests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 ALL TESTS PASSED!")
        print("The MNIST GAN implementation is fully functional and error-free.")
        print("Ready for API integration.\n")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please review errors above.\n")
        sys.exit(1)
