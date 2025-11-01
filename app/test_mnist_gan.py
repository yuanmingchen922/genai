"""
Test script for MNIST GAN API
Validates all GAN endpoints and ensures code accuracy
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from mnist_gan_model import MNISTGenerator, MNISTDiscriminator, MNISTGANGenerator, get_mnist_gan_generator
import numpy as np
from PIL import Image
import base64
from io import BytesIO


def test_generator_architecture():
    """Test 1: Validate Generator architecture"""
    print("=" * 70)
    print("Test 1: Generator Architecture")
    print("=" * 70)
    
    device = torch.device('cpu')
    generator = MNISTGenerator(noise_dim=100).to(device)
    
    # Test input/output shapes
    batch_size = 4
    noise = torch.randn(batch_size, 100)
    output = generator(noise)
    
    print(f"✓ Input shape: {noise.shape}")
    print(f"✓ Output shape: {output.shape}")
    
    # Validate shapes
    assert output.shape == (batch_size, 1, 28, 28), f"Expected (4, 1, 28, 28), got {output.shape}"
    print(f"✓ Output shape correct: (batch_size, 1, 28, 28)")
    
    # Validate output range (tanh activation)
    assert output.min() >= -1.0 and output.max() <= 1.0, f"Output range error: [{output.min():.3f}, {output.max():.3f}]"
    print(f"✓ Output range correct: [{output.min():.3f}, {output.max():.3f}]")
    
    # Count parameters
    total_params = sum(p.numel() for p in generator.parameters())
    print(f"✓ Total parameters: {total_params:,}")
    
    print("\n✅ Generator architecture test PASSED\n")
    return True


def test_discriminator_architecture():
    """Test 2: Validate Discriminator architecture"""
    print("=" * 70)
    print("Test 2: Discriminator Architecture")
    print("=" * 70)
    
    device = torch.device('cpu')
    discriminator = MNISTDiscriminator().to(device)
    
    # Test input/output shapes
    batch_size = 4
    images = torch.randn(batch_size, 1, 28, 28)
    output = discriminator(images)
    
    print(f"✓ Input shape: {images.shape}")
    print(f"✓ Output shape: {output.shape}")
    
    # Validate shapes
    assert output.shape == (batch_size, 1), f"Expected (4, 1), got {output.shape}"
    print(f"✓ Output shape correct: (batch_size, 1)")
    
    # Validate output range (sigmoid activation)
    assert output.min() >= 0.0 and output.max() <= 1.0, f"Output range error: [{output.min():.3f}, {output.max():.3f}]"
    print(f"✓ Output range correct: [{output.min():.3f}, {output.max():.3f}]")
    
    # Count parameters
    total_params = sum(p.numel() for p in discriminator.parameters())
    print(f"✓ Total parameters: {total_params:,}")
    
    print("\n✅ Discriminator architecture test PASSED\n")
    return True


def test_gan_forward_pass():
    """Test 3: Test forward pass compatibility"""
    print("=" * 70)
    print("Test 3: GAN Forward Pass Compatibility")
    print("=" * 70)
    
    device = torch.device('cpu')
    generator = MNISTGenerator(noise_dim=100).to(device)
    discriminator = MNISTDiscriminator().to(device)
    
    # Generate fake images
    noise = torch.randn(8, 100)
    fake_images = generator(noise)
    
    print(f"✓ Generated images shape: {fake_images.shape}")
    
    # Discriminate fake images
    predictions = discriminator(fake_images)
    
    print(f"✓ Discriminator predictions shape: {predictions.shape}")
    print(f"✓ Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    
    print("\n✅ Forward pass compatibility test PASSED\n")
    return True


def test_gan_generator_service():
    """Test 4: Test MNISTGANGenerator service class"""
    print("=" * 70)
    print("Test 4: MNISTGANGenerator Service")
    print("=" * 70)
    
    # Initialize service
    gan_service = MNISTGANGenerator(model_path=None, device='cpu')
    
    print(f"✓ Service initialized on device: {gan_service.device}")
    
    # Test tensor generation
    images_tensor = gan_service.generate_images(num_images=4, return_format='tensor')
    assert images_tensor.shape == (4, 1, 28, 28), f"Tensor shape error: {images_tensor.shape}"
    print(f"✓ Tensor generation: {images_tensor.shape}")
    
    # Test numpy generation
    images_numpy = gan_service.generate_images(num_images=2, return_format='numpy')
    assert images_numpy.shape == (2, 1, 28, 28), f"Numpy shape error: {images_numpy.shape}"
    print(f"✓ Numpy generation: {images_numpy.shape}")
    
    # Test PIL generation
    images_pil = gan_service.generate_images(num_images=2, return_format='pil')
    assert len(images_pil) == 2, f"PIL list length error: {len(images_pil)}"
    assert isinstance(images_pil[0], Image.Image), "Not a PIL Image"
    print(f"✓ PIL generation: {len(images_pil)} images")
    
    # Test base64 generation
    images_base64 = gan_service.generate_images(num_images=2, return_format='base64')
    assert len(images_base64) == 2, f"Base64 list length error: {len(images_base64)}"
    assert isinstance(images_base64[0], str), "Not a base64 string"
    print(f"✓ Base64 generation: {len(images_base64)} images")
    
    # Verify base64 can be decoded
    img_data = base64.b64decode(images_base64[0])
    img = Image.open(BytesIO(img_data))
    assert img.size == (28, 28), f"Decoded image size error: {img.size}"
    print(f"✓ Base64 decoding successful: {img.size}")
    
    # Test single digit generation
    digit_base64 = gan_service.generate_digit(seed=42)
    assert isinstance(digit_base64, str), "Digit generation failed"
    print(f"✓ Single digit generation successful")
    
    # Test batch generation with grid
    grid_image = gan_service.generate_batch(batch_size=16, grid=True)
    assert isinstance(grid_image, str), "Grid generation failed"
    print(f"✓ Batch grid generation successful")
    
    # Test batch generation without grid
    batch_images = gan_service.generate_batch(batch_size=8, grid=False)
    assert len(batch_images) == 8, f"Batch size error: {len(batch_images)}"
    print(f"✓ Batch list generation: {len(batch_images)} images")
    
    # Test model info
    model_info = gan_service.get_model_info()
    assert 'model_type' in model_info, "Model info incomplete"
    assert model_info['noise_dimension'] == 100, "Noise dimension mismatch"
    print(f"✓ Model info: {model_info['model_type']}")
    print(f"  - Noise dim: {model_info['noise_dimension']}")
    print(f"  - Output size: {model_info['output_size']}")
    print(f"  - Parameters: {model_info['total_parameters']:,}")
    
    print("\n✅ MNISTGANGenerator service test PASSED\n")
    return True


def test_global_instance():
    """Test 5: Test global instance getter"""
    print("=" * 70)
    print("Test 5: Global Instance Getter")
    print("=" * 70)
    
    # Get first instance
    gan1 = get_mnist_gan_generator()
    print(f"✓ First instance created: {type(gan1).__name__}")
    
    # Get second instance (should be same)
    gan2 = get_mnist_gan_generator()
    assert gan1 is gan2, "Global instance not properly cached"
    print(f"✓ Second call returns same instance")
    
    # Test it works
    image = gan1.generate_digit(seed=123)
    assert isinstance(image, str), "Global instance generation failed"
    print(f"✓ Global instance generation successful")
    
    print("\n✅ Global instance test PASSED\n")
    return True


def test_loss_compatibility():
    """Test 6: Test loss function compatibility"""
    print("=" * 70)
    print("Test 6: Loss Function Compatibility")
    print("=" * 70)
    
    import torch.nn as nn
    
    device = torch.device('cpu')
    generator = MNISTGenerator(noise_dim=100).to(device)
    discriminator = MNISTDiscriminator().to(device)
    criterion = nn.BCELoss()
    
    batch_size = 4
    noise = torch.randn(batch_size, 100)
    real_images = torch.randn(batch_size, 1, 28, 28)
    
    # Test discriminator loss
    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)
    
    fake_images = generator(noise)
    output_real = discriminator(real_images)
    output_fake = discriminator(fake_images.detach())
    
    loss_d_real = criterion(output_real, real_labels)
    loss_d_fake = criterion(output_fake, fake_labels)
    loss_d = loss_d_real + loss_d_fake
    
    print(f"✓ Discriminator loss calculated: {loss_d.item():.4f}")
    assert not torch.isnan(loss_d), "Discriminator loss is NaN"
    
    # Test generator loss
    output_fake_for_g = discriminator(fake_images)
    loss_g = criterion(output_fake_for_g, real_labels)
    
    print(f"✓ Generator loss calculated: {loss_g.item():.4f}")
    assert not torch.isnan(loss_g), "Generator loss is NaN"
    
    print("\n✅ Loss function compatibility test PASSED\n")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n")
    print("*" * 70)
    print("MNIST GAN API VALIDATION TEST SUITE")
    print("*" * 70)
    print("\n")
    
    tests = [
        ("Generator Architecture", test_generator_architecture),
        ("Discriminator Architecture", test_discriminator_architecture),
        ("GAN Forward Pass", test_gan_forward_pass),
        ("MNISTGANGenerator Service", test_gan_generator_service),
        ("Global Instance", test_global_instance),
        ("Loss Compatibility", test_loss_compatibility),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
        except Exception as e:
            print(f"❌ {test_name} FAILED with error: {str(e)}\n")
            results.append((test_name, False, str(e)))
    
    # Print summary
    print("\n")
    print("*" * 70)
    print("TEST SUMMARY")
    print("*" * 70)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    for test_name, result, error in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {test_name}")
        if error:
            print(f"         Error: {error}")
    
    print("\n" + "=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Code is accurate and error-free.\n")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.\n")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
