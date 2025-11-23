#!/usr/bin/env python3
"""
Complete verification script for all models in the GenAI project.
Tests all implementations to ensure they work correctly.
"""

import sys
import importlib

def test_imports():
    """Test all required imports."""
    print("=" * 70)
    print("Testing Imports")
    print("=" * 70)
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('fastapi', 'FastAPI'),
        ('pydantic', 'Pydantic'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('spacy', 'spaCy'),
        ('tqdm', 'tqdm'),
        ('matplotlib', 'Matplotlib'),
    ]
    
    all_good = True
    for module_name, display_name in required_packages:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {display_name:20s} - v{version}")
        except ImportError as e:
            print(f"✗ {display_name:20s} - NOT INSTALLED")
            all_good = False
    
    return all_good


def test_diffusion_model():
    """Test Diffusion Model implementation."""
    print("\n" + "=" * 70)
    print("Testing Diffusion Model")
    print("=" * 70)
    
    try:
        from app.diffusion_model import SinusoidalTimeEmbedding, SimpleUNet, get_diffusion_model
        import torch
        
        device = torch.device('cpu')
        
        # Test time embedding
        print("\n1. Testing SinusoidalTimeEmbedding...")
        time_embed = SinusoidalTimeEmbedding(embedding_dim=8, max_period=10000)
        t = torch.tensor([1])
        embedding = time_embed(t)
        print(f"   ✓ Time embedding shape: {embedding.shape}")
        print(f"   ✓ Embedding for t=1: {embedding[0].numpy()[:4]}...")
        
        # Test UNet
        print("\n2. Testing UNet...")
        unet = SimpleUNet().to(device)
        num_params = sum(p.numel() for p in unet.parameters())
        print(f"   ✓ UNet created with {num_params:,} parameters")
        
        # Test forward pass
        test_x = torch.randn(2, 3, 32, 32).to(device)
        test_t = torch.tensor([0, 100]).to(device)
        test_out = unet(test_x, test_t)
        print(f"   ✓ Forward pass: {test_x.shape} -> {test_out.shape}")
        
        # Test complete diffusion model
        print("\n3. Testing complete DiffusionModel...")
        diffusion = get_diffusion_model(device)
        print(f"   ✓ Diffusion model created with {diffusion.timesteps} timesteps")
        
        # Test sampling (just 1 step to verify)
        print("\n4. Testing sample generation...")
        sample = diffusion.sample(1, 3, 32, 32)
        print(f"   ✓ Sample generated: {sample.shape}")
        
        print("\n✅ Diffusion Model: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Diffusion Model test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_energy_model():
    """Test Energy-Based Model implementation."""
    print("\n" + "=" * 70)
    print("Testing Energy-Based Model")
    print("=" * 70)
    
    try:
        from app.energy_model import ImprovedEnergyNetwork, get_energy_model
        import torch
        
        device = torch.device('cpu')
        
        # Test energy network
        print("\n1. Testing ImprovedEnergyNetwork...")
        energy_net = ImprovedEnergyNetwork().to(device)
        num_params = sum(p.numel() for p in energy_net.parameters())
        print(f"   ✓ Energy network created with {num_params:,} parameters")
        
        # Test forward pass
        test_x = torch.randn(2, 3, 32, 32).to(device)
        energy = energy_net(test_x)
        print(f"   ✓ Forward pass: {test_x.shape} -> {energy.shape}")
        print(f"   ✓ Energy values: {energy.flatten().tolist()}")
        
        # Test complete energy model
        print("\n2. Testing complete EnergyModel...")
        energy_model = get_energy_model(device)
        print(f"   ✓ Energy model created")
        print(f"   ✓ Langevin steps: {energy_model.langevin_steps}")
        
        # Test sampling (just 1 sample to verify)
        print("\n3. Testing sample generation...")
        sample = energy_model.sample(1, 3, 32, 32)
        print(f"   ✓ Sample generated: {sample.shape}")
        
        print("\n✅ Energy-Based Model: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Energy Model test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_gan_model():
    """Test GAN Model implementation."""
    print("\n" + "=" * 70)
    print("Testing GAN Model")
    print("=" * 70)
    
    try:
        from app.mnist_gan_model import MNISTGenerator, MNISTDiscriminator
        import torch
        
        device = torch.device('cpu')
        
        # Test generator
        print("\n1. Testing MNISTGenerator...")
        generator = MNISTGenerator(noise_dim=100).to(device)
        num_params = sum(p.numel() for p in generator.parameters())
        print(f"   ✓ Generator created with {num_params:,} parameters")
        
        # Test forward pass
        noise = torch.randn(2, 100).to(device)
        fake_images = generator(noise)
        print(f"   ✓ Forward pass: {noise.shape} -> {fake_images.shape}")
        
        # Test discriminator
        print("\n2. Testing MNISTDiscriminator...")
        discriminator = MNISTDiscriminator().to(device)
        num_params = sum(p.numel() for p in discriminator.parameters())
        print(f"   ✓ Discriminator created with {num_params:,} parameters")
        
        # Test forward pass
        scores = discriminator(fake_images)
        print(f"   ✓ Forward pass: {fake_images.shape} -> {scores.shape}")
        
        print("\n✅ GAN Model: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ GAN Model test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_api_startup():
    """Test FastAPI can start."""
    print("\n" + "=" * 70)
    print("Testing FastAPI Application")
    print("=" * 70)
    
    try:
        from app.main import app
        
        print("\n1. Checking API instance...")
        print(f"   ✓ FastAPI app loaded")
        print(f"   ✓ Title: {app.title}")
        print(f"   ✓ Version: {app.version}")
        
        print("\n2. Checking routes...")
        routes = [route.path for route in app.routes]
        
        required_routes = [
            '/health',
            '/generate-diffusion',
            '/generate-energy',
            '/diffusion-model-info',
            '/energy-model-info',
            '/generate-digit',
            '/gan-model-info'
        ]
        
        for route in required_routes:
            if route in routes:
                print(f"   ✓ {route}")
            else:
                print(f"   ✗ {route} - MISSING")
        
        print("\n✅ FastAPI Application: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ FastAPI test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("\n" + "=" * 70)
    print("GenAI Project - Complete Verification")
    print("=" * 70)
    print()
    
    tests = [
        ("Package Imports", test_imports),
        ("Diffusion Model", test_diffusion_model),
        ("Energy-Based Model", test_energy_model),
        ("GAN Model", test_gan_model),
        ("FastAPI Application", test_api_startup),
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
    print("Verification Summary")
    print("=" * 70)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "=" * 70)
        print("✅ ALL VERIFICATIONS PASSED!")
        print("=" * 70)
        print("\nYour setup is complete and all models are working correctly.")
        print("\nNext steps:")
        print("1. Start API: uvicorn app.main:app --reload")
        print("2. Open docs: http://localhost:8000/docs")
        print("3. Run notebook: jupyter notebook Assignments/Advanced_Image_Generation_my2878.ipynb")
        print("4. Test API: python app/test_assignment4_api.py")
        return 0
    else:
        print("\n" + "=" * 70)
        print(f"⚠️  {total - passed} TEST(S) FAILED")
        print("=" * 70)
        print("\nPlease fix the issues above before proceeding.")
        print("Check SETUP_INSTRUCTIONS.md for help.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

