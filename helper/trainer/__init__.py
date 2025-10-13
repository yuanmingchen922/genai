"""
Trainer Module
Provides training functions for neural network models.
"""

import torch
import torch.nn as nn
from tqdm import tqdm


def train_model(model, train_loader, criterion, optimizer, device, epochs=10, verbose=True):
    """
    Train a neural network model.
    
    Args:
        model (nn.Module): PyTorch model to train
        train_loader (DataLoader): DataLoader for training data
        criterion: Loss function (e.g., nn.CrossEntropyLoss())
        optimizer: Optimizer (e.g., torch.optim.Adam())
        device (str or torch.device): Device to train on ('cuda' or 'cpu')
        epochs (int): Number of training epochs. Default: 10
        verbose (bool): If True, print training progress. Default: True
    
    Returns:
        dict: Dictionary containing training history with keys:
              - 'train_loss': List of average loss per epoch
              - 'train_accuracy': List of accuracy per epoch
    
    Example:
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> model = SimpleNN([784, 128, 10]).to(device)
        >>> criterion = nn.CrossEntropyLoss()
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> history = train_model(model, train_loader, criterion, optimizer, device, epochs=5)
    """
    model.to(device)
    model.train()
    
    history = {
        'train_loss': [],
        'train_accuracy': []
    }
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        if verbose:
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        else:
            progress_bar = train_loader
        
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            if verbose:
                progress_bar.set_postfix({
                    'loss': running_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100. * correct / total
        
        history['train_loss'].append(epoch_loss)
        history['train_accuracy'].append(epoch_accuracy)
        
        if verbose:
            print(f'Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
    
    return history


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): PyTorch model
        train_loader (DataLoader): DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device (str or torch.device): Device to train on
    
    Returns:
        tuple: (average_loss, accuracy) for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def vae_loss_function(recon_x, x, mu, logvar, beta=500):
    """
    VAE loss function combining reconstruction loss and KL divergence.
    
    Args:
        recon_x (torch.Tensor): Reconstructed images
        x (torch.Tensor): Original images
        mu (torch.Tensor): Mean of latent distribution
        logvar (torch.Tensor): Log variance of latent distribution
        beta (float): Weight for reconstruction loss. Default: 500
    
    Returns:
        torch.Tensor: Total loss
    """
    # Reconstruction loss (Binary Cross Entropy)
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL Divergence: measures how much the learned distribution deviates from standard normal
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return beta * BCE + KLD


def train_vae_model(model, train_loader, optimizer, device, epochs=10, beta=500, verbose=True):
    """
    Train a Variational Autoencoder model.
    
    Args:
        model (nn.Module): VAE model to train
        train_loader (DataLoader): DataLoader for training data
        optimizer: Optimizer (e.g., torch.optim.Adam())
        device (str or torch.device): Device to train on ('cuda' or 'cpu')
        epochs (int): Number of training epochs. Default: 10
        beta (float): Weight for reconstruction loss in VAE loss. Default: 500
        verbose (bool): If True, print training progress. Default: True
    
    Returns:
        dict: Dictionary containing training history with keys:
              - 'train_loss': List of average loss per epoch
              - 'recon_loss': List of reconstruction loss per epoch
              - 'kl_loss': List of KL divergence per epoch
    
    Example:
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> vae = VAE(latent_dim=2).to(device)
        >>> optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
        >>> history = train_vae_model(vae, train_loader, optimizer, device, epochs=10)
    """
    model.to(device)
    model.train()
    
    history = {
        'train_loss': [],
        'recon_loss': [],
        'kl_loss': []
    }
    
    for epoch in range(epochs):
        running_loss = 0.0
        running_recon_loss = 0.0
        running_kl_loss = 0.0
        
        # Progress bar
        if verbose:
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        else:
            progress_bar = train_loader
        
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            # Move data to device
            inputs = inputs.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            recon, mu, logvar = model(inputs)
            
            # Calculate loss
            loss = vae_loss_function(recon, inputs, mu, logvar, beta)
            
            # Calculate individual components for monitoring
            recon_loss = nn.functional.binary_cross_entropy(recon, inputs, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            running_recon_loss += recon_loss.item()
            running_kl_loss += kl_loss.item()
            
            # Update progress bar
            if verbose:
                progress_bar.set_postfix({
                    'loss': loss.item() / inputs.size(0),
                    'recon': recon_loss.item() / inputs.size(0),
                    'kl': kl_loss.item() / inputs.size(0)
                })
        
        # Calculate epoch metrics
        num_samples = len(train_loader.dataset)
        epoch_loss = running_loss / num_samples
        epoch_recon_loss = running_recon_loss / num_samples
        epoch_kl_loss = running_kl_loss / num_samples
        
        history['train_loss'].append(epoch_loss)
        history['recon_loss'].append(epoch_recon_loss)
        history['kl_loss'].append(epoch_kl_loss)
        
        if verbose:
            print(f'Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Recon: {epoch_recon_loss:.4f}, KL: {epoch_kl_loss:.4f}')
    
    if verbose:
        print("Finished Training")
    
    return history


def train_gan(generator, critic, train_loader, opt_gen, opt_critic, device, epochs=10, 
              n_critic=5, clip_value=0.01, z_dim=100, verbose=True):
    """
    Train a WGAN (Wasserstein GAN) model.
    
    Args:
        generator (nn.Module): Generator network
        critic (nn.Module): Critic/Discriminator network
        train_loader (DataLoader): DataLoader for training data
        opt_gen: Generator optimizer
        opt_critic: Critic optimizer  
        device (str or torch.device): Device to train on
        epochs (int): Number of training epochs. Default: 10
        n_critic (int): Number of critic updates per generator update. Default: 5
        clip_value (float): Weight clipping value for WGAN. Default: 0.01
        z_dim (int): Dimension of noise vector. Default: 100
        verbose (bool): If True, print training progress. Default: True
    
    Returns:
        dict: Dictionary containing training history with keys:
              - 'gen_loss': List of generator loss per batch
              - 'critic_loss': List of critic loss per batch
              - 'epochs_completed': Number of epochs completed
    
    Example:
        >>> from helper.model import Generator, Critic
        >>> generator = Generator(z_dim=100).to(device)
        >>> critic = Critic().to(device)
        >>> opt_gen = torch.optim.RMSprop(generator.parameters(), lr=5e-5)
        >>> opt_critic = torch.optim.RMSprop(critic.parameters(), lr=5e-5)
        >>> history = train_gan(generator, critic, train_loader, opt_gen, opt_critic, device)
    """
    generator.to(device)
    critic.to(device)
    
    history = {
        'gen_loss': [],
        'critic_loss': [],
        'epochs_completed': 0
    }
    
    # Fixed noise for visualization
    fixed_noise = torch.randn(64, z_dim).to(device)
    
    for epoch in range(epochs):
        if verbose:
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        else:
            progress_bar = train_loader
            
        for batch_idx, (real, _) in enumerate(progress_bar):
            real = real.to(device)
            batch_size = real.size(0)
            
            # Train Critic multiple times
            for _ in range(n_critic):
                # Generate fake images
                noise = torch.randn(batch_size, z_dim).to(device)
                fake = generator(noise).detach()  # detach to avoid training generator
                
                # Critic scores
                critic_real = critic(real).mean()
                critic_fake = critic(fake).mean()
                
                # WGAN loss: maximize real scores, minimize fake scores
                loss_critic = -(critic_real - critic_fake)
                
                # Update critic
                critic.zero_grad()
                loss_critic.backward()
                opt_critic.step()
                
                # Weight clipping for WGAN
                for p in critic.parameters():
                    p.data.clamp_(-clip_value, clip_value)
            
            # Train Generator
            noise = torch.randn(batch_size, z_dim).to(device)
            fake = generator(noise)
            
            # Generator loss: maximize critic scores for fake images
            loss_gen = -critic(fake).mean()
            
            # Update generator
            generator.zero_grad()
            loss_gen.backward()
            opt_gen.step()
            
            # Record losses
            history['gen_loss'].append(loss_gen.item())
            history['critic_loss'].append(loss_critic.item())
            
            # Update progress bar
            if verbose:
                progress_bar.set_postfix({
                    'G_loss': f'{loss_gen.item():.4f}',
                    'C_loss': f'{loss_critic.item():.4f}'
                })
            
            # Print progress
            if batch_idx % 100 == 0 and verbose:
                print(f"[Epoch {epoch+1}/{epochs}] [Batch {batch_idx}/{len(train_loader)}] "
                      f"[D loss: {loss_critic.item():.4f}] [G loss: {loss_gen.item():.4f}]")
        
        history['epochs_completed'] = epoch + 1
        
        if verbose:
            print(f'Epoch {epoch+1}/{epochs} completed')
    
    if verbose:
        print("Finished GAN Training")
    
    return history


def generate_gan_samples(generator, num_samples=64, z_dim=100, device='cpu'):
    """
    Generate samples using a trained GAN generator.
    
    Args:
        generator (nn.Module): Trained generator network
        num_samples (int): Number of samples to generate. Default: 64
        z_dim (int): Dimension of noise vector. Default: 100
        device (str or torch.device): Device to generate on. Default: 'cpu'
    
    Returns:
        torch.Tensor: Generated images
    
    Example:
        >>> samples = generate_gan_samples(generator, num_samples=16, device='cuda')
    """
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, z_dim).to(device)
        samples = generator(noise)
    return samples


__all__ = ['train_model', 'train_epoch', 'train_vae_model', 'vae_loss_function', 'train_gan', 'generate_gan_samples']
