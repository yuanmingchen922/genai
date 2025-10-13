"""
Model GANs
Provides reusable neural network model architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNN(nn.Module):
    """
    Simple fully connected neural network with configurable layers.
    
    Args:
        layer_sizes (list): List of integers specifying the size of each layer.
                           First element is input size, last is output size.
                           Example: [784, 128, 64, 10] creates network with
                           784 inputs, two hidden layers (128, 64), and 10 outputs.
        activation (str): Activation function ('relu', 'sigmoid', 'tanh'). Default: 'relu'
        dropout_rate (float): Dropout rate for regularization. Default: 0.0
    
    Example:
        >>> model = SimpleNN([784, 128, 64, 10], activation='relu', dropout_rate=0.2)
        >>> output = model(input_tensor)
    """
    
    def __init__(self, layer_sizes, activation='relu', dropout_rate=0.0):
        super(SimpleNN, self).__init__()
        
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must contain at least 2 elements (input and output)")
        
        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.dropout_rate = dropout_rate
        
        # Build layers
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        
        # Dropout for regularization
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        
        # Select activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output tensor
        """
        # Flatten input if needed (for image data)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Pass through all layers except the last
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
            if self.dropout is not None:
                x = self.dropout(x)
        
        # Final layer (no activation, will be applied in loss function)
        x = self.layers[-1](x)
        
        return x


class SimpleCNN(nn.Module):
    """
    Simple Convolutional Neural Network for image classification.
    
    Args:
        num_classes (int): Number of output classes. Default: 10
        input_channels (int): Number of input channels (1 for grayscale, 3 for RGB). Default: 1
    
    Example:
        >>> model = SimpleCNN(num_classes=10, input_channels=1)
        >>> output = model(image_tensor)
    """
    
    def __init__(self, num_classes=10, input_channels=1):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers (assuming 28x28 input -> 3x3 after pooling)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        """
        Forward pass through the CNN.
        
        Args:
            x (torch.Tensor): Input image tensor [batch_size, channels, height, width]
        
        Returns:
            torch.Tensor: Output tensor [batch_size, num_classes]
        """
        # Conv block 1: Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        
        # Conv block 2: Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        
        # Conv block 3: Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv3(x)))  # 7x7 -> 3x3
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x


class VariationalEncoder(nn.Module):
    """
    Variational Encoder for VAE.
    Maps input to a distribution (mean and log variance) in latent space.
    
    Args:
        latent_dim (int): Dimension of the latent space. Default: 2
        input_channels (int): Number of input channels. Default: 1
    
    Example:
        >>> encoder = VariationalEncoder(latent_dim=2)
        >>> mu, logvar = encoder(image_tensor)
    """
    
    def __init__(self, latent_dim=2, input_channels=1):
        super(VariationalEncoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Convolutional layers (32x32 -> 4x4)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        
        # Output mean and log variance
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)
    
    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Args:
            x (torch.Tensor): Input image tensor
        
        Returns:
            tuple: (mu, logvar) - mean and log variance of latent distribution
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class VariationalDecoder(nn.Module):
    """
    Variational Decoder for VAE.
    Maps latent space back to image space using transpose convolutions.
    
    Args:
        latent_dim (int): Dimension of the latent space. Default: 2
        output_channels (int): Number of output channels. Default: 1
    
    Example:
        >>> decoder = VariationalDecoder(latent_dim=2)
        >>> reconstructed = decoder(latent_vector)
    """
    
    def __init__(self, latent_dim=2, output_channels=1):
        super(VariationalDecoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Fully connected layer to expand latent vector
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        
        # Transpose convolutional layers (4x4 -> 32x32)
        self.convtrans1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtrans2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtrans3 = nn.ConvTranspose2d(32, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
    
    def forward(self, x):
        """
        Forward pass through the decoder.
        
        Args:
            x (torch.Tensor): Latent vector
        
        Returns:
            torch.Tensor: Reconstructed image tensor
        """
        x = self.fc(x)
        x = x.view(-1, 128, 4, 4)
        x = F.relu(self.convtrans1(x))
        x = F.relu(self.convtrans2(x))
        x = torch.sigmoid(self.convtrans3(x))
        return x


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE).
    Combines encoder and decoder with reparameterization trick.
    
    Args:
        latent_dim (int): Dimension of the latent space. Default: 2
        input_channels (int): Number of input channels. Default: 1
        beta (float): Weight for KL divergence term. Default: 1.0
    
    Example:
        >>> vae = VAE(latent_dim=2)
        >>> reconstruction, mu, logvar = vae(image_tensor)
    """
    
    def __init__(self, latent_dim=2, input_channels=1, beta=1.0):
        super(VAE, self).__init__()
        self.encoder = VariationalEncoder(latent_dim, input_channels)
        self.decoder = VariationalDecoder(latent_dim, input_channels)
        self.beta = beta
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: sample from N(mu, var) using N(0,1).
        
        Args:
            mu (torch.Tensor): Mean of the latent distribution
            logvar (torch.Tensor): Log variance of the latent distribution
        
        Returns:
            torch.Tensor: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """
        Forward pass through VAE.
        
        Args:
            x (torch.Tensor): Input image tensor
        
        Returns:
            tuple: (reconstruction, mu, logvar)
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar
    
    def generate(self, num_samples=1, device='cpu'):
        """
        Generate new samples by sampling from standard normal distribution.
        
        Args:
            num_samples (int): Number of samples to generate
            device (str or torch.device): Device to generate on
        
        Returns:
            torch.Tensor: Generated images
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.encoder.latent_dim).to(device)
            samples = self.decoder(z)
        return samples


class Generator(nn.Module):
    """
    GAN Generator network using transpose convolutions.
    Generates RGB images from random noise vectors.
    
    Args:
        z_dim (int): Dimension of input noise vector. Default: 100
        output_channels (int): Number of output channels (3 for RGB). Default: 3
        image_size (int): Output image size (assumes square images). Default: 64
    
    Example:
        >>> generator = Generator(z_dim=100)
        >>> noise = torch.randn(32, 100)
        >>> fake_images = generator(noise)
    """
    
    def __init__(self, z_dim=100, output_channels=3, image_size=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        
        # Reshape function to convert 1D noise to 4D tensor
        self.reshape = lambda x: x.view(x.size(0), z_dim, 1, 1)
        
        # Transpose convolution layers (1x1 -> 64x64)
        self.deconv1 = nn.ConvTranspose2d(z_dim, 512, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(512, momentum=0.9)
        self.act1 = nn.ReLU(True)
        
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256, momentum=0.9)
        self.act2 = nn.ReLU(True)
        
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128, momentum=0.9)
        self.act3 = nn.ReLU(True)
        
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64, momentum=0.9)
        self.act4 = nn.ReLU(True)
        
        self.deconv5 = nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        """
        Forward pass through the generator.
        
        Args:
            x (torch.Tensor): Input noise vector [batch_size, z_dim]
        
        Returns:
            torch.Tensor: Generated images [batch_size, channels, height, width]
        """
        x = self.reshape(x)
        
        x = self.deconv1(x)      # -> 512 x 4 x 4
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.deconv2(x)      # -> 256 x 8 x 8
        x = self.bn2(x)
        x = self.act2(x)
        
        x = self.deconv3(x)      # -> 128 x 16 x 16
        x = self.bn3(x)
        x = self.act3(x)
        
        x = self.deconv4(x)      # -> 64 x 32 x 32
        x = self.bn4(x)
        x = self.act4(x)
        
        x = self.deconv5(x)      # -> 3 x 64 x 64
        x = self.tanh(x)         # Output range [-1, 1]
        
        return x


class Critic(nn.Module):
    """
    GAN Critic/Discriminator network using convolutions.
    Evaluates whether input images are real or fake.
    
    Args:
        input_channels (int): Number of input channels (3 for RGB). Default: 3
        image_size (int): Input image size (assumes square images). Default: 64
    
    Example:
        >>> critic = Critic(input_channels=3)
        >>> images = torch.randn(32, 3, 64, 64)
        >>> scores = critic(images)
    """
    
    def __init__(self, input_channels=3, image_size=64):
        super(Critic, self).__init__()
        
        # Convolutional layers (64x64 -> 1x1)
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(512)
        self.act4 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        """
        Forward pass through the critic.
        
        Args:
            x (torch.Tensor): Input images [batch_size, channels, height, width]
        
        Returns:
            torch.Tensor: Critic scores [batch_size, 1]
        """
        x = self.conv1(x)        # -> 64 x 32 x 32
        x = self.act1(x)
        
        x = self.conv2(x)        # -> 128 x 16 x 16
        x = self.batchnorm2(x)
        x = self.act2(x)
        
        x = self.conv3(x)        # -> 256 x 8 x 8
        x = self.batchnorm3(x)
        x = self.act3(x)
        
        x = self.conv4(x)        # -> 512 x 4 x 4
        x = self.batchnorm4(x)
        x = self.act4(x)
        
        x = self.conv5(x)        # -> 1 x 1 x 1
        x = self.flatten(x)      # -> 1
        
        return x


class GAN(nn.Module):
    """
    Complete GAN model combining Generator and Critic.
    
    Args:
        z_dim (int): Dimension of noise vector. Default: 100
        image_channels (int): Number of image channels. Default: 3
        image_size (int): Size of square images. Default: 64
    
    Example:
        >>> gan = GAN(z_dim=100)
        >>> noise = torch.randn(32, 100)
        >>> fake_images = gan.generate(noise)
        >>> scores = gan.discriminate(fake_images)
    """
    
    def __init__(self, z_dim=100, image_channels=3, image_size=64):
        super(GAN, self).__init__()
        self.generator = Generator(z_dim, image_channels, image_size)
        self.critic = Critic(image_channels, image_size)
        self.z_dim = z_dim
    
    def generate(self, noise):
        """Generate images from noise."""
        return self.generator(noise)
    
    def discriminate(self, images):
        """Evaluate images with critic."""
        return self.critic(images)
    
    def sample_noise(self, batch_size, device='cpu'):
        """Sample random noise vectors."""
        return torch.randn(batch_size, self.z_dim).to(device)


__all__ = ['SimpleNN', 'SimpleCNN', 'VariationalEncoder', 'VariationalDecoder', 'VAE', 'Generator', 'Critic', 'GAN']
