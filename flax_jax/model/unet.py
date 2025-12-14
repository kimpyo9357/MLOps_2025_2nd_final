import jax
import jax.numpy as jnp
from flax import linen as nn

class CNN(nn.Module):
    num_classes: int

    def setup(self):
        self.conv1 = nn.Conv(features=16, kernel_size=(5, 5), strides=(2, 2), padding='VALID')
        self.conv2 = nn.Conv(features=16, kernel_size=(5, 5), strides=(2, 2), padding='VALID')
        self.dense1 = nn.Dense(features=self.num_classes)

    def __call__(self, x,):
        x = self.conv1(x)
        x = nn.relu(x)
        x = self.conv2(x)
        x = nn.relu(x)
        x = jnp.mean(x, axis=(1, 2))
        x = self.dense1(x)
        return x
    
class ConvBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.features, (3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(self.features, (3, 3), padding='SAME')(x)
        x = nn.relu(x)
        return x

class DownBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = ConvBlock(self.features)(x)
        skip = x
        x = nn.max_pool(x, (2, 2), (2, 2))
        return x, skip

class UpBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x, skip):
        # Upsample
        x = nn.ConvTranspose(self.features, (2, 2), strides=(2, 2))(x)
        # Concatenate skip features
        x = jnp.concatenate([x, skip], axis=-1)
        x = ConvBlock(self.features)(x)
        return x

class UNet(nn.Module):
    num_classes: int

    def setup(self):
        # Encoder
        self.down1 = DownBlock(64)
        self.down2 = DownBlock(128)
        self.down3 = DownBlock(256)

        # Bottleneck
        self.bottleneck = ConvBlock(512)

        # Decoder
        self.up3 = UpBlock(256)
        self.up2 = UpBlock(128)
        self.up1 = UpBlock(64)

        # Output conv
        self.out_conv = nn.Conv(self.num_classes, (1, 1))
        self.out_dense = nn.Dense(self.num_classes)

    def __call__(self, x):
        # Encoder
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)

        # Output
        x = self.out_conv(x)
        
        x = jnp.mean(x, axis=(1,2))
        x = self.out_dense(x)
        return x