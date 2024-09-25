import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader , Dataset
import torchvision 
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F


import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection to match dimensions if necessary
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNetUpsample(nn.Module):
    def __init__(self, input_dims):
        super(ResNetUpsample, self ).__init__()
        
        # Input is 1024x4x4
        self.in_channels = 1024

        self.input_layer = nn.Linear(input_dims, 4 * 4 * 1024)
        # Residual block 1
        self.layer1 = self._make_layer(1024, 512, stride=1, num_blocks=1)  # Output: 512x4x4

        # Upsample blocks to increase the spatial resolution
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # Output: 256x8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: 128x16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # Output: 64x32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        
        self.upsample4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # Output: 32x64x64
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        self.upsample5 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    # Output: 16x128x128
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        
        # Final Conv Layer to output 3 channels (RGB)
        self.final_conv = nn.Conv2d(16, 3, kernel_size=3, padding=1)  # Output: 3x128x128
    
    def _make_layer(self, in_channels, out_channels, stride, num_blocks):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Residual block 1 (input: 1024x4x4)
        x = self.input_layer(x)
        x = x.view(x.size(0), 1024, 4, 4) 
        out = self.layer1(x)  # Output: 512x4x4

        # Upsample layers
        out = self.upsample1(out)  # Output: 256x8x8
        out = self.upsample2(out)  # Output: 128x16x16
        out = self.upsample3(out)  # Output: 64x32x32
        out = self.upsample4(out)  # Output: 32x64x64
        out = self.upsample5(out)  # Output: 16x128x128
        
        # Final output (RGB)
        out = torch.nn.functional.tanh(self.final_conv(out))  # Output: 3x128x128
        return out
    
class Critic(nn.Module):
    def __init__(self, output_dims):
        super(Critic, self).__init__()

        # Initial convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=5, stride=2, padding=2)

        # Additional convolutional layers to increase complexity
        self.conv6 = nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1)

        # # Spectral Normalization for stability
        # self.conv1 = nn.utils.spectral_norm(self.conv1)
        # self.conv2 = nn.utils.spectral_norm(self.conv2)
        # self.conv3 = nn.utils.spectral_norm(self.conv3)
        # self.conv4 = nn.utils.spectral_norm(self.conv4)
        # self.conv5 = nn.utils.spectral_norm(self.conv5)
        # self.conv6 = nn.utils.spectral_norm(self.conv6)
        # self.conv7 = nn.utils.spectral_norm(self.conv7)

        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.4)

        # # Instance Normalization for better training dynamics
        # self.inst_norm1 = nn.InstanceNorm2d(64)
        # self.inst_norm2 = nn.InstanceNorm2d(128)
        # self.inst_norm3 = nn.InstanceNorm2d(256)
        # self.inst_norm4 = nn.InstanceNorm2d(512)
        # self.inst_norm5 = nn.InstanceNorm2d(1024)
        #self.inst_norm6 = nn.InstanceNorm2d(2048)

        # Fully connected output layer
        self.fc = nn.Linear(4 * 4 * 2048, output_dims)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu((self.conv1(x)))
        x = torch.nn.functional.leaky_relu((self.conv2(x)))
        x = torch.nn.functional.leaky_relu((self.conv3(x)))
        x = torch.nn.functional.leaky_relu((self.conv4(x)))
        x = torch.nn.functional.leaky_relu((self.conv5(x)))

        # Additional convolutional layers
        x = torch.nn.functional.leaky_relu((self.conv6(x)))
        x = torch.nn.functional.leaky_relu(self.conv7(x))

        # Dropout for regularization
        x = self.dropout(x)

        # Global Average Pooling
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, features)

        # Final output layer
        return self.fc(x)


    

class DCGan(nn.Module):
    def __init__(self,input_dims =100, output_dims = 1):
        super(DCGan,self).__init__()
        self.gen = ResNetUpsample(input_dims)
        self.critic = Critic(output_dims)
    def forward(self , latents , images):
        x = self.gen(latents)
        generated = self.critic(x)
        real = self.critic(images) 
        return real , generated
        





class DCTrainer():
    def __init__(self,model , dataloader , optim_gen , optim_critic  , epochs, device ,latent_dims, lambda_ = 5):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.optim_gen = optim_gen 
        self.optim_critc = optim_critic
        self.epochs = epochs
        self.device = device
        self.lambda_ = lambda_
        self.latent_dims = latent_dims
        self.mvn = torch.distributions.MultivariateNormal( torch.zeros(latent_dims)  , torch.eye(latent_dims) )
    
    
    def latent_sampler(self ,batch_size ):
        samples = self.mvn.sample((batch_size,))
        return samples
    
    def W_loss_critic(self , real_y, gen_y,):
        real = torch.mean(real_y)
        fake = torch.mean(gen_y)
        return (fake - real) 
    
    def W_loss_gen(self , gen_y):
        return -torch.mean(gen_y)
    

    def noramlize_weights(self ):
        for p in self.model.critic.parameters():
            p.data.clamp_(-0.01 ,0.01)

    def gradient_penalty(self , real , fake):
        alpha = torch.rand(real.shape[0], 1, 1, 1).to(self.device)
        interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        disc_interpolates = self.model.critic(interpolates)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train_step(self) :
        total_loss_critic = 0
        total_loss_gen = 0
        for i , (data , _) in enumerate(self.dataloader):
            data = data.to(self.device)
            self.optim_critc.zero_grad()
            self.optim_gen.zero_grad()
            z = self.latent_sampler(data.shape[0]).to(device=self.device)
            gen_data = self.model.gen(z).detach()
            gen_y   = self.model.critic(gen_data) 
            real_y = self.model.critic(data)
            loss_critic = self.W_loss_critic(real_y , gen_y) + self.lambda_ * self.gradient_penalty(data , gen_data)
            loss_critic.backward()
            self.optim_critc.step()
           # self.noramlize_weights()

            if( i% 5 == 0):
                z = self.latent_sampler(data.shape[0]).to(device=self.device)
            
                gen_data = self.model.gen(z)
                gen_y = self.model.critic(gen_data)
                loss_gen = -torch.mean(gen_y)
                loss_gen.backward()
                self.optim_gen.step()
                
                total_loss_gen += loss_gen.item()
            total_loss_critic += loss_critic.item()
            
        return total_loss_critic/len(self.dataloader) , total_loss_gen/len(self.dataloader)
    

    def show(self):
        images = self.generate_images(10)
        images  = [images[i].permute(1,2,0).detach().cpu().numpy() for i in range(images.shape[0])]
        plt.figure(figsize=(15, 10))  # Adjust the size as needed

        # Loop through the images and display each one
        for i, image in enumerate(images):
            plt.subplot(2, 5, i + 1)  # Change '1, 5' to the desired grid layout
            plt.imshow((image * 255).astype(np.uint8))
            plt.axis('off')  # This hides the axis

        plt.show()
        plt.save("img.png")

    def generate_images(self , number = 5):
        self.model.eval()
        with torch.no_grad():
            z = self.latent_sampler(number).to(device=self.device)
            images = self.model.gen(z)
            return images
        
    def train(self):
        for epoch in range(self.epochs):
            loss_critic , loss_gen = self.train_step()
            print(f"Epoch {epoch} Generator Loss {loss_gen} Critic loss {loss_critic}")
            if(epoch%500 == 0):
                torch.save(self.model , f"models_resnet/model{epoch}.pt")
            # if(epoch%50 == 0):
            #     self.show()

batch_size = 128
input_dims = 100
epochs = 1000
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

TRANSFORM_IMG = transforms.Compose([
    transforms.Resize((128  , 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5 , 0.5 , 0.5) , (0.5 , 0.5 , 0.5))
    ])
dataset = torchvision.datasets.ImageFolder("./dataset/animals" ,transform=TRANSFORM_IMG)

dataloader = DataLoader(dataset, batch_size= batch_size, shuffle=True, num_workers=4)

model = torch.load("models_resnet/model500.pt" , map_location=device)
learning_rate = 1e-4

optim_critc = torch.optim.RMSprop(model.critic.parameters() , learning_rate)
optim_gen = torch.optim.RMSprop(model.gen.parameters() , learning_rate)

trainer = DCTrainer(model , dataloader , optim_critic=optim_critc, optim_gen=optim_gen , epochs=epochs , device = device , latent_dims=input_dims  ,lambda_=10)
trainer.show()
