# %%
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

# %% [markdown]
# # Parameters Section 

# %%
batch_size = 128
input_dims = 100
epochs = 1000
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# # Data and Dataloader preparing section

# %%
TRANSFORM_IMG = transforms.Compose([
    transforms.Resize((128  , 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5 , 0.5 , 0.5) , (0.5 , 0.5 , 0.5))
    ])
dataset = torchvision.datasets.ImageFolder("./dataset/animals" ,transform=TRANSFORM_IMG)

dataloader = DataLoader(dataset, batch_size= batch_size, shuffle=True, num_workers=4)


# %% [markdown]
# # Model definition

# %%
class Generator(nn.Module):
    def __init__(self, input_dims , dim = 4):
        super(Generator, self).__init__()
        def upsample_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_channels , out_channels , kernel_size = 3 , stride = 1 , padding = 'same'),
                nn.BatchNorm2d(out_channels),
            )
        self.input_layer = nn.Linear(100 , 64 * 4*4)
        self.first = upsample_block(64 , 128)
        self.second = upsample_block(128 , 256)
        layers = [upsample_block(256 , 256) for _ in range(2)]
        self.middle = nn.Sequential(*layers)
        self.last = upsample_block(256 , 3)
        self.final = nn.Conv2d(3 , 3 , kernel_size = 3 , stride = 1 , padding = 'same')



    def forward(self, x):
        x = self.input_layer(x)
        x = x.view(x.size(0), 64, 4, 4)  # Reshape for ConvTranspose layers
        x = self.first(x)
        x = self.second(x)
        x = self.middle(x)
        x = self.last(x)
        x = self.final(x)

        x = torch.nn.functional.tanh(x)

        return x
    
    
class Critic(nn.Module):
    def __init__(self, output_dims):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=4, padding=1)
        # self.norm1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=4, padding=1)
        # self.norm2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=4, padding=1)
        # self.norm3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=4, padding=1)

        self.conv4 = nn.utils.spectral_norm(self.conv4)
        self.conv3 = nn.utils.spectral_norm(self.conv3)
        self.conv2 = nn.utils.spectral_norm(self.conv2)
        self.conv1 = nn.utils.spectral_norm(self.conv1)

        self.output_layer = nn.Linear(512*1*1, output_dims)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu((self.conv1(x)))
        x = torch.nn.functional.leaky_relu((self.conv2(x)))
        x = torch.nn.functional.leaky_relu((self.conv3(x)))
        x = torch.nn.functional.leaky_relu((self.conv4(x)))
        # Global Average Pooling
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, features)
        return self.output_layer(x)


    

class DCGan(nn.Module):
    def __init__(self,input_dims =100, output_dims = 1):
        super(DCGan,self).__init__()
        self.gen = Generator(input_dims)
        self.critic = Critic(output_dims)
    def forward(self , latents , images):
        x = self.gen(latents)
        generated = self.critic(x)
        real = self.critic(images) 
        return real , generated
        




# %%
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
            if(epoch%50 == 0):
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_critic_state_dict': self.optim_critc.state_dict(),
                     'optimizer_gen_state_dict' : self.optim_gen.state_dict(),
                    'epoch': epoch  # Optional: Save the epoch to resume from
                }, f'newFolder/checkpoint{epoch}.pth')
            # if(epoch%5 == 0):
            #     self.show()




# %% [markdown]
# # Instantiation

# %%

model = torch.load("model550.pt" , map_location=device)
learning_rate = 1e-4

optim_critc = torch.optim.RMSprop(model.critic.parameters() , learning_rate)
optim_gen = torch.optim.RMSprop(model.gen.parameters() , learning_rate)



# %%
trainer = DCTrainer(model , dataloader , optim_critic=optim_critc, optim_gen=optim_gen , epochs=epochs , device = device , latent_dims=input_dims  ,lambda_=10)

# %%
trainer.train()

# %%


# %%


# %%



