import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
import numpy as np

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, img_shape),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 3, 64, 64) 
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_shape, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

class GAN:
    def __init__(self, latent_dim, img_shape):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = Generator(latent_dim, img_shape).to(self.device)
        self.discriminator = Discriminator(img_shape).to(self.device)

        self.adversarial_loss = nn.BCELoss()
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def train(self, dataloader, num_epochs, sample_interval=100):
        for epoch in range(num_epochs):
            for i, real_images in enumerate(dataloader):
                valid = torch.ones(real_images.size(0), 1).to(self.device)
                fake = torch.zeros(real_images.size(0), 1).to(self.device)

                real_images = real_images.to(self.device)

                self.generator_optimizer.zero_grad()
                z = torch.randn(real_images.size(0), latent_dim).to(self.device)
                generated_images = self.generator(z)
                g_loss = self.adversarial_loss(self.discriminator(generated_images), valid)
                g_loss.backward()
                self.generator_optimizer.step()

                self.discriminator_optimizer.zero_grad()
                real_loss = self.adversarial_loss(self.discriminator(real_images), valid)
                fake_loss = self.adversarial_loss(self.discriminator(generated_images.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.discriminator_optimizer.step()

                if i % 100 == 0:
                    print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

                if i % sample_interval == 0:
                    self.sample_images(epoch, i)

    def sample_images(self, epoch, batch_idx):
        with torch.no_grad():
            z = torch.randn(5, latent_dim).to(self.device)
            generated_images = self.generator(z)
            save_image(generated_images, f"C:/Users/BobLoblaw/Desktop/Course Materials/ARI5004/project/generated/sample_{epoch}_{batch_idx}.png", nrow=5, normalize=True)

class CancerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
if __name__ == '__main__':
    torch.manual_seed(42)

    latent_dim = 100
    img_shape = 3 * 64 * 64
    batch_size = 4
    num_epochs = 5

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = CancerDataset(root_dir='../assignment2_part1/images/images', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    gan = GAN(latent_dim, img_shape)
    gan.train(dataloader, num_epochs=num_epochs)