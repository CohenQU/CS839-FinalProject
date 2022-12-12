import torch
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import torch.nn.functional as F
import os

class VAE(torch.nn.Module):
    def __init__(self, native_dim, latent_dim, hidden_layer):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(native_dim, hidden_layer),
            torch.nn.LeakyReLU(0.2),
        )

        self.mean = torch.nn.Linear(hidden_layer, latent_dim)
        self.log_std = torch.nn.Linear(hidden_layer, latent_dim)
        self.tanh = torch.nn.Tanh()

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_layer),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(hidden_layer, native_dim),
            torch.nn.Tanh(),
        )
        
    def forward(self, x):
        z = self.encoder(x)
        mean = self.mean(z)
        log_std = self.log_std(z)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)
        z = self.tanh(z)
        reconstruction = self.decoder(z)
        return reconstruction, mean, std

class VAE_loss_criterion(torch.nn.Module):
    def __init__(self):
        super(VAE_loss_criterion, self).__init__()
 
    def forward(self, action, recon, mean, std, beta):        
        recon_loss = F.mse_loss(recon, action)
        kl_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + beta * kl_loss
        return vae_loss, recon_loss, kl_loss

class VAE_Trainer():
    def __init__(self, native_dim, latent_dim, optimizer, model, loss_criterion, epochs):
        super(VAE_Trainer, self).__init__()
        self.native_dim = native_dim
        self.latent_dim = latent_dim
        self.optimizer = optimizer
        self.model = model
        self.loss_criterion = loss_criterion
        self.epochs = epochs
        self.records = {}
        self.init_records()
        self.min_loss = float('inf')
    
    def init_records(self):
        self.records = {
            "train_recon_loss": [],
            "train_kl_loss": [],
            "train_total_loss": [],
            "test_recon_loss": [],
            "test_kl_loss": [],
            "test_total_loss": []
        }

    def train(self, train_dataloader, epoch):
        running_recon_loss = 0.0
        running_kl_loss = 0.0
        running_total_loss = 0.0
        epochs = self.epochs
        stretch = 10
        sigmoid_x = (epoch - epochs/2) * stretch / (epochs/2)
        beta = 0.001 * np.exp(sigmoid_x) / (np.exp(sigmoid_x) + 1.0)

        size = len(train_dataloader.dataset)

        for i, input_data in enumerate(train_dataloader, 0):
            self.optimizer.zero_grad()
            output_data, mean, std = self.model(input_data)
            loss, recon_loss, kl_loss = self.loss_criterion(input_data, output_data, mean, std, beta)
            loss.backward()
            self.optimizer.step()
            running_recon_loss += recon_loss.item() * len(input_data)
            running_kl_loss += kl_loss.item() * len(input_data)
            running_total_loss += loss.item() * len(input_data)

        self.records["train_recon_loss"].append(running_recon_loss / size)
        self.records["train_kl_loss"].append(running_kl_loss / size)
        self.records["train_total_loss"].append(running_total_loss / size)

        print('====> {}-{} Epoch: {} Train recon loss: {}, Train kl loss: {}, Train total loss: {}'.format(
            "VAE",
            self.latent_dim,
            epoch, 
            running_recon_loss / size,
            running_kl_loss / size,
            running_total_loss / size))

    def test(self, test_dataloader, epoch, model_path):
        running_recon_loss = 0.0
        running_kl_loss = 0.0
        running_total_loss = 0.0
        num_of_total_epochs = 100
        stretch = 10
        sigmoid_x = (epoch - num_of_total_epochs / 2) * stretch / (num_of_total_epochs /2)
        beta = 0.001 * np.exp(sigmoid_x) / (np.exp(sigmoid_x) + 1.0)
        size = len(test_dataloader.dataset)
        
        for i, input_data in enumerate(test_dataloader, 0):
            self.optimizer.zero_grad()
            output_data, mean, std = self.model(input_data)
            loss, recon_loss, kl_loss = self.loss_criterion(input_data, output_data, mean, std, beta)
            running_recon_loss += recon_loss.item() * len(input_data)
            running_kl_loss += kl_loss.item() * len(input_data)
            running_total_loss += loss.item() * len(input_data)

        self.records["test_recon_loss"].append(running_recon_loss / size)
        self.records["test_kl_loss"].append(running_kl_loss / size)
        self.records["test_total_loss"].append(running_total_loss / size)

        print('====> {}-{} Epoch: {} Test recon loss: {}, Test kl loss: {}, Test total loss: {}'.format(
            "VAE",
            self.latent_dim,
            epoch, 
            running_recon_loss / size,
            running_kl_loss / size,
            running_total_loss / size))

        if running_total_loss < self.min_loss:
            self.min_loss = running_total_loss
            PATH = os.path.join(model_path, "{}_{}.pt".format("VAE",self.latent_dim))
            print("save model=========")
            torch.save(self.model.state_dict(), PATH)

    def draw_records(self, title, result_path):
        for mode in ["train", "test"]:
            if mode == "train":
                line_style = 'solid'
            elif mode == "test":
                line_style = 'dashed'
            for loss in ["recon", "kl", "total"]:
                if loss == "recon":
                    color = "#fc8d62"
                elif loss == "kl":
                    color = "#7fc97f"
                elif loss == "total":
                    color = "#beaed4"
                data = self.records["{}_{}_loss".format(mode, loss)]
                record_name = "{} {}".format(mode, loss)
                plt.plot(data, linestyle=line_style, color=color, label=record_name)
            plt.ylabel("Loss")
            plt.yscale("log")
            plt.xlabel("Epochs")
            plt.title(title)
            plt.legend(loc="upper right")
            plt.savefig(os.path.join(result_path, "{}_dim{}_running_loss".format("VAE",self.latent_dim)), dpi=400)