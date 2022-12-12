import torch
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import torch.nn.functional as F
import os

class AE(torch.nn.Module):
    def __init__(self, native_dim, latent_dim, hidden_layer):
        super().__init__()

        # If using unsquashed actions, actions are no longer bounded between -1 and +1.
        self.latent_dim = latent_dim

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(native_dim, hidden_layer),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(hidden_layer, latent_dim),
            torch.nn.Tanh(),
        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_layer),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(hidden_layer, native_dim),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        encoded = self.encoder(x) 
        decoded = self.decoder(encoded)
        return decoded

class AE_Trainer():
    def __init__(self, native_dim, latent_dim, optimizer, model, loss_criterion):
        super(AE_Trainer, self).__init__()
        self.native_dim = native_dim
        self.latent_dim = latent_dim
        self.optimizer = optimizer
        self.model = model
        self.loss_criterion = loss_criterion
        self.records = {}
        self.init_records()
        self.min_loss = float('inf')
    
    def init_records(self):
        self.records = {
            "train_recon_loss": [],
            "test_recon_loss": [],
        }

    def train(self, train_dataloader, epoch):
        running_recon_loss = 0.0
        size = len(train_dataloader.dataset)
        for i, input_data in enumerate(train_dataloader, 0):
            self.optimizer.zero_grad()
            output_data = self.model(input_data)
            loss = self.loss_criterion(input_data, output_data)
            loss.backward()
            self.optimizer.step()
            running_recon_loss += loss.item() * len(input_data)
        self.records["train_recon_loss"].append(running_recon_loss / size)
        print('====> {}-{} Epoch: {} Train recon loss: {}'.format(
            "AE",
            self.latent_dim,
            epoch, 
            running_recon_loss / size))

    def test(self, test_dataloader, epoch, model_path):
        running_recon_loss = 0.0
        size = len(test_dataloader.dataset)
        for i, input_data in enumerate(test_dataloader, 0):
            output_data = self.model(input_data)
            loss = self.loss_criterion(input_data, output_data)
            running_recon_loss += loss.item() * len(input_data)
        running_recon_loss = running_recon_loss / size
        self.records["test_recon_loss"].append(running_recon_loss)
        print('====> {}-{} Epoch: {} Test recon loss: {}'.format(
            "AE",
            self.latent_dim,
            epoch, 
            running_recon_loss))

        if running_recon_loss < self.min_loss:
            self.min_loss = running_recon_loss
            PATH = os.path.join(model_path, "{}_{}.pt".format("AE",self.latent_dim))
            print("save model=========")
            torch.save(self.model.state_dict(), PATH)


    def draw_records(self, title, result_path):
        for mode in ["train", "test"]:
            if mode == "train":
                line_style = 'solid'
                color = "#1f78b4"
            elif mode == "test":
                line_style = 'dashed'
                color = "#a6cee3"
            data = self.records["{}_recon_loss".format(mode)]
            record_name = "{} recon".format(mode)
            plt.plot(data, linestyle=line_style, color=color, label=record_name)
            plt.ylabel("Loss")
            plt.yscale("log")
            plt.xlabel("Epochs")
            plt.title(title)
            plt.legend(loc="upper right")
            plt.savefig(os.path.join(result_path, "{}_dim{}_running_loss".format("AE",self.latent_dim)), dpi=400)