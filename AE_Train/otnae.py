import torch
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import torch.nn.functional as F
import os

class Encoder(torch.nn.Module):
    def __init__(self, native_dim, hidden_layer, latent_dim):
        super().__init__()
    
        self.linear1 = torch.nn.Linear(native_dim, hidden_layer)
        self.linear2 = torch.nn.Linear(hidden_layer, latent_dim)
        self.leaky = torch.nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.leaky(self.linear1(x))
        return self.linear2(x) 

class Decoder(torch.nn.Module):
    def __init__(self, native_dim, hidden_layer, latent_dim):
        super().__init__()

        self.linear1 = torch.nn.Linear(latent_dim, hidden_layer)
        self.linear2 = torch.nn.Linear(hidden_layer, native_dim)
        self.leaky = torch.nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky(self.linear1(x))
        return self.linear2(x)

class OTNAE(torch.nn.Module):
    def __init__(self, native_dim, hidden_layer):
        super().__init__()

        self.native_dim = native_dim
        self.encoder = Encoder(native_dim, hidden_layer, native_dim)
        self.decoders = torch.nn.ModuleList(
            [Decoder(native_dim, hidden_layer, 1) for i in range(native_dim)]
        )
        self.tanh = torch.nn.Tanh()
    
    def forward(self, x):
        encoded = self.tanh(self.encoder(x))
        num_of_inputs = x.shape[0]
        outputs = torch.zeros(self.native_dim, num_of_inputs, self.native_dim)
        index = 0
        for decoder in self.decoders:
            for i in range(index, self.native_dim):
                outputs[i, :, :] += decoder(encoded[:, index].reshape(-1, 1))
            index += 1
        outputs = self.tanh(outputs)
        return outputs

class OTNAE_loss_criterion(torch.nn.Module):
    def __init__(self, native_dim):
        super(OTNAE_loss_criterion, self).__init__()
        self.native_dim = native_dim
 
    def forward(self, recons, original):
        total_loss = 0.0
        recon_loss_list = []
        # weights = [1, 2, 3, 4, 5, 6, 7, 8]
        for i in range(self.native_dim): 
            recon = recons[i,:,:]      
            recon_loss = F.mse_loss(recon, original)
            total_loss += (i+1) * recon_loss
            recon_loss_list.append(recon_loss.item())

        total_loss = total_loss / self.native_dim
        return total_loss, recon_loss_list

class OTNAE_Trainer():
    def __init__(self, native_dim, optimizer, model, loss_criterion):
        super(OTNAE_Trainer, self).__init__()
        self.native_dim = native_dim
        self.optimizer = optimizer
        self.model = model
        self.loss_criterion = loss_criterion
        self.records = {}
        self.init_records()
        self.min_loss = float('inf')
    
    def init_records(self):
        for i in range(self.native_dim):
            self.records[i+1] = {
                "train_recon_loss": [],
                "test_recon_loss": [],
            }
        self.records[self.native_dim] = {
            "train_recon_loss": [],
            "test_recon_loss": [],
            "train_total_loss": [],
            "test_total_loss":[],
        }

    def train(self, train_dataloader, epoch):
        train_recon_loss_list = np.array([0.0 for i in range(self.native_dim)])
        train_total_loss = 0.0
        size = len(train_dataloader.dataset)
        for i, input_data in enumerate(train_dataloader, 0):
            self.optimizer.zero_grad()
            outputs = self.model(input_data)
            loss, recon_loss_list = self.loss_criterion(outputs, input_data)
            loss.backward()
            self.optimizer.step()
            train_recon_loss_list = train_recon_loss_list + np.array(recon_loss_list) * len(input_data)
            train_total_loss += loss.item() * len(input_data)        

        for i in range(self.native_dim):
            self.records[i+1]["train_recon_loss"].append(train_recon_loss_list[i] / size)
        self.records[self.native_dim]["train_total_loss"].append(train_total_loss / size)
        print('====> {}-{} Epoch: {} Train total loss: {:.6f},\t Train recon loss: {}'.format(
            "OTNAE",
            self.native_dim,
            epoch, 
            train_total_loss / size, 
            train_recon_loss_list / size))

    def test(self, test_dataloader, epoch, model_path):
        test_recon_loss_list = np.array([0.0 for i in range(self.native_dim)])
        test_total_loss = 0.0
        size = len(test_dataloader.dataset)
        for i, input_data in enumerate(test_dataloader, 0):
            outputs = self.model(input_data)
            loss, recon_loss_list = self.loss_criterion(outputs, input_data)
            # test_recon_loss += recon_loss.item() * len(input_data)
            test_total_loss += loss.item() * len(input_data)
            test_recon_loss_list = test_recon_loss_list + np.array(recon_loss_list) * len(input_data)
        
        for i in range(self.native_dim):
            self.records[i+1]["test_recon_loss"].append(test_recon_loss_list[i] / size)
        self.records[self.native_dim]["test_total_loss"].append(test_total_loss / size)

        if test_total_loss < self.min_loss:
            self.min_loss = test_total_loss
            PATH = os.path.join(model_path, "{}_{}.pt".format("OTNAE",self.native_dim))
            print("save model=========")
            torch.save(self.model.state_dict(), PATH)
        
        print('====> {}-{} Epoch: {} Test total loss: {:.6f},\t Test recon loss: {}'.format(
            "OTNAE",
            self.native_dim,
            epoch, 
            test_total_loss / size, 
            test_recon_loss_list / size))

        return test_total_loss

    def draw_records(self, title, result_path, num_records):
        colors = [
            "#a6cee3", "#1f78b4", "#b2df8a",
            "#33a02c", "#fb9a99", "#e31a1c",
            "#fdbf6f", "#ff7f00", "#cab2d6"
        ]
                    
        for i in range(0, self.native_dim, num_records):
            loss = "recon"
            for j in range(i, min(i+num_records, self.native_dim)):
                for mode in ["train", "test"]:
                    if mode == "train":
                        line_style = 'solid'
                    elif mode == "test":
                        line_style = 'dashed'
                    color = colors[j%9]
                    data = self.records[j+1]["{}_{}_loss".format(mode, loss)]
                    record_name = "{}-{}".format(j+1, mode)
                    plt.plot(data, linestyle=line_style, color=color, label=record_name)

                
            loss = "total"
            for mode in ["train", "test"]:
                if mode == "train":
                    line_style = 'solid'
                elif mode == "test":
                    line_style = 'dashed'
                color = colors[self.native_dim%9]
                data = self.records[self.native_dim]["{}_{}_loss".format(mode, loss)]
                record_name = "{}-{}".format(mode, loss)
                plt.plot(data, linestyle=line_style, color=color, label=record_name)

            plt.ylabel("Loss")
            plt.yscale("log")
            plt.xlabel("Epochs")
            plt.title(title)
            plt.legend(loc="upper right")
            plt.savefig(os.path.join(result_path, "{}_{}dim-to-{}dim_running_loss".format("OTNAE",i+1, min(i+num_records, self.native_dim))), dpi=400)
            plt.clf()
            plt.cla()