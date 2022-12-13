import torch
from numpy import linalg as LA

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