import torch

class VanillaAE(torch.nn.Module):
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

# class VariationalAE(torch.nn.Module):
#     def __init__(self, native_dim, hidden_layer, latent_dim):
#         super().__init__()

#         # If using unsquashed actions, actions are no longer bounded between -1 and +1.

#         self.encoder = torch.nn.Sequential(
#             torch.nn.Linear(native_dim, hidden_layer),
#             torch.nn.LeakyReLU(0.2),
#         )

#         self.mean = torch.nn.Linear(hidden_layer, latent_dim)
#         self.log_std = torch.nn.Linear(hidden_layer, latent_dim)

#         self.decoder = torch.nn.Sequential(
#             torch.nn.Linear(latent_dim, hidden_layer),
#             torch.nn.LeakyReLU(0.2),
#             torch.nn.Linear(hidden_layer, native_dim),
#             torch.nn.Tanh(),
#         )

#     def forward(self, x):
#         z = self.encoder(x)
#         mean = self.mean(z)
#         log_std = self.log_std(z)
#         std = torch.exp(log_std)
#         z = mean + std * torch.randn_like(std)
#         reconstruction = self.decoder(z)
       
#         return reconstruction, mean, std