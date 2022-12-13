import torch

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