import torch


class AE(torch.nn.Module):
    def __init__(self, inp_dim, enc_dim, noise_factor):
        super().__init__()

        self.inp_dim = inp_dim
        self.enc_dim = enc_dim
        self.noise_factor = noise_factor
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(inp_dim, enc_dim)
        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(enc_dim, inp_dim),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)


class FC(torch.nn.Module):
    def __init__(self, inp_dim):
        super().__init__()
        
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(inp_dim, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


class AEFCN(torch.nn.Module):
    def __init__(self, ae, fc):
        super().__init__()
        
        self.ae = ae
        self.fc = fc

    def forward(self, x):
        enc = self.ae.encode(x[:, :self.ae.inp_dim])
        return self.fc(torch.cat((enc, x[:, self.ae.inp_dim:]), dim=1))
