import torch

import torch.nn as nn




class SimpleUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.sigmo = nn.Sigmoid()

        # Encoder
        self.e11 = nn.Conv2d(1, 64, kernel_size=3, padding=1) # output: 570x570x64
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 568x568x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 284x284x64

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # output: 282x282x128
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 280x280x128


        # Decoder

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, output_padding=(1,1))
        self.d11 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # self.upconv2 = nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2, output_padding=(0,1))
        # self.d21 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.d22 = nn.Conv2d(1, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, 1, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        print(x.shape)
        xe11 = self.relu(self.e11(x))
        xe12 = self.relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = self.relu(self.e21(xp1))
        xe22 = self.relu(self.e22(xe21))
        print(xe22.shape)

        # Decoder
        xu1 = self.upconv1(xe22)
        xu11 = torch.cat([xu1, xe12], dim=1)
        xd11 = self.relu(self.d11(xu11))
        xd12 = self.relu(self.d12(xd11))

        # xu2 = self.upconv1(xd12)
        # xu21 = torch.cat([xu2, xe12], dim=1)
        # xd21 = self.relu(self.d11(xu21))
        # xd22 = self.relu(self.d12(xd21))

        # Output layer
        out = self.outconv(xd12)
        print(out.shape)
        print()

        return self.sigmo(out)
    
    def load_model(self, model_path, device):
        model = SimpleUnet().to(device)
        # Ajouter map_location pour charger le modèle sur le CPU
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        return model


    def test_model(self, model, test_loader, device):
        model.eval()
        all_output = []
        # all_target = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                all_output.append(output[0][0].cpu().numpy())
                # all_target.append(target)
        return all_output#, all_target