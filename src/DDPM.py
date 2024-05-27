import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import UNet
import tqdm
from create_kanji_dataset import Dataset
import matplotlib.pyplot as plt
import os
import wandb
import torchvision
from dotenv import load_dotenv
load_dotenv()

assert os.getenv("WANDB_API_KEY") is not None, "Please set WANDB_API_KEY in .env file"
wandb.login()


class DDPM(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T
        self.beta1 = 1e-4
        self.betaT = 0.02
        self.betas = torch.linspace(self.beta1, self.betaT, T)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, 0)

    def forward_process(self, x0, t=None):
        device = x0.device
        self._transfer_device(x0, device)
        if t is None:
            t = torch.randint(1, self.T, size=(x0.shape[0],)).to(device)
        noise = torch.randn_like(x0).to(device)
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
        return xt, t, noise
    
    def backward_process(self, model: nn.Module, xt: torch.Tensor):
        device = xt.device
        self._transfer_device(xt, device)
        batch_size = xt.shape[0]
        model.eval()
        with torch.no_grad():
            time_steps = tqdm.tqdm(reversed(range(1, self.T)), desc="backward process")
            for t in time_steps:
                time_tensor = torch.full((batch_size,), t, dtype=torch.long, device=xt.device)
                predicted_noise = model(xt, time_tensor)
                xt = self._calc_noise_per_step(xt, time_tensor, predicted_noise)
        model.train()
        xt = torch.clamp(xt, -1, 1)
        xt = (xt + 1) / 2
        return xt
    
    def _transfer_device(self, x, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)



    def _calc_noise_per_step(self, xt, t, predicted_noise):
        beta = self.betas[t].reshape(-1, 1, 1, 1)
        sqrt_alpha = torch.sqrt(self.alphas[t]).reshape(-1, 1, 1, 1)
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        prev_alpha_bar = self.alpha_bars[t-1].reshape(-1, 1, 1, 1)
        sigma_t = torch.sqrt(((1 - prev_alpha_bar) / (1 - alpha_bar)) * beta)
        noise = torch.randn_like(xt) if t[0].item() > 0 else torch.zeros_like(xt)
        x_t_1 = 1 / sqrt_alpha * (xt - (beta / torch.sqrt(1 - alpha_bar)) * predicted_noise) + sigma_t * noise
        return x_t_1

def ddpm_train():
    wandb.init(project="ddpm", group="kanji_ddpm", name="moji_ddpm")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(c_in=1, c_out=1).to(device)
    ddpm = DDPM(T=1000).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.MSELoss()
    dataset = Dataset(pix=32, font_file="./src/gomarice_mukasi_mukasi.ttf")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    num_epochs = 100
    epoch_losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for x in tqdm.tqdm(dataloader):
            x = x.to(device)
            xt, t, noise = ddpm.forward_process(x)
            optimizer.zero_grad()
            predicted_noise = model(xt, t)
            loss = criterion(predicted_noise, noise)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
        epoch_loss /= len(dataloader)
        print(f"epoch: {epoch}, loss: {epoch_loss}")
        epoch_losses.append(epoch_loss)
        # plot the generated image
        if epoch % 10 == 0:
            model.eval()
            noises = torch.randn(16, 1, 32, 32).to(device)
            generated_images = ddpm.backward_process(model, noises)
            generated_images = generated_images.cpu().detach()
            grid_images = torchvision.utils.make_grid(generated_images, nrow=4)
            wandb.log({"generated_images": [wandb.Image(grid_images)],
                        "loss": epoch_loss,
                        "epoch": epoch})
        else:
            wandb.log({"loss": epoch_loss,
                        "epoch": epoch})

    
    
    # save the model
    torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    ddpm_train()