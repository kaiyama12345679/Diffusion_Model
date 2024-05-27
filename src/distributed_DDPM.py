import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import deepspeed
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
from DDPM import DDPM

ds_config = {
    "train_batch_size": 128,
    "gradient_accumulation_steps": 1,
    "gradient_clipping": 1.0,
    "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-4,
            },
        },
    "fp16": {
        "enabled": False,
    },
    "zero_optimization": {
        "stage": 2,
        "reduce_bucket_size": 2e8,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        }
    }
}

def distributed_train():
    dataset = Dataset(pix=32, font_file="./src/gomarice_mukasi_mukasi.ttf")
    model = UNet(c_in=1, c_out=1)
    ddpm = DDPM(T=1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model_engine, _, _, _ = deepspeed.initialize(config=ds_config, model=model, model_parameters=model.parameters())

    if dist.get_rank() == 0:
        wandb.login()
        wandb.init(project="distributed_DDPM", name="DDPM")
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=sampler)
    for epoch in range(100):
        epoch_loss = 0
        for x in dataloader:
            x = x.cuda()
            xt, t, noise = ddpm.forward_process(x)
            predicted_noise = model_engine(xt, t)
            loss = F.mse_loss(predicted_noise, noise)
            model_engine.backward(loss)
            epoch_loss += loss.item()
            model_engine.step()
        epoch_loss /= len(dataloader)
        if dist.get_rank() == 0:
            
            if epoch % 10 == 0:
                # plot the generated image
                xT = torch.randn(16, 1, 32, 32).cuda()
                generated_images = ddpm.backward_process(model_engine.module, xT)
                generated_images = generated_images.cpu().detach()
                grid_images = torchvision.utils.make_grid(generated_images, nrow=4)
                wandb.log({"generated_images": [wandb.Image(grid_images)],
                            "loss": epoch_loss,
                            "epoch": epoch})
            else:
                wandb.log({"loss": epoch_loss,
                            "epoch": epoch})
                
        # save the model
        if epoch % 50 == 0:
            model_engine.save_checkpoint(f"model_{epoch}.pt")


if __name__ == "__main__":
    distributed_train()
            