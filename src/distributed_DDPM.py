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
from PIL import Image
from torchvision import datasets, transforms
from dotenv import load_dotenv
load_dotenv()
torch.multiprocessing.set_start_method("spawn")

assert os.getenv("WANDB_API_KEY") is not None, "Please set WANDB_API_KEY in .env file"
from DDPM import DDPM

class DatasetFromFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.imgs = [os.path.join(root, img) for img in os.listdir(root)]

    def __getitem__(self, index):
        path = self.imgs[index]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)

ds_config = {
    "train_batch_size": 32,
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

def distributed_train(args):
    deepspeed.init_distributed()
    transform = transforms.Compose([
        transforms.Resize(64),  # 生成モデルの入力サイズに合わせてリサイズ
        transforms.CenterCrop(64),  # 64x64の中央部分を切り出し
        transforms.ToTensor(),  # PIL ImageからTensorに変換
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 正規化
    ])
    dataset = DatasetFromFolder("./data/img_align_celeba", transform=transform)
    model = UNet(c_in=3, c_out=3)
    ddpm = DDPM(T=1000)
    model_engine, _, dataloader, _ = deepspeed.initialize(args=args, config=ds_config, model=model, model_parameters=model.parameters(), training_data=dataset)

    if dist.get_rank() == 0:
        wandb.login()
        wandb.init(project="distributed_DDPM", name="image-DDPM-fuga")
        world_size = dist.get_world_size()
        wandb.log({"world_size": world_size})
        # Save World size
        with open("world_size.txt", "w") as f:
            f.write(str(world_size))
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
                xT = torch.randn(36, 3, 64, 64).cuda()
                generated_images = ddpm.backward_process(model_engine.module, xT)
                generated_images = generated_images.cpu().detach()
                grid_images = torchvision.utils.make_grid(generated_images, nrow=6)
                wandb.log({"generated_images": [wandb.Image(grid_images)],
                            "loss": epoch_loss,
                            "epoch": epoch})
            else:
                wandb.log({"loss": epoch_loss,
                            "epoch": epoch})
                
        # save the model
        if epoch % 5 == 0:
            model_engine.save_checkpoint("model-hoge")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    distributed_train(args)
            