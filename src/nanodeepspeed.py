import torch
import torch.nn as nn
import torch.optim as optim
import deepspeed

# 簡単なモデルの定義
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# データの準備
inputs = torch.randn(1200, 10)  # バッチサイズ64、入力次元10
targets = torch.randn(1200, 1)  # 出力次元1

# モデルの初期化
model = SimpleModel()

# DeepSpeedの設定
ds_config = {
  "train_batch_size": 40,
  "train_micro_batch_size_per_gpu": 10,
  "gradient_accumulation_steps": 1,
  "optimizer": {
      "type": "AdamW",
      "params": {
          "lr": 0.0001,
          "betas": [
            0.9,
            0.95
          ],
          "weight_decay": 0.1
      }
  },
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
        "device": "cpu",
        "pin_memory": True
    },
    "allgather_partitions": True,
    "allgather_bucket_size": 2e8,
    "overlap_comm": True,
    "reduce_scatter": True,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": True
  }
}
deepspeed.init_distributed(dist_backend='nccl')
# DeepSpeedの初期化
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config,
)

# 損失関数の定義
criterion = nn.MSELoss()

# トレーニングループ
for epoch in range(10): # 10エポックのトレーニング
    model_engine.train()
    optimizer.zero_grad()
    
    outputs = model_engine(inputs.cuda())
    loss = criterion(outputs, targets.cuda())
    
    model_engine.backward(loss)
    model_engine.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')