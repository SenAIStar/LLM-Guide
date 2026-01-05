import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

class SimpleMLP(nn.Module):
   def __init__(self):
       super(SimpleMLP, self).__init__()
       self.fc1 = nn.Linear(10, 5)
       self.fc2 = nn.Linear(5, 2)

   def forward(self, x):
       x = torch.relu(self.fc1(x))
       x = self.fc2(x)
       return x
   
model = SimpleMLP().cuda()
model.train()
scaler = GradScaler()

for epoch in range(num_epochs):
    for batch in data_loader:
        x, y = batch
        x, y = x.cuda(), y.cuda()

        with autocast():
            outputs = model(x)
            loss = criterion(outputs, y)

        # 反向传播和权重更新
        # 放大梯度
        scaler.scale(loss).backward() 
        # 应用缩放后的梯度进行权重更新
        scaler.step(optimizer)
        # 更新缩放因子
        scaler.update()