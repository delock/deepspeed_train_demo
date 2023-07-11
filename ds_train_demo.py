import torch.nn as nn
import argparse
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed import comm

 

import torch
from torch.utils.data import Dataset, DataLoader

 

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size).to(torch.bfloat16)

 

    def __getitem__(self, index):
        return self.data[index]

 

    def __len__(self):
        return self.len

 

data_size = 1024
data_length = 100
rand_loader = DataLoader(dataset=RandomDataset(data_size, data_length),
                         batch_size=1,
                         shuffle=False)

 

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(1024, 1, bias=False)

 

    def forward(self, x):
        x = self.fc(x)
        return x

 

#model = MyModel().to(torch.bfloat16)
model = MyModel()
params = model.parameters()

 

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
parser.add_argument('--deepspeed_config', type=str, default='ds_config.json',
                    help='path to DeepSpeed configuration file')
cmd_args = parser.parse_args()

 

# initialize the DeepSpeed engine
model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args, model=model, model_parameters=params)

 

for step, batch in enumerate(rand_loader):
    if step % 10 == 0 and comm.get_rank() == 0:
        print (f'step={step}')
    # forward() method
    loss = model_engine(batch.to(get_accelerator().current_device_name()))
    # runs backpropagation
    model_engine.backward(loss)
    # weight update
    model_engine.step()
