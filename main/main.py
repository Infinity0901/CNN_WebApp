from __future__ import annotations

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

if __name__ == "__main__": 
    from torchvision.datasets import MNIST
    from torch.optim import SGD
    from torch.utils.data import DataLoader
    from itertools import cycle
    
    
    transform = T.Compose([T.ToTensor()])
    train_set = MNIST("D://CreaTech//A5//AICG//CNN_WebApp//CNN_WebApp-1//main//datasets", train = True, download=True,  transform=transform)
    test_set = MNIST("D://CreaTech//A5//AICG//CNN_WebApp//CNN_WebApp-1//main//datasets", train = False, download=True, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle = True, drop_last = True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle = False)
        
    model = nn.Sequential(
        nn.Flatten(), 
        nn.Linear(1*28*28, 128), nn.ReLU(inplace=True),
        nn.Linear(128, 10),  
    )
    optim = SGD(model.parameters(), lr=1e-3)
    
    model.train()
    train_batches = iter(cycle(train_loader))
    
    for step in range(10*len(train_loader)):
        x, l = next(train_batches)
        optim.zero_grad(set_to_none=True)
        # Before Softmax it is called logits
        logits = model(x)
        loss = F.nll_loss(torch.log_softmax(logits, dim=1), l)
        loss.backward()
        
        
        
        
        
    
    
    
    
    
    
    
    