import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model.unet import UNet
from tools import train_loop, eval_loop, save_checkpoint
from dataset import get_datasets
from logger import Logging

train_epoch = 10
learning_rate = 0.001
batch_size = 64
eval_batch_size = 100

logger = Logging('mlops_fin','pytorch','unet',learning_rate,batch_size,train_epoch,'adam')

start = time.time()
first_time = start

train_ds, test_ds = get_datasets()

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(num_classes=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_path = os.path.abspath("./ckpt")
os.makedirs(train_path, exist_ok=True)

used_bytes = torch.cuda.memory_allocated(device)
mem = used_bytes / 1024**2

logger.log('train',-1,0,0,time.time()-first_time,time.time()-first_time,mem)
logger.log('test',-1,0,0,time.time()-first_time,time.time()-first_time,mem)

for epoch in range(train_epoch):
    start = time.time()
    metrics = train_loop(model=model, train_loader=train_loader, optimizer=optimizer, epoch=epoch, device=device)
    torch.cuda.synchronize(device)
    used_bytes = torch.cuda.memory_allocated(device)
    mem = used_bytes / 1024**2
    logger.log('train',epoch,metrics['loss'],metrics['accuracy'],time.time()-start,time.time()-first_time,mem)
    
    start = time.time()
    metrics = eval_loop(model, test_loader, device)
    torch.cuda.synchronize(device)
    used_bytes = torch.cuda.memory_allocated(device)
    mem = used_bytes / 1024**2
    logger.log('test',epoch,metrics['loss'],metrics['accuracy'],time.time()-start,time.time()-first_time,mem)
    
    save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, path=train_path, max_to_keep=3)