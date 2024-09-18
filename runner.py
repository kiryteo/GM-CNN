import torch
import torch.nn as nn
import torch.optim as optim

from torch.cuda.amp import autocast, GradScaler
torch.set_float32_matmul_precision('medium')
import wandb

import pytorch_lightning as pl
from datetime import datetime

import hydra
from omegaconf import DictConfig

from gmcnn_base import GMCNN
from dataset.dataset_manager import DatasetManager

from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torch.nn.utils as nnutils
#from lightning.pytorch.utilities import grad_norm


def train_one_epoch(epoch_idx, model, train_loader, optimizer, loss_fn, device, scaler, overfit):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad(set_to_none=True) # memory use reduction
        
        # AMP Torch
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
        
        # loss.backward()
        scaler.scale(loss).backward()

        # Adjust learning weights
        #optimizer.step()
        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()
        
        # overfit case
        if overfit:
            last_loss = running_loss
            print('loss: {}'.format(last_loss))
            running_loss = 0.
        else:
            if i % 20 == 19:
                last_loss = running_loss / 20 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0.

    return last_loss


import os
os.environ['WANDB_MODE'] = 'dryrun'

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.experiment.seed)

    print(cfg)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    scaler = GradScaler()

    model = GMCNN(cfg).to(device)

    print("number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # get the optim and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.004, weight_decay=0.0153)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.7)

    overfit = False
    #overfit = True

    dataset_manager = DatasetManager(cfg, overfit=overfit)
    train_loader, valid_loader, test_loader = dataset_manager.get_dataloader()

    epochs = cfg.exp.trainer.max_epochs
    best_vloss = 1_000_000.

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch + 1))
        
        model.train()
        avg_loss = train_one_epoch(epoch, model, train_loader, optimizer, criterion, device, scaler, overfit)

        running_vloss = 0.0

        model.eval()

        with torch.no_grad():
            for i, vdata in enumerate(valid_loader):
                vinputs, vlabels = vdata[0].to(device), vdata[1].to(device)
                voutputs = model(vinputs)
                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch)
            torch.save(model.state_dict(), model_path)
    
        scheduler.step(avg_vloss)

    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct // total} %')


if __name__ == "__main__":
    main()


