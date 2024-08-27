import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, OneCycleLR
import hydra
from omegaconf import DictConfig
import numpy as np
import random
import math

# Group Element Classes and Functions
class GroupElement:
    def __init__(self, *args):
        pass

    def product(self, other):
        pass

    def inverse(self):
        pass

    def __str__(self):
        pass

class CyclicGroupElement(GroupElement):
    def __init__(self, k, N):
        self.k = k % N
        self.N = N

    def product(self, other):
        k_hat = (self.k + other.k) % self.N
        return CyclicGroupElement(k_hat, self.N)

    def inverse(self):
        k_inv = (self.N - self.k) % self.N
        return CyclicGroupElement(k_inv, self.N)

    def __str__(self):
        return f'{self.k} (mod {self.N})'

class DihedralElement(GroupElement):
    def __init__(self, r, f, n):
        self.n = n
        self.r = r % n
        self.f = f % 2

    def product(self, other):
        if other.f == 0:
            return DihedralElement(self.r + other.r, self.f, self.n)
        else:
            return DihedralElement(self.n - self.r + other.r, 1 - self.f, self.n)

    def inverse(self):
        if self.f == 0:
            return DihedralElement(-self.r, 0, self.n)
        else:
            return DihedralElement(self.r, 1, self.n)

    def __str__(self):
        return f'f^{self.f} * r^{self.r}' if self.f else f'r^{self.r}' 

def generate_elements(group, order):
    if group == 'dihedral':
        elements_rotations = [DihedralElement(r, 0, order) for r in range(order)]
        elements_flips = [DihedralElement(r, 1, order) for r in range(order)]
        return elements_rotations + elements_flips
    elif group == 'cyclic':
        return [CyclicGroupElement(k, order) for k in range(order)]

def get_group_matrix(elements):
    inverses = [e.inverse() for e in elements]
    mat = np.array([[f'{inverses[k].product(elements[j])}' for j in range(len(elements))] for k in range(len(elements))])
    return mat

def kronecker_product(M1, M2):
    return [
        [entry1 + entry2 for entry1 in row1 for entry2 in row2]
        for row1 in M1 for row2 in M2
    ]

def generate_neighborhood(group, n, t):
    neighborhood = []
    if group == 'dihedral':
        for i in range(-t, t + 1):
            r_element = DihedralElement(i, 0, n)
            f_element = DihedralElement(i, 1, n)
            neighborhood.append(r_element.__str__())
            neighborhood.append(f_element.__str__())
    elif group == 'cyclic':
        for i in range(-t, t+1):
            element = CyclicGroupElement(i, n)
            neighborhood.append(element.__str__())
    return neighborhood

def get_nbrhood_elements(nbrhood):
    return [e1 + e2 for e1 in nbrhood for e2 in nbrhood]

def get_nbr_distances(group, order):
    if group == 'cyclic':
        center = (order//2, order//2)
        valid_range = range(order)
    elif group == 'dihedral':
        center = (order, order)
        valid_range = range(order*2)
    else:
        print('Currently supported groups: dihedral, permutation, and cyclic')
        return None

    distances = []
    for x in valid_range:
        for y in valid_range:
            distance = math.sqrt((x - center[0])**2 + (y - center[1])**2)
            distances.append((distance, (x, y)))
    distances.sort()
    return distances

def get_central_indices(group, order, out_channels):
    elements = generate_elements(group, order)
    distances = get_nbr_distances(group, order)
    if group == 'cyclic':
        central_indices = [f'{point[0]} (mod {order}){point[1]} (mod {order})' for _, point in distances[:out_channels]]
    elif group == 'dihedral':
        central_indices = [f'{elements[point[0]].__str__()}{elements[point[1]].__str__()}' for _, point in distances[:out_channels]]
    return central_indices

class GMConv(nn.Module):
    def __init__(self, group, order, nbr_size, group_matrix, out_channels):
        super().__init__()
        self.out_channels = out_channels

        if group == 'cyclic':
            vec_size = order * order
        elif group == 'dihedral':
            vec_size = order * order * 4

        nbrhood = generate_neighborhood(group, order, nbr_size)
        target_elements = get_nbrhood_elements(nbrhood)

        self.index_matrix = np.array([np.where(group_matrix.T == element)[1] for element in target_elements])
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self.weight_coeff = nn.Parameter(torch.empty(self.out_channels, 1, len(target_elements)))
        nn.init.kaiming_uniform_(self.weight_coeff, a=math.sqrt(5))

        self.err_vector = nn.Parameter(torch.empty(1, vec_size))
        nn.init.kaiming_uniform_(self.err_vector, a=math.sqrt(5))

    def forward(self, x):
        self.err_vector.data /= torch.norm(self.err_vector.data)
        adj_input_tensor = x[:, :, 0, self.index_matrix] * self.err_vector[None, None, :, :]

        self.weight_coeff.data /= torch.norm(self.weight_coeff.data, dim=-1, keepdim=True)
        res = torch.einsum('ojk, bikl -> bojl', self.weight_coeff, adj_input_tensor)

        results = res + self.bias[None, :, None, None]
        return results

class ImplicitCNN(LightningModule):
    def __init__(self, group, order, nbr, in_channels, out_channels, num_blocks, lr, weight_decay, scheduler_params):
        super().__init__()
        self.save_hyperparameters()

        self.num_blocks = num_blocks

        elements = generate_elements(group, order)
        GM1 = get_group_matrix(elements)
        GM2 = get_group_matrix(elements)
        group_matrix = np.array(kronecker_product(GM1, GM2))

        self.layers = nn.ModuleList([
            nn.Sequential(GMConv(group, order, nbr, group_matrix, out_channels), nn.PReLU())
            for _ in range(self.num_blocks)
        ])

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.out_conv = nn.Conv2d(out_channels, 2, kernel_size=1)
        self.criterion = nn.MSELoss()

        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_params = scheduler_params

    def forward(self, x):
        x = x.view(-1, x.size(1), 1, x.size(-1) * x.size(-1))
        out = self.conv1x1(x)
        for layer in self.layers:
            x = layer(x)
            x += out
            out = x
        out = self.out_conv(out)
        logits = out.view(-1, out.size(1), 64, 64)
        return logits

    def training_step(self, batch, batch_idx):
        xx, yy = batch
        loss = 0
        for y in yy.transpose(0, 1):
            im = self(xx)
            xx = torch.cat([xx[:, im.shape[1]:], im], 1)
            loss += self.criterion(im, y)
        return {"loss": loss, "yy_shape": yy.shape[1]}

    def training_epoch_end(self, outputs):
        train_mse = [x['loss'].item() / x['yy_shape'] for x in outputs]
        train_rmse = round(np.sqrt(np.mean(train_mse)), 5)
        self.log('train_rmse', train_rmse, on_epoch=True, prog_bar=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        xx, yy = batch
        loss = 0
        for y in yy.transpose(0, 1):
            im = self(xx)
            xx = torch.cat([xx[:, im.shape[1]:], im], 1)
            loss += self.criterion(im, y)
        return {"loss": loss, "yy_shape": yy.shape[1]}

    def validation_epoch_end(self, outputs):
        valid_mse = [x['loss'].item() / x['yy_shape'] for x in outputs]
        valid_rmse = round(np.sqrt(np.mean(valid_mse)), 5)
        self.log('valid_rmse', valid_rmse, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        xx, yy = batch
        loss = 0
        for y in yy.transpose(0, 1):
            im = self(xx)
            xx = torch.cat([xx[:, im.shape[1]:], im], 1)
            loss += self.criterion(im, y)
        return {"loss": loss, "yy_shape": yy.shape[1]}

    def test_epoch_end(self, outputs):
        test_mse = [x['loss'].item() / x['yy_shape'] for x in outputs]
        test_rmse = round(np.sqrt(np.mean(test_mse)), 5)
        self.log('test_rmse', test_rmse, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer_name = self.hparams.optimizer.name
        optimizer_params = self.hparams.optimizer.params

        if optimizer_name == 'AdamW':
            optimizer = optim.AdamW(self.parameters(), **optimizer_params)
        elif optimizer_name == 'Adam':
            optimizer = optim.Adam(self.parameters(), **optimizer_params)
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(self.parameters(), **optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        scheduler_params = self.scheduler_params
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode=scheduler_params.mode, 
            factor=scheduler_params.factor, 
            patience=scheduler_params.patience, 
            verbose=scheduler_params.verbose
        )

        return {
            'optimizer': optimizer, 
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': scheduler_params.monitor_metric
            }
        }

class EqDataset(Dataset):
    def __init__(self, input_length, mid, output_length, direc, task_list, sample_list, stack=False):
        self.input_length = input_length
        self.mid = mid
        self.output_length = output_length
        self.direc = direc
        self.task_list = task_list
        self.sample_list = sample_list
        self.stack = stack
        try:
            self.data_lists = [torch.load(self.direc + "/raw_data_" + str(idx[0]) + "_" + str(idx[1]) + ".pt") for idx in task_list]
        except:
            self.data_lists = [torch.load(self.direc + "/raw_data_" + str(idx) + ".pt") for idx in task_list]

    def __len__(self):
        return len(self.task_list) * len(self.sample_list)

    def __getitem__(self, index):
        task_idx = index // len(self.sample_list)
        sample_idx = index % len(self.sample_list)
        y = self.data_lists[task_idx][(self.sample_list[sample_idx]+self.mid):(self.sample_list[sample_idx]+self.mid+self.output_length)]
        if not self.stack:
            x = self.data_lists[task_idx][(self.mid-self.input_length+self.sample_list[sample_idx]):(self.mid+self.sample_list[sample_idx])]
        else:
            x = self.data_lists[task_idx][(self.mid-self.input_length+self.sample_list[sample_idx]):(self.mid+self.sample_list[sample_idx])].reshape(-1, y.shape[-2], y.shape[-1])
        return x.float(), y.float()

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(42)

    implicit_cnn = ImplicitCNN(
        group=cfg.model.group,
        order=cfg.model.order,
        nbr=cfg.model.nbr,
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        num_blocks=cfg.model.num_blocks,
        lr=cfg.model.optimizer.params.lr,
        weight_decay=cfg.model.optimizer.params.weight_decay,
        scheduler_params=cfg.scheduler
    )

    data_direc = cfg.data.direc

    train_set = EqDataset(
        input_length=cfg.data.input_length, 
        mid=cfg.data.mid, 
        output_length=cfg.data.output_length,
        direc=data_direc, 
        task_list=cfg.data.train_task, 
        sample_list=cfg.data.train_time, 
        stack=cfg.data.stack
    )

    valid_set = EqDataset(
        input_length=cfg.data.input_length, 
        mid=cfg.data.mid, 
        output_length=cfg.data.output_length, 
        direc=data_direc, 
        task_list=cfg.data.valid_task,
        sample_list=cfg.data.valid_time, 
        stack=cfg.data.stack
    )

    test_set_future = EqDataset(
        input_length=cfg.data.input_length, 
        mid=cfg.data.mid, 
        output_length=cfg.data.output_length, 
        direc=data_direc, 
        task_list=cfg.data.test_future_task, 
        sample_list=cfg.data.test_future_time, 
        stack=cfg.data.stack
    )

    test_set_domain = EqDataset(
        input_length=cfg.data.input_length, 
        mid=cfg.data.mid, 
        output_length=cfg.data.output_length,
        direc=data_direc, 
        task_list=cfg.data.test_domain_task, 
        sample_list=cfg.data.test_domain_time, 
        stack=cfg.data.stack
    )

    train_loader = DataLoader(train_set, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    valid_loader = DataLoader(valid_set, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    test_loader_future = DataLoader(test_set_future, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    test_loader_domain = DataLoader(test_set_domain, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    wandb_logger = WandbLogger(project=cfg.logger.project, log_model=cfg.logger.log_model)
    wandb_logger.experiment.config['batch_size'] = cfg.data.batch_size
    wandb_logger.watch(implicit_cnn, log='all')

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.callbacks.checkpoint.dirpath,
        filename=cfg.callbacks.checkpoint.filename,
        monitor=cfg.callbacks.checkpoint.monitor,
        mode=cfg.callbacks.checkpoint.mode
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    callbacks = [checkpoint_callback, lr_monitor]

    if cfg.callbacks.early_stop.enabled:
        early_stop_callback = EarlyStopping(
            monitor=cfg.callbacks.early_stop.monitor,
            patience=cfg.callbacks.early_stop.patience,
            mode=cfg.callbacks.early_stop.mode
        )
        callbacks.append(early_stop_callback)

    trainer = Trainer(
        **cfg.trainer,
        logger=wandb_logger,
        callbacks=callbacks
    )

    trainer.fit(implicit_cnn, train_loader, valid_loader)

    tester = Trainer(**cfg.tester)
    model = ImplicitCNN.load_from_checkpoint(cfg.callbacks.checkpoint.best_model_path)
    model.eval()
    tester.test(model, dataloaders=test_loader_domain)
    tester.test(model, dataloaders=test_loader_future)

if __name__ == "__main__":
    main()
