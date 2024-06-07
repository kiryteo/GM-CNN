import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
#import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader, Subset, random_split

from itertools import permutations
import numpy as np

import wandb
import random
import time
import math

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
#from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import OneCycleLR, CyclicLR
from lightning.pytorch.utilities import grad_norm

import hydra
from omegaconf import DictConfig, OmegaConf

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
        # k_inv = pow(self.k, self.N - 1, self.N)
        k_inv = (self.N - self.k) % self.N
        return CyclicGroupElement(k_inv, self.N)

    def __str__(self):
        return f'{self.k} (mod {self.N})'

"""Dihedral group D4"""
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
        #return [DihedralElement(r, f, order) for r in range(order) for f in range(2)]
    elif group == 'cyclic':
        return [CyclicGroupElement(k, order) for k in range(order)]

def get_group_matrix(elements):
    # Precompute inverses of all elements
    inverses = [e.inverse() for e in elements]

    # Use list comprehensions to build the matrix
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


class GMConv(nn.Module):
    def __init__(self, group, order, nbr_size, group_matrix, out_channels):
        super().__init__()

        self.out_channels = out_channels

        nbrhood = generate_neighborhood(group, order, nbr_size)
        target_elements = get_nbrhood_elements(nbrhood)

        self.index_matrix = np.array([np.where(group_matrix.T == element)[1] for element in target_elements])
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self.weight_coeff = nn.Parameter(torch.empty(self.out_channels, 1, len(target_elements)))
        nn.init.kaiming_uniform_(self.weight_coeff, a=math.sqrt(5))

        self.err_vector = nn.Parameter(torch.empty(1, 4096))
        nn.init.kaiming_uniform_(self.err_vector, a=math.sqrt(5))

    def forward(self, x):
        self.err_vector.data /= torch.norm(self.err_vector.data)
        adj_input_tensor = x[:, :, 0, self.index_matrix] * self.err_vector[None, None, :, :]

        self.weight_coeff.data /= torch.norm(self.weight_coeff.data, dim=-1, keepdim=True)
        res = torch.einsum('ojk, bikl -> bojl', self.weight_coeff, adj_input_tensor)

        results = res + self.bias[None, :, None, None]
        return results

def get_nbr_elements(kron_elements, k):
    nbr_elements = [kron_elements[0]]

    for i in range(k-1):
        if i % 2 == 0:
            nbr_elements.append(kron_elements[1:][i//2])
        else:
            nbr_elements.append(kron_elements[1:][-(i//2+1)])

    return nbr_elements


def get_generator_elements(order):
    gen1_elements = []
    gen2_elements = []

    for i in range(order):
        if i % 2 == 0:
            gen1_elements.append((CyclicGroupElement(0, order), CyclicGroupElement(i, order)))
            gen2_elements.append((CyclicGroupElement(i, order), CyclicGroupElement(0, order)))

    return gen1_elements, gen2_elements

def get_dihedral_generators(order):
    generators = []
    for i in range(order):
        if i % 2 == 0:
            generators.append(DihedralElement(i, 0, order))
    for i in range(order):
        if i % 2 == 0:
            generators.append(DihedralElement(i, 1, order))

    return generators

def get_subgroup(order):
    subgroup_op = []
    # subgroup_str = []
    gen1_elements, gen2_elements = get_generator_elements(order)

    for e1 in gen1_elements:
        for e2 in gen2_elements:
            subgroup_op.append((e1[0].product(e2[0]), e1[1].product(e2[1])))
            # subgroup_str.append(e1[0].product(e2[0]).__str__() + e1[1].product(e2[1]).__str__())

    return subgroup_op

def get_dihedral_subgroup(order):
    generators = get_dihedral_generators(order)
    return [(a, b) for a, b in product(generators, repeat=2)]

def get_subgroup_cosets(subgroup_op, nbr_elements):
    cosets = []
    for element in nbr_elements:
        coset = []
        for subgroup_element in subgroup_op:
            coset.append((subgroup_element[0].product(element[0]).__str__() + subgroup_element[1].product(element[1]).__str__()))
        cosets.append(coset)

    return cosets


def get_indices(cosets, kron_prod_strings):
    indices = []
    for coset in cosets:
        coset_indices = []
        for element in coset:
            index = kron_prod_strings.index(element)
            coset_indices.append(index)
        indices.append(coset_indices)

    return indices


from itertools import product

from torchmetrics.classification import MulticlassAccuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torch.nn.utils as nnutils
#from lightning.pytorch.utilities import grad_norm


class ImplicitCNN(pl.LightningModule):
    def __init__(self, group, order, nbr, in_channels, out_channels, num_blocks, optimizer, lr, weight_decay, scheduler):
        super().__init__()

        self.save_hyperparameters() 

        vec_size = 64*64

        self.order = order
        self.num_blocks = num_blocks

        elements = generate_elements(group, order)
        M1 = get_group_matrix(elements)
        M2 = get_group_matrix(elements)

        group_matrix = np.array(kronecker_product(M1, M2))

        self.layers = nn.ModuleList([
            nn.Sequential(GMConv(group, order, nbr, group_matrix, out_channels), nn.PReLU())
            for _ in range(self.num_blocks)
        ])      

        self.conv1x1 = nn.Conv2d(20, out_channels, kernel_size=1, stride=1, padding=0)
        self.out_conv = nn.Conv2d(out_channels, 2, kernel_size=1)
        #self.out_conv = GMConv(group, order, nbr, group_matrix, 2)

        #self.optimizer = optimizer
        #self.lr = lr
        #self.weight_decay = weight_decay
        #self.scheduler_params = scheduler_params
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = x.view(-1, x.size(1), 1, x.size(-1)*x.size(-1))

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
        for y in yy.transpose(0,1):
            im = self(xx)
            xx = torch.cat([xx[:, im.shape[1]:], im], 1)
            loss += self.criterion(im, y)

        return {"loss": loss, "yy_shape": yy.shape[1]}

    def training_epoch_end(self, outputs):
        train_mse = []
        for x in outputs:
            train_mse.append(x['loss'].item() / x['yy_shape'])
        train_rmse = round(np.sqrt(np.mean(train_mse)), 5)
        self.log('train_rmse', train_rmse, on_epoch=True, prog_bar=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        xx, yy = batch
        loss = 0
        for y in yy.transpose(0,1):
            im = self(xx)
            xx = torch.cat([xx[:, im.shape[1]:], im], 1)
            loss += self.criterion(im, y)

        return {"loss": loss, "yy_shape": yy.shape[1]}

    def validation_epoch_end(self, outputs):
        valid_mse = []
        for x in outputs:
            valid_mse.append(x['loss'].item() / x['yy_shape'])
        valid_rmse = round(np.sqrt(np.mean(valid_mse)), 5)
        self.log('valid_rmse', valid_rmse, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        xx, yy = batch
        loss = 0
        for y in yy.transpose(0,1):
            im = self(xx)
            xx = torch.cat([xx[:, im.shape[1]:], im], 1)
            loss += self.criterion(im, y)

        return {"loss": loss, "yy_shape": yy.shape[1]}

    def test_epoch_end(self, outputs):
        test_mse = []
        for x in outputs:
            test_mse.append(x['loss'].item() / x['yy_shape'])
        test_rmse = round(np.sqrt(np.mean(test_mse)), 5)
        self.log('test_rmse', test_rmse, on_epoch=True, prog_bar=True, sync_dist=True)
    
    def configure_optimizers(self):
        #print(self.hparams)
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

        scheduler_name = self.hparams.scheduler.name
        scheduler_params = self.hparams.scheduler.params

        if scheduler_name == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=scheduler_params.mode,
                factor=scheduler_params.factor,
                patience=scheduler_params.patience,
                min_lr=scheduler_params.min_lr,
                verbose=scheduler_params.verbose
            )
        elif scheduler_name == 'OneCycleLR':
            scheduler = OneCycleLR(
                optimizer,
                max_lr=scheduler_params.max_lr,
                steps_per_epoch=scheduler_params.steps_per_epoch,
                epochs=scheduler_params.epochs
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': scheduler_params.monitor_metric
        }

    ##def configure_optimizers(self):
    #    optimizer = optim.AdamW(self.parameters(), lr=0.004, weight_decay=0.0153)#, amsgrad=True)#, weight_decay=1e-3) use wd with Adam
    #    #optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.85, weight_decay=0.0153, nesterov=True)
    #    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    #    #scheduler = CyclicLR(optimizer, base_lr=0.009, max_lr=0.1, cycle_momentum=False)
    #    #scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=4000//(16*8), epochs=15)
    #    return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'valid_rmse'}

class EqDataset(Dataset):
    def __init__(self, input_length, mid, output_length, direc, task_list, sample_list, stack = False):
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


# Split time ranges
train_time = list(range(0, 160))
valid_time = list(range(160, 200))
test_future_time = list(range(200, 250))
test_domain_time = list(range(0, 100))

# Split tasks
# if args.dataset == "PhiFlow":

symmetry = "Rotation"

#symmetry = "Scale"

h_size, w_size = 64, 64

if symmetry == "Translation":
    train_task = [(48, 10), (56, 10), (8, 20),  (40, 5),  (56, 25),
                (48, 20), (48, 5),  (16, 20), (56, 5),  (32, 10),
                (56, 15), (16, 5),  (40, 15), (40, 25), (48, 25),
                (48, 15), (24, 10), (56, 20), (32, 15), (16, 15),
                (8, 10),  (24, 15), (8, 15),  (32, 25), (8, 5)]

    test_domain_task = [(32, 20), (32, 5), (24, 20), (16, 25), (24, 5),
                        (16, 10), (40, 20), (8, 25), (24, 25), (40, 10)]


elif symmetry == "Rotation":
    train_task = [(27, 2), (33, 0), (3, 2), (28, 3),(9, 0),
                    (12, 3), (22, 1), (8, 3), (30, 1), (25, 0),
                    (16, 3), (11, 2), (23, 2), (29, 0), (36, 3),
                    (26, 1), (1, 0), (35, 2), (19, 2), (34, 1),
                    (4, 3), (2, 1), (7, 2), (31, 2), (17, 0)]

    test_domain_task = [(6, 1), (14, 1), (15, 2), (10, 1), (18, 1),
                        (20, 3), (24, 3), (13, 0), (21, 0), (5, 0)]

import os
os.environ['WANDB_MODE'] = 'dryrun'

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

@hydra.main(config_path=".", config_name="corr-plume-config")
def main(cfg: DictConfig):
    pl.seed_everything(42)

    print(OmegaConf.to_yaml(cfg))

    #implicit_cnn = ImplicitCNN(**cfg.model)
    implicit_cnn = ImplicitCNN(
            group=cfg.model.group,
            order=cfg.model.order,
            nbr=cfg.model.nbr,
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            num_blocks=cfg.model.num_blocks,
            optimizer=cfg.model.optimizer,
            lr=cfg.model.optimizer.params.lr,
            weight_decay=cfg.model.optimizer.params.weight_decay,
            scheduler=cfg.scheduler
        )

    valid_task = train_task
    test_future_task = train_task

    #data_direc = '/localscratch/asa420/Approximately-Equivariant-Nets-master/PhiFlow/Rotation'
    data_direc = cfg.data.direc

    train_set = EqDataset(input_length = 10,
                        mid = 22,
                        output_length = 6,
                        direc = data_direc,
                        task_list = train_task,
                        sample_list = train_time,
                        stack = True)

    valid_set = EqDataset(input_length = 10,
                        mid = 22,
                        output_length = 6,
                        direc = data_direc,
                        task_list = valid_task,
                        sample_list = valid_time,
                        stack = True)

    test_set_future = EqDataset(input_length = 10,
                            mid = 22,
                            output_length = 20,
                            direc = data_direc,
                            task_list = test_future_task,
                            sample_list = test_future_time,
                            stack = True)

    test_set_domain = EqDataset(input_length = 10,
                            mid = 22,
                            output_length = 20,
                            direc = data_direc,
                            task_list = test_domain_task,
                            sample_list = test_domain_time,
                            stack = True)

    train_loader = DataLoader(train_set, batch_size=cfg.data.batch_size, shuffle = True, num_workers = cfg.data.num_workers)
    valid_loader = DataLoader(valid_set, batch_size=cfg.data.batch_size, shuffle = False, num_workers = cfg.data.num_workers)
    test_loader_future = DataLoader(test_set_future,  batch_size = cfg.data.batch_size, shuffle=False, num_workers =cfg.data.num_workers)
    test_loader_domain = DataLoader(test_set_domain,  batch_size = cfg.data.batch_size, shuffle=False, num_workers =cfg.data.num_workers)

    wandb_logger = WandbLogger(project=cfg.logger.project, log_model=cfg.logger.log_model)
    wandb_logger.experiment.config['batch_size'] = cfg.data.batch_size
    wandb_logger.watch(implicit_cnn, log='all')

    #wandb_logger = WandbLogger(project='Partial_equivariance', log_model="gradients")

    #checkpoint_callback = CustomModelCheckpoint(**cfg.callbacks.model_checkpoint)
    #lr_monitor = CustomLearningRateMonitor(**cfg.callbacks.lr_monitor)

    #wandb_logger.experiment.config['batch_size'] = cfg.data.batch_size
    #wandb_logger.watch(implicit_cnn, log='all')

    checkpoint_filename = f"Plume_Rot_{cfg.model.group}_order{cfg.model.order}_nbr{cfg.model.nbr}_ch{cfg.model.out_channels}_blocks{cfg.model.num_blocks}_epochs{cfg.trainer.max_epochs}_{cfg.scheduler.name}"


    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=cfg.callbacks.checkpoint.dirpath,
        filename=checkpoint_filename,
        monitor=cfg.callbacks.checkpoint.monitor,
        mode=cfg.callbacks.checkpoint.mode
    )

    #checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath='/localscratch/asa420/Approximately-Equivariant-Nets-master/chkpt/rot/', filename='nbr2-rot-128ch-cycleLR', monitor="valid_rmse", mode="min")
    #cfg.callbacks.model_checkpoint
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    callbacks = [checkpoint_callback, lr_monitor]

    if cfg.callbacks.early_stop.enabled:
        early_stop_callback = EarlyStopping(
            monitor=cfg.callbacks.early_stop.monitor,
            patience=cfg.callbacks.early_stop.patience,
            mode=cfg.callbacks.early_stop.mode
        )
        callbacks.append(early_stop_callback)

    #cfg.callbacks.lr_monitor
    #early_stop = EarlyStopping(monitor='valid_rmse', patience=6, mode='min')

    #trainer = pl.Trainer(**cfg.trainer, logger=wandb_logger, callbacks=[checkpoint_callback, lr_monitor])#, early_stop])
    trainer = pl.Trainer(**cfg.trainer, logger=wandb_logger, callbacks=callbacks)
    trainer.fit(implicit_cnn, train_loader, valid_loader)

    tester = pl.Trainer(**cfg.tester)
    #model = ImplicitCNN.load_from_checkpoint('/localscratch/asa420/Approximately-Equivariant-Nets-master/chkpt/rot/nbr2-rot-128ch-cycleLR.ckpt')
    #model = ImplicitCNN.load_from_checkpoint(cfg.callbacks.checkpoint.best_model_path)
    model = ImplicitCNN.load_from_checkpoint(os.path.join(cfg.callbacks.checkpoint.dirpath, f"{checkpoint_filename}.ckpt"))
    model.eval()
    tester.test(model, dataloaders=test_loader_domain)
    tester.test(model, dataloaders=test_loader_future)


if __name__ == "__main__":
    main()
