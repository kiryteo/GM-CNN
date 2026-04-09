from pathlib import Path
import sys
import types

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.dataset_manager import DatasetManager
from gmcnn_base import GMCNN


def build_cfg(dataset: str):
    model = types.SimpleNamespace(
        dataset=dataset,
        group="cyclic",
        order=32,
        nbr=3,
        lr=0.003,
        dropout=0.2,
        num_classes=10,
        blocks=[types.SimpleNamespace(num_layers=1, out_channels=[8])],
    )
    data = types.SimpleNamespace(path="/tmp", batch_size=2, num_workers=0)
    overfit = types.SimpleNamespace(bs=1)
    exp = types.SimpleNamespace(model=model, data=data, overfit=overfit)
    return types.SimpleNamespace(exp=exp)


def test_dataset_manager_supports_cifar_and_rot_names():
    cifar_cfg = build_cfg("cifar10")
    rot_cfg = build_cfg("rot")

    cifar_manager = DatasetManager(cifar_cfg)
    rot_manager = DatasetManager(rot_cfg)

    assert hasattr(cifar_manager, "_split_dataset")
    assert rot_manager.cfg.exp.model.dataset == "rot"


def test_dataset_manager_routes_rot_alias(monkeypatch):
    cfg = build_cfg("rot")
    manager = DatasetManager(cfg)

    monkeypatch.setattr(manager, "_load_rot_mnist", lambda: "rot-loader")

    assert manager.get_dataloader() == "rot-loader"


def test_gmcnn_cifar_forward_runs():
    cfg = build_cfg("cifar10")

    output = GMCNN(cfg)(torch.randn(2, 3, 32, 32))

    assert output.shape == (2, 10)
