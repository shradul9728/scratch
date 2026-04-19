"""Tests for utils/dataset.py"""
import torch
import pytest

from utils.dataset import TextDataset, train_val_split, get_dataloader


class TestTextDataset:
    def test_length(self):
        ids = list(range(100))
        ds = TextDataset(ids, block_size=10)
        assert len(ds) == 90  # 100 - 10

    def test_shapes(self):
        ids = list(range(100))
        ds = TextDataset(ids, block_size=10)
        x, y = ds[0]
        assert x.shape == (10,)
        assert y.shape == (10,)

    def test_y_is_shifted_x(self):
        ids = list(range(20))
        ds = TextDataset(ids, block_size=5)
        x, y = ds[0]
        # y should be x shifted right by 1
        assert x.tolist() == [0, 1, 2, 3, 4]
        assert y.tolist() == [1, 2, 3, 4, 5]

    def test_last_sample(self):
        ids = list(range(20))
        ds = TextDataset(ids, block_size=5)
        x, y = ds[len(ds) - 1]
        assert x.tolist() == [14, 15, 16, 17, 18]
        assert y.tolist() == [15, 16, 17, 18, 19]


class TestTrainValSplit:
    def test_default_split(self):
        ids = list(range(1000))
        train_ids, val_ids = train_val_split(ids)
        assert len(train_ids) == 900
        assert len(val_ids) == 100
        assert train_ids + val_ids == ids

    def test_custom_fraction(self):
        ids = list(range(100))
        train_ids, val_ids = train_val_split(ids, val_fraction=0.2)
        assert len(train_ids) == 80
        assert len(val_ids) == 20


class TestDataLoader:
    def test_batch_shapes(self):
        ids = list(range(500))
        dl = get_dataloader(ids, block_size=16, batch_size=4, shuffle=False)
        batch = next(iter(dl))
        x, y = batch
        assert x.shape == (4, 16)
        assert y.shape == (4, 16)
