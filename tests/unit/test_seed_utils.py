import random

import numpy as np
import torch

from utils.seed_utils import get_seed_from_env, set_seed


def _draw():
    return random.random(), float(np.random.rand()), float(torch.rand(1))


def test_same_seed_gives_same_draws():
    set_seed(123)
    first = _draw()
    set_seed(123)
    assert _draw() == first


def test_different_seeds_give_different_draws():
    set_seed(1)
    first = _draw()
    set_seed(2)
    second = _draw()
    assert first[0] != second[0]
    assert first[1] != second[1]
    assert first[2] != second[2]


def test_get_seed_from_env_reads_value(monkeypatch):
    monkeypatch.setenv("SEED", "7")
    assert get_seed_from_env() == 7


def test_get_seed_from_env_defaults_to_42(monkeypatch):
    monkeypatch.delenv("SEED", raising=False)
    assert get_seed_from_env() == 42
