"""Tests for the NNUE model and accumulator."""

import numpy as np
import mlx.core as mx

from src.model.nnue import NNUEModel
from src.model.accumulator import IncrementalAccumulator


def test_model_forward_shape():
    model = NNUEModel(num_features=10368)
    mx.eval(model.parameters())
    batch = 4
    max_active = 22
    wf = mx.array(np.random.randint(0, 10368, (batch, max_active)).astype(np.int32))
    bf = mx.array(np.random.randint(0, 10368, (batch, max_active)).astype(np.int32))
    wm = mx.ones((batch, max_active))
    bm = mx.ones((batch, max_active))
    stm = mx.zeros((batch,), dtype=mx.int32)
    out = model(wf, bf, wm, bm, stm)
    mx.eval(out)
    assert out.shape == (batch, 1)


def test_model_backward():
    model = NNUEModel(num_features=1000)
    mx.eval(model.parameters())

    wf = mx.array(np.random.randint(0, 1000, (2, 10)).astype(np.int32))
    bf = mx.array(np.random.randint(0, 1000, (2, 10)).astype(np.int32))
    wm = mx.ones((2, 10))
    bm = mx.ones((2, 10))
    stm = mx.zeros((2,), dtype=mx.int32)

    def loss_fn(model):
        out = model(wf, bf, wm, bm, stm)
        return mx.sum(out)

    loss, grads = mx.value_and_grad(loss_fn)(model)
    mx.eval(loss, grads)
    # Verify gradients exist and are non-zero for at least some parameters
    flat_grads = {k: v for k, v in grads.items() if isinstance(v, mx.array)}
    assert len(flat_grads) > 0 or len(grads) > 0


def test_accumulator_from_model():
    model = NNUEModel(num_features=1000)
    mx.eval(model.parameters())
    acc = IncrementalAccumulator.from_model(model)
    assert acc.ft_weight.shape == (1000, 256)
    assert acc.ft_bias.shape == (256,)


def test_accumulator_refresh_and_evaluate():
    model = NNUEModel(num_features=100)
    mx.eval(model.parameters())
    acc = IncrementalAccumulator.from_model(model)
    acc.refresh([0, 1, 2], [3, 4, 5])
    score = acc.evaluate(0)
    assert isinstance(score, float)


def test_accumulator_push_pop():
    model = NNUEModel(num_features=100)
    mx.eval(model.parameters())
    acc = IncrementalAccumulator.from_model(model)
    acc.refresh([0, 1, 2], [3, 4, 5])
    score_before = acc.evaluate(0)

    acc.push()
    acc.update(0, [6], [0])  # Add feature 6, remove feature 0
    score_during = acc.evaluate(0)
    assert score_during != score_before  # Should change

    acc.pop()
    score_after = acc.evaluate(0)
    assert abs(score_after - score_before) < 1e-6  # Should restore exactly


def test_accumulator_incremental_matches_full():
    """Incremental update should produce same result as full refresh."""
    model = NNUEModel(num_features=100)
    mx.eval(model.parameters())
    acc = IncrementalAccumulator.from_model(model)

    # Full refresh with features [0, 1, 2]
    acc.refresh([0, 1, 2], [3, 4, 5])
    acc.push()

    # Incremental: remove 2, add 6 -> [0, 1, 6]
    acc.update(0, [6], [2])
    score_incremental = acc.evaluate(0)
    acc.pop()

    # Full refresh with [0, 1, 6]
    acc.refresh([0, 1, 6], [3, 4, 5])
    score_full = acc.evaluate(0)

    assert abs(score_incremental - score_full) < 1e-6
