"""Tests for the accelerated NNUE inference C extension.

Verifies numerical equivalence with the pure-numpy IncrementalAccumulator.
"""

import numpy as np
import pytest

from src.model.accumulator import IncrementalAccumulator

try:
    from src.accel._nnue_accel import AcceleratedAccumulator
    HAS_ACCEL = True
except ImportError:
    HAS_ACCEL = False

pytestmark = pytest.mark.skipif(not HAS_ACCEL, reason="C extension not built")


def _random_weights(num_features=100, acc_size=256, l1_size=32, l2_size=32, seed=42):
    """Generate random weights matching NNUE architecture."""
    rng = np.random.RandomState(seed)
    return {
        "ft_weight": rng.randn(num_features, acc_size).astype(np.float64) * 0.01,
        "ft_bias": rng.randn(acc_size).astype(np.float64) * 0.01,
        "l1_weight": rng.randn(l1_size, acc_size * 2).astype(np.float64) * 0.1,
        "l1_bias": rng.randn(l1_size).astype(np.float64) * 0.01,
        "l2_weight": rng.randn(l2_size, l1_size).astype(np.float64) * 0.1,
        "l2_bias": rng.randn(l2_size).astype(np.float64) * 0.01,
        "out_weight": rng.randn(1, l2_size).astype(np.float64) * 0.1,
        "out_bias": rng.randn(1).astype(np.float64) * 0.01,
    }


def _make_both(weights):
    """Create both numpy and accelerated accumulators from the same weights."""
    w = weights
    np_acc = IncrementalAccumulator(
        ft_weight=w["ft_weight"], ft_bias=w["ft_bias"],
        l1_weight=w["l1_weight"], l1_bias=w["l1_bias"],
        l2_weight=w["l2_weight"], l2_bias=w["l2_bias"],
        out_weight=w["out_weight"], out_bias=w["out_bias"],
    )
    c_acc = AcceleratedAccumulator(
        w["ft_weight"], w["ft_bias"],
        w["l1_weight"], w["l1_bias"],
        w["l2_weight"], w["l2_bias"],
        w["out_weight"].flatten(), w["out_bias"].flatten(),
    )
    return np_acc, c_acc


def test_evaluate_equivalence():
    """AcceleratedAccumulator.evaluate() matches IncrementalAccumulator."""
    weights = _random_weights()
    np_acc, c_acc = _make_both(weights)

    wf = [0, 5, 10, 15, 20]
    bf = [1, 6, 11, 16, 21]

    np_acc.refresh(wf, bf)
    c_acc.refresh(wf, bf)

    for stm in [0, 1]:
        np_score = np_acc.evaluate(stm)
        c_score = c_acc.evaluate(stm)
        assert abs(np_score - c_score) < 1e-4, \
            f"stm={stm}: numpy={np_score}, accel={c_score}, diff={abs(np_score - c_score)}"


def test_evaluate_many_features():
    """Test with more features (closer to real chess positions)."""
    weights = _random_weights(num_features=10368, seed=123)
    np_acc, c_acc = _make_both(weights)

    rng = np.random.RandomState(99)
    wf = rng.randint(0, 10368, size=22).tolist()
    bf = rng.randint(0, 10368, size=22).tolist()

    np_acc.refresh(wf, bf)
    c_acc.refresh(wf, bf)

    for stm in [0, 1]:
        np_score = np_acc.evaluate(stm)
        c_score = c_acc.evaluate(stm)
        assert abs(np_score - c_score) < 1e-3, \
            f"stm={stm}: numpy={np_score}, accel={c_score}, diff={abs(np_score - c_score)}"


def test_push_pop():
    """Push/pop restores exact accumulator state."""
    weights = _random_weights()
    np_acc, c_acc = _make_both(weights)

    wf = [0, 1, 2, 3, 4]
    bf = [5, 6, 7, 8, 9]

    c_acc.refresh(wf, bf)
    score_before = c_acc.evaluate(0)

    c_acc.push()
    c_acc.update(0, [10], [0])  # Add 10, remove 0
    score_during = c_acc.evaluate(0)
    assert score_during != score_before  # Should change

    c_acc.pop()
    score_after = c_acc.evaluate(0)
    assert abs(score_after - score_before) < 1e-6  # Should restore exactly


def test_incremental_matches_full():
    """Incremental update produces same result as full refresh."""
    weights = _random_weights()
    np_acc, c_acc = _make_both(weights)

    # Full refresh with [0, 1, 2] / [3, 4, 5]
    c_acc.refresh([0, 1, 2], [3, 4, 5])
    c_acc.push()

    # Incremental: remove 2, add 6 -> [0, 1, 6]
    c_acc.update(0, [6], [2])
    score_incremental = c_acc.evaluate(0)
    c_acc.pop()

    # Full refresh with [0, 1, 6]
    c_acc.refresh([0, 1, 6], [3, 4, 5])
    score_full = c_acc.evaluate(0)

    assert abs(score_incremental - score_full) < 1e-5


def test_refresh_perspective():
    """refresh_perspective matches full refresh for that perspective."""
    weights = _random_weights()
    _, c_acc1 = _make_both(weights)
    _, c_acc2 = _make_both(weights)

    wf = [0, 5, 10]
    bf = [1, 6, 11]

    # c_acc1: full refresh both perspectives
    c_acc1.refresh(wf, bf)

    # c_acc2: refresh each perspective separately
    c_acc2.refresh_perspective(0, wf)
    c_acc2.refresh_perspective(1, bf)

    for stm in [0, 1]:
        s1 = c_acc1.evaluate(stm)
        s2 = c_acc2.evaluate(stm)
        assert abs(s1 - s2) < 1e-6, \
            f"stm={stm}: full={s1}, perspective={s2}"


def test_update_both_perspectives():
    """Test updating both white and black perspectives."""
    weights = _random_weights()
    np_acc, c_acc = _make_both(weights)

    wf = [0, 1, 2]
    bf = [3, 4, 5]

    np_acc.refresh(wf, bf)
    c_acc.refresh(wf, bf)

    # Update white: remove 2, add 10
    np_acc.update(0, [10], [2])
    c_acc.update(0, [10], [2])

    # Update black: remove 5, add 15
    np_acc.update(1, [15], [5])
    c_acc.update(1, [15], [5])

    for stm in [0, 1]:
        np_score = np_acc.evaluate(stm)
        c_score = c_acc.evaluate(stm)
        assert abs(np_score - c_score) < 1e-4


def test_empty_features():
    """Handles empty feature lists gracefully."""
    weights = _random_weights()
    _, c_acc = _make_both(weights)

    c_acc.refresh([], [])
    score = c_acc.evaluate(0)
    assert isinstance(score, float)


def test_single_feature():
    """Handles single-element feature lists."""
    weights = _random_weights()
    np_acc, c_acc = _make_both(weights)

    np_acc.refresh([0], [1])
    c_acc.refresh([0], [1])

    for stm in [0, 1]:
        np_score = np_acc.evaluate(stm)
        c_score = c_acc.evaluate(stm)
        assert abs(np_score - c_score) < 1e-4


def test_deep_stack():
    """Push/pop works correctly at depth 100+."""
    weights = _random_weights()
    _, c_acc = _make_both(weights)

    c_acc.refresh([0, 1, 2], [3, 4, 5])
    score_root = c_acc.evaluate(0)

    # Push 100 times with different updates
    for i in range(100):
        c_acc.push()
        feat = (i + 10) % weights["ft_weight"].shape[0]
        c_acc.update(0, [feat], [])

    # Pop all 100 back
    for _ in range(100):
        c_acc.pop()

    score_restored = c_acc.evaluate(0)
    assert abs(score_restored - score_root) < 1e-6


def test_stack_overflow():
    """Exceeding max stack depth raises RuntimeError."""
    weights = _random_weights()
    _, c_acc = _make_both(weights)

    c_acc.refresh([0], [1])

    with pytest.raises(RuntimeError, match="overflow"):
        for _ in range(200):
            c_acc.push()


def test_stack_underflow():
    """Popping empty stack raises RuntimeError."""
    weights = _random_weights()
    _, c_acc = _make_both(weights)

    with pytest.raises(RuntimeError, match="underflow"):
        c_acc.pop()


def test_multiple_seeds():
    """Test with different random weight seeds for robustness."""
    for seed in [0, 1, 42, 123, 999]:
        weights = _random_weights(seed=seed)
        np_acc, c_acc = _make_both(weights)

        rng = np.random.RandomState(seed + 1000)
        wf = rng.randint(0, 100, size=15).tolist()
        bf = rng.randint(0, 100, size=15).tolist()

        np_acc.refresh(wf, bf)
        c_acc.refresh(wf, bf)

        for stm in [0, 1]:
            np_score = np_acc.evaluate(stm)
            c_score = c_acc.evaluate(stm)
            assert abs(np_score - c_score) < 1e-3, \
                f"seed={seed}, stm={stm}: diff={abs(np_score - c_score)}"
