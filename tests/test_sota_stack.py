"""Integration tests for the SOTA stack: output bucketing, WDL head,
feature factorization, full int8 quantization."""

import os
import tempfile

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from src.features.halfkp import chess_features, minichess_features
from src.features.halfkp_shogi import (
    minishogi_features,
    minishogi_features_onehot,
    shogi_features,
    shogi_features_onehot,
)
from src.games.chess.state import initial_state as chess_initial
from src.games.minichess.state import initial_state as minichess_initial
from src.games.shogi.state import initial_state as shogi_initial
from src.model.accumulator import IncrementalAccumulator
from src.model.nnue import NNUEModel
from src.search.evaluator import NNUEEvaluator
from src.training.factorize import (
    bake_virtual_into_main,
    build_factor_map,
    expand_batch_with_virtual,
    extend_num_features,
)
from src.training.loss import nnue_loss, nnue_loss_wdl
from src.training.trainer import Trainer

try:
    from src.accel import AcceleratedAccumulator
    HAS_ACCEL = True
except ImportError:
    HAS_ACCEL = False


def _weights_from_model(model):
    return {k: np.array(v) for k, v in tree_flatten(model.parameters())}


def _pack_features(wf, bf, max_active=32):
    wfa = np.zeros((1, max_active), dtype=np.int32)
    bfa = np.zeros((1, max_active), dtype=np.int32)
    wm = np.zeros((1, max_active), dtype=np.float32)
    bm = np.zeros((1, max_active), dtype=np.float32)
    wfa[0, :len(wf)] = wf
    wm[0, :len(wf)] = 1.0
    bfa[0, :len(bf)] = bf
    bm[0, :len(bf)] = 1.0
    return wfa, bfa, wm, bm


class TestBackwardCompat:

    def test_default_model_shape_unchanged(self):
        m = NNUEModel(40960)
        assert m.output.weight.shape == (1, 32)
        assert m.output.bias.shape == (1,)
        assert not hasattr(m, "wdl_output")

    def test_default_forward_shape(self):
        m = NNUEModel(1000, accumulator_size=32, l1_size=16, l2_size=8)
        mx.eval(m.parameters())
        wf = mx.zeros((2, 16), dtype=mx.int32)
        bf = mx.zeros((2, 16), dtype=mx.int32)
        wm = mx.ones((2, 16))
        bm = mx.ones((2, 16))
        out = m(wf, bf, wm, bm, mx.array([0, 1]))
        assert out.shape == (2, 1)

    def test_legacy_accumulator_out_weight_shape(self):
        rng = np.random.default_rng(0)
        acc = IncrementalAccumulator(
            ft_weight=rng.normal(0, 0.01, (1000, 32)).astype(np.float32),
            ft_bias=np.zeros(32, dtype=np.float32),
            l1_weight=rng.normal(0, 0.01, (16, 64)).astype(np.float32),
            l1_bias=np.zeros(16, dtype=np.float32),
            l2_weight=rng.normal(0, 0.01, (8, 16)).astype(np.float32),
            l2_bias=np.zeros(8, dtype=np.float32),
            out_weight=rng.normal(0, 0.01, (1, 8)).astype(np.float32),
            out_bias=np.zeros(1, dtype=np.float32),
        )
        assert acc.num_buckets == 1
        acc.refresh([], [])
        v0 = acc.evaluate(0)
        v1 = acc.evaluate(0, bucket_idx=0)
        assert v0 == v1


class TestOutputBucketing:

    def test_model_forward_gathers_bucket(self):
        mx.random.seed(0)
        m = NNUEModel(1000, accumulator_size=32, l1_size=16, l2_size=8,
                      num_output_buckets=4)
        mx.eval(m.parameters())
        wf = mx.array([[1, 2, 0], [3, 4, 0]], dtype=mx.int32)
        bf = mx.array([[5, 6, 0], [7, 8, 0]], dtype=mx.int32)
        wm = mx.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        bm = mx.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        stm = mx.array([0, 1])

        for b0 in range(4):
            for b1 in range(4):
                out = m(wf, bf, wm, bm, stm,
                        bucket_idx=mx.array([b0, b1], dtype=mx.int32))
                assert out.shape == (2, 1)

    def test_mlx_numpy_c_consistency(self):
        mx.random.seed(42)
        fs = chess_features()
        model = NNUEModel(fs.num_features(), accumulator_size=64, l1_size=32,
                          l2_size=16, num_output_buckets=4)
        mx.eval(model.parameters())
        state = chess_initial()
        wf_idx = fs.active_features(state, 0)
        bf_idx = fs.active_features(state, 1)
        wf, bf, wm, bm = _pack_features(wf_idx, bf_idx)

        weights = _weights_from_model(model)
        np_acc = IncrementalAccumulator(
            ft_weight=weights["feature_table.weight"],
            ft_bias=weights["ft_bias"],
            l1_weight=weights["l1.weight"],
            l1_bias=weights["l1.bias"],
            l2_weight=weights["l2.weight"],
            l2_bias=weights["l2.bias"],
            out_weight=weights["output.weight"],
            out_bias=weights["output.bias"],
        )
        np_acc.refresh(wf_idx, bf_idx)

        for b in range(4):
            out_mlx = model(mx.array(wf), mx.array(bf), mx.array(wm),
                            mx.array(bm), mx.array([0], dtype=mx.int32),
                            bucket_idx=mx.array([b], dtype=mx.int32))
            mx.eval(out_mlx)
            mlx_val = float(out_mlx[0, 0])
            np_val = np_acc.evaluate(state.side_to_move(), bucket_idx=b)
            assert abs(mlx_val - np_val) < 1e-5, (
                f"MLX vs numpy mismatch at bucket {b}: {mlx_val} vs {np_val}"
            )

    @pytest.mark.skipif(not HAS_ACCEL, reason="C accelerator not built")
    def test_c_path_bucket_consistency_with_numpy(self):
        mx.random.seed(123)
        fs = chess_features()
        model = NNUEModel(fs.num_features(), accumulator_size=64, l1_size=32,
                          l2_size=16, num_output_buckets=4)
        mx.eval(model.parameters())
        weights = _weights_from_model(model)
        state = chess_initial()
        wf_idx = fs.active_features(state, 0)
        bf_idx = fs.active_features(state, 1)

        np_acc = IncrementalAccumulator(
            ft_weight=weights["feature_table.weight"], ft_bias=weights["ft_bias"],
            l1_weight=weights["l1.weight"], l1_bias=weights["l1.bias"],
            l2_weight=weights["l2.weight"], l2_bias=weights["l2.bias"],
            out_weight=weights["output.weight"], out_bias=weights["output.bias"],
        )
        c_acc = AcceleratedAccumulator(
            ft_weight=weights["feature_table.weight"], ft_bias=weights["ft_bias"],
            l1_weight=weights["l1.weight"], l1_bias=weights["l1.bias"],
            l2_weight=weights["l2.weight"], l2_bias=weights["l2.bias"],
            out_weight=weights["output.weight"], out_bias=weights["output.bias"],
        )
        np_acc.refresh(wf_idx, bf_idx)
        c_acc.refresh(wf_idx, bf_idx)

        for b in range(4):
            np_val = np_acc.evaluate(state.side_to_move(), bucket_idx=b)
            c_val = c_acc.evaluate(state.side_to_move(), b)
            assert abs(np_val - c_val) < 1e-5


class TestWDLHead:

    def test_wdl_model_returns_tuple(self):
        mx.random.seed(0)
        m = NNUEModel(1000, accumulator_size=32, l1_size=16, l2_size=8,
                      use_wdl_head=True)
        mx.eval(m.parameters())
        out = m(mx.zeros((2, 8), dtype=mx.int32),
                mx.zeros((2, 8), dtype=mx.int32),
                mx.ones((2, 8)), mx.ones((2, 8)),
                mx.array([0, 1]))
        assert isinstance(out, tuple)
        assert len(out) == 2
        assert out[0].shape == (2, 1)
        assert out[1].shape == (2, 1)

    def test_wdl_loss_non_negative(self):
        mx.random.seed(0)
        s = mx.array([[0.1], [-0.2]])
        w = mx.array([[0.5], [-0.3]])
        score = mx.array([100.0, -50.0])
        result = mx.array([1.0, 0.0])
        loss = nnue_loss_wdl(s, w, score, result, lambda_=1.0, wdl_weight=0.5)
        mx.eval(loss)
        assert float(loss) >= 0.0


class TestFeatureFactorization:

    def test_chess_factor_map(self):
        fs = chess_features()
        factor_map, num_virtual = build_factor_map(fs)
        assert num_virtual == 640
        assert factor_map.shape == (40960,)
        assert factor_map.max() == 639
        assert factor_map.min() == 0

    def test_minichess_factor_map(self):
        fs = minichess_features()
        factor_map, num_virtual = build_factor_map(fs)
        assert num_virtual == 288
        assert factor_map.shape == (10368,)

    def test_bake_virtual_into_main_roundtrip(self):
        fs = chess_features()
        factor_map, num_virtual = build_factor_map(fs)
        num_main = fs.num_features()
        rng = np.random.default_rng(0)
        ft_big = rng.normal(0, 0.01, (num_main + num_virtual, 64)).astype(np.float32)
        baked = bake_virtual_into_main(ft_big, factor_map, num_main)
        assert baked.shape == (num_main, 64)
        expected = ft_big[0] + ft_big[num_main + factor_map[0]]
        assert np.allclose(baked[0], expected, atol=1e-6)

    def test_expand_batch_appends_virtual_indices(self):
        fs = minichess_features()
        factor_map_np, num_virtual = build_factor_map(fs)
        num_main = fs.num_features()
        fmap_mx = mx.array(factor_map_np)

        wf = mx.array([[1, 10, 0]], dtype=mx.int32)
        bf = mx.array([[5, 20, 0]], dtype=mx.int32)
        wm = mx.array([[1.0, 1.0, 0.0]])
        bm = mx.array([[1.0, 1.0, 0.0]])
        batch = {
            "white_features": wf, "black_features": bf,
            "white_mask": wm, "black_mask": bm,
        }
        expanded = expand_batch_with_virtual(batch, fmap_mx, num_main)
        mx.eval(expanded["white_features"])
        wf_exp = np.array(expanded["white_features"])
        assert wf_exp.shape == (1, 6)
        assert wf_exp[0, 0] == 1
        assert wf_exp[0, 3] == num_main + (1 % 288)
        assert wf_exp[0, 4] == num_main + (10 % 288)


class TestShogiOneHot:

    def test_shogi_onehot_preserves_feature_count(self):
        assert shogi_features().num_features() == shogi_features_onehot().num_features()
        assert minishogi_features().num_features() == minishogi_features_onehot().num_features()

    def test_shogi_onehot_active_features_at_start(self):
        fs = shogi_features_onehot()
        state = shogi_initial()
        feats = fs.active_features(state, 0)
        assert len(feats) == 38  # 38 non-king pieces on the board at start
        assert len(set(feats)) == len(feats)  # no duplicates

    def test_shogi_onehot_mirror_table_is_none(self):
        assert shogi_features_onehot().mirror_table() is None

    def test_shogi_ordinal_mirror_table_shape(self):
        mt = shogi_features().mirror_table()
        assert mt.shape == (shogi_features().num_features(),)
        assert mt.dtype == np.int32


class TestMaterialBucket:

    def test_chess_start_position_bucket(self):
        fs = chess_features()
        state = chess_initial()
        for n_buckets in [4, 8, 16]:
            b = fs.material_bucket(state, n_buckets)
            assert 0 <= b < n_buckets
            assert b == n_buckets - 1  # 30 non-king pieces = max bucket

    def test_shogi_start_position_bucket(self):
        fs = shogi_features()
        state = shogi_initial()
        b = fs.material_bucket(state, 8)
        assert 0 <= b < 8

    def test_feature_count_bucket_monotone(self):
        fs = chess_features()
        prev = -1
        for num_features in range(0, 31, 3):
            b = fs.bucket_from_feature_counts(num_features, num_features, 8)
            assert b >= prev
            prev = b


class TestQuantizationRoundtrip:

    def _make_small_model(self, num_buckets=1, wdl=False):
        mx.random.seed(1)
        m = NNUEModel(1000, accumulator_size=32, l1_size=16, l2_size=8,
                      num_output_buckets=num_buckets, use_wdl_head=wdl)
        mx.eval(m.parameters())
        return m

    def test_int16_ft_roundtrip(self):
        from scripts.quantize import quantize_model
        m = self._make_small_model(num_buckets=4)
        weights = _weights_from_model(m)
        weights["num_output_buckets"] = np.array(4, dtype=np.int32)
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "m.npz")
            dst = os.path.join(td, "mq.npz")
            np.savez(src, **weights)
            quantize_model(src, dst)
            loaded = np.load(dst)
            assert loaded["feature_table.weight"].dtype == np.int16
            assert loaded["l1.weight"].dtype == np.float32
            assert loaded["output.weight"].shape == (4, 8)
            assert int(loaded["num_output_buckets"]) == 4

    def test_full_int8_roundtrip(self):
        from scripts.quantize import quantize_model
        m = self._make_small_model(num_buckets=2)
        weights = _weights_from_model(m)
        weights["num_output_buckets"] = np.array(2, dtype=np.int32)
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "m.npz")
            dst = os.path.join(td, "mq.npz")
            np.savez(src, **weights)
            quantize_model(src, dst, full_int8=True)
            loaded = np.load(dst)
            assert loaded["l1.weight"].dtype == np.int8
            assert loaded["l2.weight"].dtype == np.int8
            assert loaded["output.weight"].dtype == np.int8
            assert "l1_scale" in loaded.files
            assert "l2_scale" in loaded.files
            assert "output_scale" in loaded.files


class TestEvaluatorMetadata:

    def test_loads_output_eval_scale_from_metadata(self):
        fs = chess_features()
        m = NNUEModel(fs.num_features(), accumulator_size=32, l1_size=16, l2_size=8)
        mx.eval(m.parameters())
        weights = _weights_from_model(m)
        weights["output_eval_scale"] = np.array(96.0, dtype=np.float32)
        weights["num_output_buckets"] = np.array(1, dtype=np.int32)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "m.npz")
            np.savez(path, **weights)
            ev = NNUEEvaluator.from_numpy(path, fs)
            assert abs(ev.output_eval_scale - 96.0) < 1e-6
            assert ev.num_output_buckets == 1

    def test_legacy_model_default_metadata(self):
        fs = chess_features()
        m = NNUEModel(fs.num_features(), accumulator_size=32, l1_size=16, l2_size=8)
        mx.eval(m.parameters())
        weights = _weights_from_model(m)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "m.npz")
            np.savez(path, **weights)
            ev = NNUEEvaluator.from_numpy(path, fs)
            # Default must equal the class constant (OUTPUT_SCALE=32), which
            # matches the inverse of the training loss scaling. Older code
            # used 128 here, inflating every NNUE score 4x and wrecking
            # adjudication thresholds during self-play and evaluation.
            assert abs(ev.output_eval_scale - 32.0) < 1e-6
            assert ev.num_output_buckets == 1


class TestTrainerVariants:

    def _mini_config(self, **kwargs):
        fs = minichess_features()
        base = dict(
            num_features=fs.num_features(),
            accumulator_size=32, l1_size=16, l2_size=8,
            feature_set=fs, batch_size=128, total_epochs=2,
        )
        base.update(kwargs)
        return base

    def test_construct_baseline(self):
        t = Trainer(**self._mini_config())
        assert t.model.output.weight.shape == (1, 8)

    def test_construct_multi_bucket(self):
        t = Trainer(**self._mini_config(num_output_buckets=4))
        assert t.model.output.weight.shape == (4, 8)

    def test_construct_factorized(self):
        t = Trainer(**self._mini_config(factorize=True))
        assert t.num_main_features == minichess_features().num_features()
        assert t.model.num_features > t.num_main_features

    def test_construct_wdl(self):
        t = Trainer(**self._mini_config(use_wdl_head=True))
        assert hasattr(t.model, "wdl_output")

    def test_construct_full_stack(self):
        t = Trainer(**self._mini_config(
            num_output_buckets=4, use_wdl_head=True, factorize=True))
        assert t.model.output.weight.shape == (4, 8)
        assert hasattr(t.model, "wdl_output")
