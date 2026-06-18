"""
tests.py — pytest test suite for the ProNDF package.

Run with:
    pytest tests.py -v
"""
import sys
import warnings
sys.path.insert(0, "src")

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_meta(dsource=2, dnum=3, dtargets=1, qual_in=False, quant_in=True, dcat=None):
    return {
        "dsource": dsource,
        "dcat": dcat if dcat is not None else [],
        "dnum": dnum,
        "dtargets": dtargets,
        "qual_in": qual_in,
        "quant_in": quant_in,
        "num_samples": [10] * dsource,
    }


def _make_batch(dsource=2, dnum=3, dtargets=1, n=8, qual_in=False, quant_in=True, dcat_sum=0):
    source = torch.zeros(n, dsource)
    source[:n // 2, 0] = 1
    source[n // 2:, 1] = 1
    cat = torch.zeros(n, dcat_sum) if qual_in else torch.zeros(n, 0)
    num = torch.randn(n, dnum) if quant_in else torch.zeros(n, 0)
    targets = torch.randn(n, dtargets)
    return {"source": source, "cat": cat, "num": num, "targets": targets}


# ---------------------------------------------------------------------------
# act_fns
# ---------------------------------------------------------------------------

class TestActFns:
    def test_registry_populated(self):
        from prondf.act_fns import ACT_FN_REGISTRY
        for name in ("Identity", "Tanh", "ReLU", "Sigmoid", "LeakyReLU", "Softmax"):
            assert name in ACT_FN_REGISTRY

    def test_custom_registration(self):
        from torch import nn
        from prondf.act_fns import ACT_FN_REGISTRY, register_act_fn

        @register_act_fn("TestActFn")
        class TestActFn(nn.Identity):
            pass

        assert "TestActFn" in ACT_FN_REGISTRY
        del ACT_FN_REGISTRY["TestActFn"]


# ---------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------

class TestDetBlock:
    def test_forward_shape(self):
        from prondf.blocks import Det_Block
        block = Det_Block(d_in=4, d_out=2, hidden_layers=[8, 8])
        x = torch.randn(10, 4)
        out = block(x)
        assert out.shape == (10, 2)

    def test_probabilistic_output_flag(self):
        from prondf.blocks import Det_Block
        block = Det_Block(d_in=4, d_out=2, hidden_layers=[8])
        assert block.probabilistic_output is False

    def test_predict_distribution_not_implemented(self):
        from prondf.blocks import Det_Block
        block = Det_Block(d_in=4, d_out=2, hidden_layers=[8])
        with pytest.raises(NotImplementedError):
            block.predict_distribution(torch.randn(5, 4))

    def test_xavier_init(self):
        from prondf.blocks import Det_Block
        block = Det_Block(d_in=4, d_out=1, hidden_layers=[8])
        # Biases should be zero-initialized
        for layer in block.architecture:
            if isinstance(layer, torch.nn.Linear):
                assert torch.all(layer.bias == 0)

    def test_registry(self):
        from prondf.blocks import BLOCK_REGISTRY
        assert "Det_Block" in BLOCK_REGISTRY

    def test_custom_act_fn(self):
        from prondf.blocks import Det_Block
        block = Det_Block(d_in=3, d_out=1, hidden_layers=[4], hidden_act_fn="ReLU", output_act_fn="Sigmoid")
        out = block(torch.randn(5, 3))
        assert out.shape == (5, 1)
        assert (out >= 0).all() and (out <= 1).all()

    def test_unknown_act_fn_raises(self):
        from prondf.blocks import Det_Block
        with pytest.raises(KeyError):
            Det_Block(d_in=3, d_out=1, hidden_layers=[4], hidden_act_fn="Nonsense")


class TestProbBlock:
    def test_forward_shape(self):
        from prondf.blocks import Prob_Block
        block = Prob_Block(d_in=4, d_out=2, hidden_layers=[8])
        x = torch.randn(10, 4)
        out = block(x)
        assert out.shape == (10, 2)

    def test_probabilistic_output_flag(self):
        from prondf.blocks import Prob_Block
        block = Prob_Block(d_in=4, d_out=2, hidden_layers=[8])
        assert block.probabilistic_output is True

    def test_distribution_params_shape(self):
        from prondf.blocks import Prob_Block
        block = Prob_Block(d_in=4, d_out=3, hidden_layers=[8])
        x = torch.randn(5, 4)
        mu, log_var = block.distribution_params(x)
        assert mu.shape == (5, 3)
        assert log_var.shape == (5, 3)

    def test_predict_distribution_returns_normal(self):
        from prondf.blocks import Prob_Block
        block = Prob_Block(d_in=4, d_out=2, hidden_layers=[8])
        dist = block.predict_distribution(torch.randn(6, 4))
        assert isinstance(dist, torch.distributions.Normal)
        assert dist.mean.shape == (6, 2)

    def test_registry(self):
        from prondf.blocks import BLOCK_REGISTRY
        assert "Prob_Block" in BLOCK_REGISTRY

    def test_base_block_not_instantiable(self):
        from prondf.blocks import Base_Block
        with pytest.raises(NotImplementedError):
            Base_Block()


# ---------------------------------------------------------------------------
# losses — loss functions
# ---------------------------------------------------------------------------

class TestLossFunctions:
    def setup_method(self):
        n = 16
        self.mu = torch.randn(n, 1)
        self.var = torch.exp(torch.randn(n, 1))
        self.sigma = self.var.sqrt()
        self.targets = torch.randn(n, 1)

    def test_nll_loss_scalar(self):
        from prondf.losses import NLL_loss
        loss = NLL_loss(self.mu, self.var, self.targets)
        assert loss.ndim == 0

    def test_nll_loss_positive(self):
        from prondf.losses import NLL_loss
        loss = NLL_loss(self.mu, self.var, self.targets)
        assert loss.item() > 0

    def test_is_loss_scalar(self):
        from prondf.losses import IS_loss
        loss = IS_loss(self.mu, self.sigma, self.targets)
        assert loss.ndim == 0

    def test_is_loss_nonnegative(self):
        from prondf.losses import IS_loss
        loss = IS_loss(self.mu, self.sigma, self.targets, alpha=0.05)
        assert loss.item() >= 0

    def test_kl_div_var_only_scalar(self):
        from prondf.losses import KL_div_var_only_loss
        loss = KL_div_var_only_loss(self.var, self.targets)
        assert loss.ndim == 0


class TestLossClasses:
    def _make_context(self, probabilistic=True, n=8, dsource=2, dnum=2, dtargets=1):
        from prondf.losses import Loss_Context
        from prondf.models import Build_ProNDF
        meta = _simple_meta(dsource=dsource, dnum=dnum, dtargets=dtargets)
        model = Build_ProNDF(dataset_meta=meta, loggers=[])
        batch = _make_batch(dsource=dsource, dnum=dnum, dtargets=dtargets, n=n)
        outputs = model.get_model_outputs(batch)
        return Loss_Context(model, batch, outputs)

    def test_output_mse_loss(self):
        from prondf.losses import Output_MSE_loss
        ctx = self._make_context()
        loss = Output_MSE_loss()(ctx)
        assert loss.ndim == 0

    def test_output_nll_loss(self):
        from prondf.losses import Output_NLL_loss
        ctx = self._make_context(probabilistic=True)
        loss = Output_NLL_loss()(ctx)
        assert loss.ndim == 0

    def test_output_is_loss(self):
        from prondf.losses import Output_IS_loss
        ctx = self._make_context(probabilistic=True)
        loss = Output_IS_loss(alpha=0.05, strength=1.0)(ctx)
        assert loss.ndim == 0

    def test_empty_batch_returns_zero(self):
        from prondf.losses import Output_NLL_loss, Loss_Context
        from prondf.models import Build_ProNDF
        meta = _simple_meta(dsource=2, dnum=2, dtargets=1)
        model = Build_ProNDF(dataset_meta=meta, loggers=[])
        # Make a batch with zero samples (must use same dnum as model)
        empty_batch = {k: v[:0] for k, v in _make_batch(dsource=2, dnum=2, dtargets=1).items()}
        outputs = model.get_model_outputs(empty_batch)
        ctx = Loss_Context(model, empty_batch, outputs)
        loss = Output_NLL_loss()(ctx)
        assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# losses — data splits
# ---------------------------------------------------------------------------

class TestDataSplits:
    def _ctx(self, dsource=3, n=12):
        from prondf.losses import Loss_Context
        from prondf.models import Build_ProNDF
        meta = _simple_meta(dsource=dsource, dnum=2, dtargets=1)
        model = Build_ProNDF(dataset_meta=meta, loggers=[])
        batch = _make_batch(dsource=dsource, dnum=2, dtargets=1, n=n)
        # Assign each sample to a source round-robin
        batch["source"] = torch.zeros(n, dsource)
        for i in range(n):
            batch["source"][i, i % dsource] = 1
        outputs = model.get_model_outputs(batch)
        return Loss_Context(model, batch, outputs)

    def test_no_split_returns_one_context(self):
        from prondf.losses import No_Split
        ctx = self._ctx()
        splits = No_Split()(ctx)
        assert len(splits) == 1

    def test_split_by_source_count(self):
        from prondf.losses import Split_by_Source
        dsource = 3
        ctx = self._ctx(dsource=dsource)
        splits = Split_by_Source({"num_sources": dsource})(ctx)
        assert len(splits) == dsource

    def test_split_by_source_targets_disjoint(self):
        from prondf.losses import Split_by_Source
        dsource = 3
        n = 12
        ctx = self._ctx(dsource=dsource, n=n)
        splits = Split_by_Source({"num_sources": dsource})(ctx)
        total = sum(s.batch["targets"].shape[0] for s in splits)
        assert total == n

    def test_split_by_output_dim_count(self):
        from prondf.losses import Loss_Context, Split_by_Output_Dim
        from prondf.models import Build_ProNDF
        dtargets = 3
        meta = _simple_meta(dsource=2, dnum=2, dtargets=dtargets)
        model = Build_ProNDF(dataset_meta=meta, loggers=[])
        batch = _make_batch(dsource=2, dnum=2, dtargets=dtargets, n=8)
        outputs = model.get_model_outputs(batch)
        ctx = Loss_Context(model, batch, outputs)
        splits = Split_by_Output_Dim({"num_outputs": dtargets})(ctx)
        assert len(splits) == dtargets
        for s in splits:
            assert s.batch["targets"].shape[1] == 1


# ---------------------------------------------------------------------------
# losses — weighting algorithms
# ---------------------------------------------------------------------------

class TestWeightingAlgorithms:
    def test_no_weighting_sums(self):
        from prondf.losses import No_Weighting
        alg = No_Weighting()
        terms = torch.tensor([1.0, 2.0, 3.0])
        assert alg(terms).item() == pytest.approx(6.0)

    def test_fixed_weights_default_uniform(self):
        from prondf.losses import Fixed_Weights
        alg = Fixed_Weights(num_loss_terms=3)
        terms = torch.tensor([1.0, 2.0, 3.0])
        assert alg(terms).item() == pytest.approx(6.0)

    def test_fixed_weights_custom(self):
        from prondf.losses import Fixed_Weights
        alg = Fixed_Weights(num_loss_terms=3, weights=[1.0, 0.0, 0.0])
        terms = torch.tensor([5.0, 99.0, 99.0])
        assert alg(terms).item() == pytest.approx(5.0)

    def test_two_moment_weighting_forward_before_update(self):
        from prondf.losses import Two_Moment_Weighting
        alg = Two_Moment_Weighting(num_loss_terms=2)
        terms = torch.tensor([1.0, 2.0])
        # Before any update, weights are zero → loss is 0
        result = alg(terms)
        assert result.item() == pytest.approx(0.0)

    def test_two_moment_weighting_ema_uses_own_state(self):
        """Regression: the EMA must accumulate each term's own previous moment
        (self.lambdas[idx]), not the never-updated reference slot
        (self.lambdas[ref_idx]). With the old bug the seeded per-term state was
        ignored, so lambdas[idx] would collapse toward ~0 after an update."""
        from prondf.losses import Two_Moment_Weighting, Loss_Context

        # Tiny model so update() can take gradients w.r.t. real parameters.
        model = torch.nn.Linear(2, 1)
        x = torch.randn(4, 2)
        out = model(x)
        # Two loss terms that both depend on the model parameters.
        loss_terms = torch.stack([(out ** 2).mean(), (out + 1.0).pow(2).mean()])
        ctx = Loss_Context(model, batch={}, outputs={})

        # alpha small → previous (seeded) value dominates the moving average.
        alg = Two_Moment_Weighting(num_loss_terms=2, ref_idx=0, alpha1=0.01, alpha2=0.01)
        # Seed reference slot at 0 and the non-reference term at a large value.
        with torch.no_grad():
            alg.lambdas.copy_(torch.tensor([0.0, 100.0]))
            alg.gammas.copy_(torch.tensor([0.0, 100.0]))

        alg.update(loss_terms, ctx)

        # New (correct) behavior: ~0.99 * 100 + tiny ≈ 99, so clearly tracks the
        # term's own seeded value. Old (buggy) behavior would give ~0.99 * 0 ≈ 0.
        assert alg.lambdas[1].item() > 50.0
        assert alg.gammas[1].item() > 50.0


# ---------------------------------------------------------------------------
# losses — loss handlers
# ---------------------------------------------------------------------------

class TestLossHandlers:
    def _build_one_stage(self, dsource=2, dnum=2, dtargets=1, n=8):
        from prondf.losses import One_Stage_Loss_Handler, Loss_Context
        from prondf.models import Build_ProNDF
        meta = _simple_meta(dsource=dsource, dnum=dnum, dtargets=dtargets)
        model = Build_ProNDF(
            dataset_meta=meta,
            loss_weighting=False,
            loggers=[],
        )
        batch = _make_batch(dsource=dsource, dnum=dnum, dtargets=dtargets, n=n)
        return model, batch

    def _build_hierarchical(self, dsource=2, dnum=2, dtargets=1, n=8):
        from prondf.models import Build_ProNDF
        meta = _simple_meta(dsource=dsource, dnum=dnum, dtargets=dtargets)
        model = Build_ProNDF(
            dataset_meta=meta,
            loss_weighting=True,
            loggers=[],
        )
        batch = _make_batch(dsource=dsource, dnum=dnum, dtargets=dtargets, n=n)
        return model, batch

    def test_one_stage_loss_scalar(self):
        model, batch = self._build_one_stage()
        outputs = model.get_model_outputs(batch)
        model.loss_handler.build_loss_context(model, batch, outputs)
        model.loss_handler.compute_loss_terms()
        model.loss_handler.update_loss_weights()
        loss = model.loss_handler.compute_loss()
        assert loss.ndim == 0

    def test_hierarchical_loss_scalar(self):
        model, batch = self._build_hierarchical()
        outputs = model.get_model_outputs(batch)
        model.loss_handler.build_loss_context(model, batch, outputs)
        model.loss_handler.compute_loss_terms()
        model.loss_handler.update_loss_weights()
        loss = model.loss_handler.compute_loss()
        assert loss.ndim == 0

    def test_hierarchical_weighting_alg_in_state_dict(self):
        """Regression: weighting alg buffers must be in state_dict (nn.ModuleList fix)."""
        model, _ = self._build_hierarchical()
        sd = model.loss_handler.state_dict()
        assert any("weights" in k for k in sd.keys()), (
            "Weighting algorithm buffers not found in state_dict"
        )

    def test_hierarchical_loss_terms_shape(self):
        model, batch = self._build_hierarchical(dsource=2, dtargets=2)
        outputs = model.get_model_outputs(batch)
        model.loss_handler.build_loss_context(model, batch, outputs)
        model.loss_handler.compute_loss_terms()
        # 2D: (num_outer_splits, num_inner_splits * num_loss_fns)
        assert model.loss_handler.loss_terms.ndim == 2

    @pytest.mark.parametrize("dsource,dtargets", [(2, 3), (3, 2)])
    def test_hierarchical_loss_unequal_dims(self, dsource, dtargets):
        """Regression: compute_loss must collapse the inner (source) axis, not the
        outer (output) axis. With the old sum(dim=0) the inner-weighted terms had
        shape (dsource,) and were multiplied by the outer (dtargets,) weights,
        which raises a broadcast error whenever dsource != dtargets (both > 1)."""
        model, batch = self._build_hierarchical(dsource=dsource, dtargets=dtargets, n=12)
        outputs = model.get_model_outputs(batch)
        model.loss_handler.build_loss_context(model, batch, outputs)
        model.loss_handler.compute_loss_terms()
        # loss_terms is (num_outer=dtargets, num_inner=dsource)
        assert model.loss_handler.loss_terms.shape == (dtargets, dsource)
        model.loss_handler.update_loss_weights()
        loss = model.loss_handler.compute_loss()
        assert loss.ndim == 0
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# data — MultiFidelityDataset
# ---------------------------------------------------------------------------

class TestMultiFidelityDataset:
    def _make_ds(self, n=20, dsource=2, dnum=3, dcat_levels=None, dtargets=1):
        from prondf.data import MultiFidelityDataset
        qual_in = dcat_levels is not None
        quant_in = dnum > 0
        source = np.eye(dsource)[np.random.randint(0, dsource, n)]
        cat = None
        if qual_in:
            dcat = dcat_levels
            cat = np.random.randint(0, dcat[0], (n, len(dcat))).astype(float)
        num = np.random.randn(n, dnum) if quant_in else None
        targets = np.random.randn(n, dtargets)
        meta = {
            "dsource": dsource, "dcat": dcat_levels or [], "dnum": dnum,
            "dtargets": dtargets, "qual_in": qual_in, "quant_in": quant_in,
            "num_samples": [n // dsource] * dsource,
        }
        return MultiFidelityDataset(source=source, cat=cat, num=num, targets=targets, meta=meta)

    def test_len(self):
        ds = self._make_ds(n=20)
        assert len(ds) == 20

    def test_getitem_keys(self):
        ds = self._make_ds(n=10)
        sample = ds[0]
        assert set(sample.keys()) == {"source", "cat", "num", "targets"}

    def test_getitem_types_are_tensors(self):
        ds = self._make_ds(n=10)
        sample = ds[0]
        for v in sample.values():
            assert isinstance(v, torch.Tensor)

    def test_empty_cat_when_qual_false(self):
        ds = self._make_ds(n=10, dnum=3)
        sample = ds[0]
        assert sample["cat"].shape == (0,)

    def test_empty_num_when_quant_false(self):
        from prondf.data import MultiFidelityDataset
        ds = self._make_ds(n=10, dnum=0, dcat_levels=[4])
        sample = ds[0]
        assert sample["num"].shape == (0,)

    def test_source_shape(self):
        ds = self._make_ds(n=10, dsource=3)
        assert ds[0]["source"].shape == (3,)

    def test_save_load_roundtrip(self, tmp_path):
        ds = self._make_ds(n=20, dsource=2, dnum=3, dtargets=2)
        ds.save(tmp_path, "test_ds")
        from prondf.data import MultiFidelityDataset
        loaded = MultiFidelityDataset.load(tmp_path, "test_ds")
        assert len(loaded) == len(ds)
        np.testing.assert_array_almost_equal(loaded.targets, ds.targets)
        np.testing.assert_array_almost_equal(loaded.num, ds.num)

    def test_load_missing_file_raises(self, tmp_path):
        from prondf.data import MultiFidelityDataset
        with pytest.raises(FileNotFoundError):
            MultiFidelityDataset.load(tmp_path, "nonexistent")


# ---------------------------------------------------------------------------
# data — split_dataset and save/load_splits
# ---------------------------------------------------------------------------

class TestSplitDataset:
    def _make_ds(self, n=100, dsource=3):
        from prondf.data import MultiFidelityDataset
        source = np.eye(dsource)[np.tile(np.arange(dsource), n // dsource + 1)[:n]]
        num = np.random.randn(n, 2)
        targets = np.random.randn(n, 1)
        meta = {"dsource": dsource, "dcat": [], "dnum": 2, "dtargets": 1,
                "qual_in": False, "quant_in": True, "num_samples": [n // dsource] * dsource}
        return MultiFidelityDataset(source=source, num=num, targets=targets, meta=meta)

    def test_split_sizes_sum(self):
        from prondf.data import split_dataset
        ds = self._make_ds(n=100)
        train, val, test = split_dataset(ds, [0.7, 0.2, 0.1])
        assert len(train) + len(val) + len(test) == len(ds)

    def test_split_ratios_must_sum_to_one(self):
        from prondf.data import split_dataset
        ds = self._make_ds()
        with pytest.raises(ValueError):
            split_dataset(ds, [0.6, 0.2, 0.1])

    def test_split_ratios_must_be_length_3(self):
        from prondf.data import split_dataset
        ds = self._make_ds()
        with pytest.raises(ValueError):
            split_dataset(ds, [0.8, 0.2])

    def test_split_reproducible_with_seeded_generator(self):
        """A seeded generator must produce identical splits across calls."""
        from prondf.data import split_dataset
        ds = self._make_ds(n=99)
        tr1, va1, te1 = split_dataset(ds, [0.7, 0.2, 0.1], random_generator=np.random.default_rng(0))
        tr2, va2, te2 = split_dataset(ds, [0.7, 0.2, 0.1], random_generator=np.random.default_rng(0))
        for a, b in ((tr1, tr2), (va1, va2), (te1, te2)):
            np.testing.assert_array_equal(a.source, b.source)
            np.testing.assert_array_equal(a.targets, b.targets)

    def test_save_load_splits_roundtrip(self, tmp_path):
        from prondf.data import split_dataset, save_splits, load_splits
        ds = self._make_ds(n=60)
        train, val, test = split_dataset(ds, [0.7, 0.2, 0.1])
        save_splits(train, val, test, tmp_path, "ds")
        tr2, va2, te2 = load_splits(tmp_path, "ds")
        assert len(tr2) == len(train)
        assert len(va2) == len(val)
        assert len(te2) == len(test)


# ---------------------------------------------------------------------------
# data — collate_fn
# ---------------------------------------------------------------------------

class TestCollateFn:
    def test_batch_shapes(self):
        from prondf.data import collate_fn, MultiFidelityDataset
        source = np.eye(2)[np.repeat([0, 1], 5)]
        num = np.random.randn(10, 3)
        targets = np.random.randn(10, 1)
        meta = {"dsource": 2, "dcat": [], "dnum": 3, "dtargets": 1,
                "qual_in": False, "quant_in": True, "num_samples": [5, 5]}
        ds = MultiFidelityDataset(source=source, num=num, targets=targets, meta=meta)
        loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
        batch = next(iter(loader))
        assert batch["source"].shape == (4, 2)
        assert batch["num"].shape == (4, 3)
        assert batch["targets"].shape == (4, 1)
        assert batch["cat"].shape == (4, 0)


# ---------------------------------------------------------------------------
# data — normalizers
# ---------------------------------------------------------------------------

class TestStandardNormalizer:
    def test_fit_transform_zero_mean_unit_std(self):
        from prondf.data import StandardNormalizer
        rng = np.random.default_rng(0)
        x = rng.standard_normal((100, 3)) * 5 + 2
        scaler = StandardNormalizer()
        scaler.fit(x)
        z = scaler.transform(x)
        np.testing.assert_allclose(z.mean(0), 0, atol=1e-10)
        np.testing.assert_allclose(z.std(0), 1, atol=1e-10)

    def test_inverse_transform_roundtrip(self):
        from prondf.data import StandardNormalizer
        rng = np.random.default_rng(1)
        x = rng.standard_normal((50, 2))
        scaler = StandardNormalizer()
        scaler.fit(x)
        np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(x)), x, atol=1e-10)

    def test_transform_before_fit_raises(self):
        from prondf.data import StandardNormalizer
        with pytest.raises(RuntimeError):
            StandardNormalizer().transform(np.ones((5, 2)))

    def test_to_from_dict_roundtrip(self):
        from prondf.data import StandardNormalizer
        x = np.random.randn(30, 2)
        scaler = StandardNormalizer()
        scaler.fit(x)
        restored = StandardNormalizer.from_dict(scaler.to_dict())
        np.testing.assert_allclose(restored.transform(x), scaler.transform(x), atol=1e-12)

    def test_zero_variance_column_handled(self):
        from prondf.data import StandardNormalizer
        x = np.column_stack([np.random.randn(20), np.ones(20)])
        scaler = StandardNormalizer()
        scaler.fit(x)
        z = scaler.transform(x)
        assert np.isfinite(z).all()


class TestMinMaxNormalizer:
    def test_fit_transform_range(self):
        from prondf.data import MinMaxNormalizer
        rng = np.random.default_rng(2)
        x = rng.standard_normal((100, 3))
        scaler = MinMaxNormalizer(feature_range=(0.0, 1.0))
        scaler.fit(x)
        z = scaler.transform(x)
        np.testing.assert_allclose(z.min(0), 0.0, atol=1e-10)
        np.testing.assert_allclose(z.max(0), 1.0, atol=1e-10)

    def test_inverse_transform_roundtrip(self):
        from prondf.data import MinMaxNormalizer
        rng = np.random.default_rng(3)
        x = rng.standard_normal((50, 2))
        scaler = MinMaxNormalizer()
        scaler.fit(x)
        np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(x)), x, atol=1e-10)

    def test_custom_feature_range(self):
        from prondf.data import MinMaxNormalizer
        x = np.random.randn(40, 2)
        scaler = MinMaxNormalizer(feature_range=(-1.0, 1.0))
        scaler.fit(x)
        z = scaler.transform(x)
        np.testing.assert_allclose(z.min(0), -1.0, atol=1e-10)
        np.testing.assert_allclose(z.max(0), 1.0, atol=1e-10)

    def test_to_from_dict_roundtrip(self):
        from prondf.data import MinMaxNormalizer
        x = np.random.randn(30, 3)
        scaler = MinMaxNormalizer()
        scaler.fit(x)
        restored = MinMaxNormalizer.from_dict(scaler.to_dict())
        np.testing.assert_allclose(restored.transform(x), scaler.transform(x), atol=1e-12)

    def test_constant_column_maps_to_lower_bound(self):
        from prondf.data import MinMaxNormalizer
        x = np.column_stack([np.random.randn(20), np.ones(20) * 5.0])
        scaler = MinMaxNormalizer(feature_range=(0.0, 1.0))
        scaler.fit(x)
        z = scaler.transform(x)
        np.testing.assert_allclose(z[:, 1], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# data — Generate_Analytic_Dataset
# ---------------------------------------------------------------------------

class TestGenerateAnalyticDataset:
    def test_quant_only(self):
        from prondf.data import Generate_Analytic_Dataset
        ds = Generate_Analytic_Dataset(
            dsource=2, dcat=None, dnum=2, dtargets=1,
            qual_in=False, quant_in=True,
            num_samples=[8, 8],
            source_functions=[lambda c, x: x[:, :1], lambda c, x: x[:, :1] * 2],
            num_ranges=[(-1.0, 1.0), (0.0, 2.0)],
            noise_variance=[(0.0,), (0.0,)],
            random_generator=np.random.default_rng(0),
        )
        assert len(ds) == 16
        assert ds.num.shape == (16, 2)
        assert ds.cat is None

    def test_qual_only(self):
        from prondf.data import Generate_Analytic_Dataset
        ds = Generate_Analytic_Dataset(
            dsource=2, dcat=[3, 2], dnum=0, dtargets=1,
            qual_in=True, quant_in=False,
            num_samples=[8, 8],
            source_functions=[lambda c, x: np.ones((len(c), 1)), lambda c, x: np.ones((len(c), 1))],
            num_ranges=None,
            noise_variance=[(0.0,), (0.0,)],
            random_generator=np.random.default_rng(0),
        )
        assert len(ds) == 16
        assert ds.cat.shape == (16, 2)
        assert ds.num is None

    def test_qual_and_quant(self):
        from prondf.data import Generate_Analytic_Dataset
        ds = Generate_Analytic_Dataset(
            dsource=2, dcat=[4], dnum=2, dtargets=1,
            qual_in=True, quant_in=True,
            num_samples=[8, 8],
            source_functions=[lambda c, x: np.ones((len(c), 1)), lambda c, x: np.ones((len(c), 1))],
            num_ranges=[(-1.0, 1.0), (0.0, 1.0)],
            noise_variance=[(0.0,), (0.0,)],
            random_generator=np.random.default_rng(0),
        )
        assert len(ds) == 16
        assert ds.cat.shape == (16, 1)
        assert ds.num.shape == (16, 2)

    def test_cat_values_in_range(self):
        from prondf.data import Generate_Analytic_Dataset
        dcat = [5, 3]
        ds = Generate_Analytic_Dataset(
            dsource=2, dcat=dcat, dnum=0, dtargets=1,
            qual_in=True, quant_in=False,
            num_samples=[16, 16],
            source_functions=[lambda c, x: np.ones((len(c), 1)), lambda c, x: np.ones((len(c), 1))],
            num_ranges=None,
            noise_variance=[(0.0,), (0.0,)],
            random_generator=np.random.default_rng(1),
        )
        for col, n_levels in enumerate(dcat):
            assert ds.cat[:, col].min() >= 0
            assert ds.cat[:, col].max() < n_levels

    def test_noise_variance_none_raises(self):
        from prondf.data import Generate_Analytic_Dataset
        with pytest.raises(ValueError, match="noise_variance"):
            Generate_Analytic_Dataset(
                dsource=2, dcat=None, dnum=1, dtargets=1,
                qual_in=False, quant_in=True,
                num_samples=[8, 8],
                source_functions=[lambda c, x: x, lambda c, x: x],
                num_ranges=[(-1.0, 1.0)],
                noise_variance=None,
            )

    def test_num_ranges_missing_raises(self):
        from prondf.data import Generate_Analytic_Dataset
        with pytest.raises(ValueError):
            Generate_Analytic_Dataset(
                dsource=2, dcat=None, dnum=1, dtargets=1,
                qual_in=False, quant_in=True,
                num_samples=[8, 8],
                source_functions=[lambda c, x: x, lambda c, x: x],
                num_ranges=None,
                noise_variance=[(0.0,), (0.0,)],
            )

    def test_source_one_hot(self):
        from prondf.data import Generate_Analytic_Dataset
        ds = Generate_Analytic_Dataset(
            dsource=3, dcat=None, dnum=1, dtargets=1,
            qual_in=False, quant_in=True,
            num_samples=[4, 4, 4],
            source_functions=[lambda c, x: x, lambda c, x: x, lambda c, x: x],
            num_ranges=[(-1.0, 1.0)],
            noise_variance=[(0.0,), (0.0,), (0.0,)],
        )
        assert ds.source.shape == (12, 3)
        assert (ds.source.sum(axis=1) == 1).all()

    def test_metadata_correct(self):
        from prondf.data import Generate_Analytic_Dataset
        ds = Generate_Analytic_Dataset(
            dsource=2, dcat=None, dnum=2, dtargets=3,
            qual_in=False, quant_in=True,
            num_samples=[5, 5],
            source_functions=[lambda c, x: np.ones((len(x), 3)), lambda c, x: np.ones((len(x), 3))],
            num_ranges=[(-1.0, 1.0), (-1.0, 1.0)],
            noise_variance=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        )
        assert ds.meta["dsource"] == 2
        assert ds.meta["dnum"] == 2
        assert ds.meta["dtargets"] == 3


# ---------------------------------------------------------------------------
# models — ProNDF and Build_ProNDF
# ---------------------------------------------------------------------------

class TestProNDF:
    def test_build_prondf_default(self):
        from prondf.models import Build_ProNDF
        model = Build_ProNDF(dataset_meta=_simple_meta(), loggers=[])
        assert model is not None

    def test_build_prondf_with_qual_in(self):
        from prondf.models import Build_ProNDF
        meta = _simple_meta(dsource=2, dnum=0, dtargets=1, qual_in=True, quant_in=False, dcat=[4, 3])
        model = Build_ProNDF(dataset_meta=meta, loggers=[])
        assert model is not None

    def test_build_prondf_no_loss_weighting(self):
        from prondf.models import Build_ProNDF
        model = Build_ProNDF(dataset_meta=_simple_meta(), loss_weighting=False, loggers=[])
        from prondf.losses import One_Stage_Loss_Handler
        assert isinstance(model.loss_handler, One_Stage_Loss_Handler)

    def test_build_prondf_default_loggers_not_shared(self):
        from prondf.models import Build_ProNDF
        m1 = Build_ProNDF(dataset_meta=_simple_meta())
        m2 = Build_ProNDF(dataset_meta=_simple_meta())
        assert m1._loggers is not m2._loggers

    def test_forward_output_shape(self):
        from prondf.models import Build_ProNDF
        model = Build_ProNDF(dataset_meta=_simple_meta(dsource=2, dnum=3, dtargets=1), loggers=[])
        batch = _make_batch(dsource=2, dnum=3, dtargets=1, n=8)
        out = model(batch)
        assert out.shape == (8, 1)

    def test_forward_multi_output(self):
        from prondf.models import Build_ProNDF
        model = Build_ProNDF(dataset_meta=_simple_meta(dsource=2, dnum=3, dtargets=4), loggers=[])
        batch = _make_batch(dsource=2, dnum=3, dtargets=4, n=6)
        out = model(batch)
        assert out.shape == (6, 4)

    def test_get_model_outputs_keys(self):
        from prondf.models import Build_ProNDF
        model = Build_ProNDF(dataset_meta=_simple_meta(), loggers=[])
        batch = _make_batch()
        outputs = model.get_model_outputs(batch)
        assert "B1" in outputs and "B3" in outputs
        assert "out" in outputs["B3"]
        assert "out_dist" in outputs["B3"]

    def test_get_model_outputs_with_qual_in(self):
        from prondf.models import Build_ProNDF
        meta = _simple_meta(dsource=2, dnum=0, qual_in=True, quant_in=False, dcat=[4, 3])
        model = Build_ProNDF(dataset_meta=meta, loggers=[])
        batch = _make_batch(dsource=2, dnum=0, qual_in=True, dcat_sum=7)
        outputs = model.get_model_outputs(batch)
        assert "B2" in outputs

    def test_training_step_returns_scalar(self):
        from prondf.models import Build_ProNDF
        model = Build_ProNDF(dataset_meta=_simple_meta(), loss_weighting=False, loggers=[])
        batch = _make_batch()
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0

    def test_validation_step_returns_scalar(self):
        from prondf.models import Build_ProNDF
        model = Build_ProNDF(dataset_meta=_simple_meta(), loss_weighting=False, loggers=[])
        batch = _make_batch()
        loss = model.validation_step(batch, 0)
        assert loss.ndim == 0

    def test_invalid_dsource_raises(self):
        from prondf.models import ProNDF
        with pytest.raises(ValueError):
            ProNDF(
                dsource=0, dcat=[], dnum=2, dout=1,
                qual_in=False, quant_in=True,
                B1_type="Det_Block", B1_config={"d_in": 0, "d_out": 2, "hidden_layers": [4]},
                B2_type="Det_Block", B2_config={"d_in": 0, "d_out": 2, "hidden_layers": [4]},
                B3_type="Prob_Block", B3_config={"d_in": 4, "d_out": 1, "hidden_layers": [8]},
                loss_handler_type="One_Stage_Loss_Handler",
                loss_handler_config={
                    "loss_function_classes": ["Output_NLL_loss"],
                    "loss_function_configs": [{}],
                    "data_split_classes": ["No_Split"],
                    "data_split_configs": [{}],
                    "LW_alg_classes": ["No_Weighting"],
                    "LW_alg_configs": [{}],
                },
                optimizer_type="Adam",
                optimizer_config={"lr": 0.001},
            )

    def test_probabilistic_manifolds_option(self):
        from prondf.models import Build_ProNDF
        from prondf.blocks import Prob_Block
        model = Build_ProNDF(
            dataset_meta=_simple_meta(),
            probabilistic_manifolds=True,
            loggers=[],
        )
        assert isinstance(model.B1, Prob_Block)

    def test_deterministic_output_option(self):
        from prondf.models import Build_ProNDF
        from prondf.blocks import Det_Block
        model = Build_ProNDF(
            dataset_meta=_simple_meta(),
            probabilistic_output=False,
            loggers=[],
        )
        assert isinstance(model.B3, Det_Block)

    def test_checkpoint_roundtrip(self, tmp_path):
        """Regression: weighting alg state must survive save/load."""
        import pytorch_lightning as pl
        from prondf.models import Build_ProNDF
        from prondf.data import MultiFidelityDataset, collate_fn
        from prondf.data import Generate_Analytic_Dataset, split_dataset

        ds = Generate_Analytic_Dataset(
            dsource=2, dcat=None, dnum=1, dtargets=1,
            qual_in=False, quant_in=True, num_samples=[32, 32],
            source_functions=[lambda c, x: x, lambda c, x: x * 2],
            num_ranges=[(-1.0, 1.0)],
            noise_variance=[(0.01,), (0.01,)],
            random_generator=np.random.default_rng(0),
        )
        train, val, _ = split_dataset(ds, [0.7, 0.2, 0.1])
        model = Build_ProNDF(dataset_meta=train.meta, loss_weighting=True, loggers=[])
        loader = DataLoader(train, batch_size=16, collate_fn=collate_fn)
        val_loader = DataLoader(val, batch_size=16, collate_fn=collate_fn)

        ckpt_path = str(tmp_path / "test.ckpt")
        trainer = pl.Trainer(
            max_epochs=2, enable_progress_bar=False, enable_model_summary=False,
            logger=False, enable_checkpointing=False,
        )
        trainer.fit(model, loader, val_loader)

        # Save and reload state dict
        torch.save(model.state_dict(), ckpt_path)
        sd = torch.load(ckpt_path)
        assert any("weights" in k for k in sd.keys()), (
            "Weighting alg buffers not in saved state_dict"
        )


# ---------------------------------------------------------------------------
# plotting — smoke tests (no crash, returns Figure)
# ---------------------------------------------------------------------------

class TestPlotting:
    @pytest.fixture(autouse=True)
    def use_agg(self):
        import matplotlib
        matplotlib.use("Agg")

    def _make_trained_model(self):
        import pytorch_lightning as pl
        from prondf.models import Build_ProNDF
        from prondf.data import Generate_Analytic_Dataset, split_dataset, collate_fn

        ds = Generate_Analytic_Dataset(
            dsource=2, dcat=None, dnum=1, dtargets=1,
            qual_in=False, quant_in=True, num_samples=[32, 32],
            source_functions=[lambda c, x: x, lambda c, x: x * 2],
            num_ranges=[(-1.0, 1.0)],
            noise_variance=[(0.0,), (0.0,)],
            random_generator=np.random.default_rng(0),
        )
        train, val, test = split_dataset(ds, [0.7, 0.2, 0.1])
        model = Build_ProNDF(dataset_meta=train.meta, loggers=[])
        loader = DataLoader(train, batch_size=16, collate_fn=collate_fn)
        val_loader = DataLoader(val, batch_size=16, collate_fn=collate_fn)
        trainer = pl.Trainer(
            max_epochs=2, enable_progress_bar=False, enable_model_summary=False,
            logger=False, enable_checkpointing=False,
        )
        trainer.fit(model, loader, val_loader)
        return model, train, test

    def test_plot_true_pred_returns_figure(self):
        import matplotlib.pyplot as plt
        from prondf.plotting import plot_true_pred
        model, _, test = self._make_trained_model()
        fig = plot_true_pred(model=model, test_dataset=test)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_1d_returns_figure(self):
        import matplotlib.pyplot as plt
        from prondf.plotting import plot_1D
        model, train, test = self._make_trained_model()
        fig = plot_1D(model=model, train_dataset=train, val_dataset=test, test_dataset=test)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_2d_latent_space_b1_returns_figure(self):
        import matplotlib.pyplot as plt
        from prondf.plotting import plot_2D_latent_space
        model, _, _ = self._make_trained_model()
        fig = plot_2D_latent_space(model=model, block_idx=0, dcat=None)
        assert isinstance(fig, plt.Figure)
        plt.close("all")
