from __future__ import annotations

import inspect
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from infer.inference_cdmm_ssm import (
    create_inference_model,
    evaluate_long_window,
    prediction_output_directory,
)
from models.model_KV_cached import create_model_cached
from models.model_cdmm_ssm import create_cdmm_ssm
from models.model_gru_ar import create_gru_ar
from train.train_cdmm_ssm import beta_for_epoch


class TestConditionalDeepMarkovSSM(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(101)
        self.model = create_cdmm_ssm()
        self.model.eval()
        self.context = torch.randn(2, 90, 16)
        self.target = torch.randn(2, 90, 16)

    def test_mean_and_sampled_forecast_shapes(self):
        with torch.no_grad():
            mean = self.model.forecast_mean(self.context, 90)
            sample = self.model.forecast_sample(self.context, 90, seed=5)
            samples = self.model.forecast_samples(
                self.context, 90, num_samples=6, seed=5
            )
        self.assertEqual(tuple(mean.shape), (2, 90, 16))
        self.assertEqual(tuple(sample.shape), (2, 90, 16))
        self.assertEqual(tuple(samples.shape), (6, 2, 90, 16))
        self.assertTrue(torch.isfinite(mean).all().item())
        self.assertTrue(torch.isfinite(samples).all().item())

    def test_long_forecasts_and_independent_block_histories(self):
        context = self.context[:1]
        encoded_contexts = []

        def capture_context(_module, inputs):
            encoded_contexts.append(inputs[0].detach().clone())

        hook = self.model.context_encoder.register_forward_pre_hook(capture_context)
        try:
            with torch.no_grad():
                mean = self.model.forecast_mean_blockwise(context, horizon=720)
                samples = self.model.forecast_samples_blockwise(
                    context, horizon=720, num_samples=6, seed=17
                )
        finally:
            hook.remove()
        self.assertEqual(tuple(mean.shape), (1, 720, 16))
        self.assertEqual(tuple(samples.shape), (6, 1, 720, 16))
        # Eight mean blocks plus eight sampled blocks are each re-encoded.
        self.assertEqual(len(encoded_contexts), 16)
        first_generated_sample_context = encoded_contexts[9]
        self.assertFalse(
            torch.equal(
                first_generated_sample_context[0],
                first_generated_sample_context[1],
            )
        )

    def test_elbo_and_gradients_are_finite(self):
        self.model.train()
        loss, diagnostics = self.model.conditional_elbo(
            self.context, self.target, beta=0.2, free_bits=0.05
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        for value in diagnostics.values():
            self.assertTrue(torch.isfinite(value).item())
        gradients = [
            parameter.grad
            for parameter in self.model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_posterior_uses_exact_causal_target_prefix(self):
        # The production model intentionally initializes the posterior delta
        # head at zero for a stable q ~= p start. Give the test-only head a
        # nonzero mapping so target-prefix influence is observable in q means.
        with torch.no_grad():
            self.model.posterior_mlp[-1].weight.normal_(mean=0.0, std=0.01)
        changed = self.target.clone()
        changed[:, 10, :] += 50.0
        with torch.no_grad():
            original = self.model.posterior_rollout(
                self.context, self.target, sample=False
            )
            modified = self.model.posterior_rollout(
                self.context, changed, sample=False
            )
        self.assertTrue(
            torch.equal(
                original["posterior_states"][:, :10],
                modified["posterior_states"][:, :10],
            )
        )
        self.assertTrue(
            torch.equal(
                original["posterior_means"][:, :10],
                modified["posterior_means"][:, :10],
            )
        )
        # q(z_11) observes target[:, :11], including changed target[:, 10].
        self.assertFalse(
            torch.equal(
                original["posterior_means"][:, 10],
                modified["posterior_means"][:, 10],
            )
        )

    def test_context_only_inference_is_target_independent_and_mean_is_deterministic(self):
        parameters = tuple(
            inspect.signature(self.model.forecast_mean).parameters
        )
        self.assertEqual(parameters, ("context", "horizon"))
        with torch.no_grad():
            first = self.model.forecast_mean(self.context, 90)
            self.target.normal_(mean=1000.0, std=100.0)
            second = self.model.forecast_mean(self.context, 90)
        self.assertTrue(torch.equal(first, second))

    def test_sampling_is_stochastic_and_seed_reproducible(self):
        with torch.no_grad():
            first = self.model.forecast_sample(self.context, 90, seed=123)
            repeated = self.model.forecast_sample(self.context, 90, seed=123)
            different = self.model.forecast_sample(self.context, 90, seed=124)
        self.assertTrue(torch.equal(first, repeated))
        self.assertFalse(torch.equal(first, different))

    def test_checkpoint_round_trip_preserves_mean_and_seeded_sample(self):
        config = {
            "model_name": "cDMM_SSM",
            "n_vars": 16,
            "latent_dim": 8,
            "encoder_hidden_dim": 64,
            "posterior_hidden_dim": 64,
            "transition_hidden_dim": 32,
            "decoder_hidden_dims": [32, 32],
            "T_in": 90,
            "T_out": 90,
            "spectral_radius": 0.99,
            "min_scale": 1e-4,
            "min_log_variance": -12.0,
            "max_log_variance": 8.0,
            "parameter_count": self.model.count_parameters(),
        }
        with torch.no_grad():
            expected_mean = self.model.forecast_mean(self.context, 90)
            expected_sample = self.model.forecast_sample(
                self.context, 90, seed=991
            )
        checkpoint_path = ROOT / "tests" / "_tmp_cdmm_checkpoint.pt"
        try:
            torch.save(
                {
                    "model_class": "ConditionalDeepMarkovSSM",
                    "model_state_dict": self.model.state_dict(),
                    "parameter_count": self.model.count_parameters(),
                    "config": config,
                },
                checkpoint_path,
            )
            loaded, loaded_config = create_inference_model(
                checkpoint_path, torch.device("cpu")
            )
            with torch.no_grad():
                actual_mean = loaded.forecast_mean(self.context, 90)
                actual_sample = loaded.forecast_sample(
                    self.context, 90, seed=991
                )
        finally:
            checkpoint_path.unlink(missing_ok=True)
        self.assertEqual(loaded_config["latent_dim"], 8)
        self.assertTrue(torch.equal(expected_mean, actual_mean))
        self.assertTrue(torch.equal(expected_sample, actual_sample))

    def test_transition_spectral_norm_is_always_constrained(self):
        with torch.no_grad():
            self.model.transition_matrix_raw.normal_(mean=0.0, std=5.0)
        self.assertAlmostEqual(
            self.model.transition_spectral_norm(), 0.99, places=5
        )
        optimizer = torch.optim.AdamW(
            [self.model.transition_matrix_raw], lr=0.1
        )
        optimizer.zero_grad(set_to_none=True)
        loss = self.model.stable_transition_matrix().sum()
        loss.backward()
        optimizer.step()
        self.assertAlmostEqual(
            self.model.transition_spectral_norm(), 0.99, places=5
        )

    def test_small_synthetic_training_decreases_elbo_loss(self):
        torch.manual_seed(7)
        model = create_cdmm_ssm(
            latent_dim=4, context_length=6, forecast_length=4
        )
        context = torch.zeros(8, 6, 16)
        target = torch.zeros(8, 4, 16)

        def fixed_noise_loss():
            generator = torch.Generator().manual_seed(55)
            return model.conditional_elbo(
                context,
                target,
                beta=0.2,
                free_bits=0.05,
                generator=generator,
            )[0]

        model.eval()
        initial = float(fixed_noise_loss().detach())
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        for _ in range(40):
            optimizer.zero_grad(set_to_none=True)
            loss = fixed_noise_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        final = float(fixed_noise_loss().detach())
        self.assertLess(final, initial)

    def test_inference_output_preserves_six_sample_order(self):
        output_dir = ROOT / "tests" / "_tmp_cdmm_output"
        outputs = {}
        try:
            val_examples = np.zeros((1, 16, 180), dtype=np.float32)
            outputs = evaluate_long_window(
                self.model,
                val_examples,
                np.array([0], dtype=np.int64),
                np.array([0], dtype=np.int64),
                output_dir,
                t_in=90,
                target_pred_length=720,
                device=torch.device("cpu"),
                seed=102,
                num_samples=6,
                n_plot_examples=0,
            )
            means = np.load(outputs["mean_npy"])
            samples = np.load(outputs["samples_npy"])
            self.assertEqual(
                outputs["mean_npy"].name,
                "long_predictions_90_810_cDMM_SSM_mean.npy",
            )
            self.assertEqual(
                outputs["samples_npy"].name,
                "long_predictions_90_810_cDMM_SSM_samples.npy",
            )
            self.assertEqual(means.shape, (1, 810, 16))
            self.assertEqual(samples.shape, (6, 1, 810, 16))
            self.assertEqual(means.dtype, np.float32)
            self.assertEqual(samples.dtype, np.float32)
            for sample_index in range(6):
                per_sample = np.load(
                    outputs[f"sample_{sample_index + 1:02d}_npy"]
                )
                self.assertTrue(np.array_equal(per_sample, samples[sample_index]))
            metadata = json.loads(
                outputs["metadata_json"].read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["sample_archive_axes"],
                ["sample", "sequence", "time", "region"],
            )
            self.assertEqual(
                metadata["sample_order"],
                [f"cDMM_SSM_samples_s{index:02d}" for index in range(1, 7)],
            )
            self.assertFalse(metadata["posterior_used_for_inference"])
        finally:
            for path in outputs.values():
                path.unlink(missing_ok=True)
            if output_dir.exists():
                output_dir.rmdir()

    def test_prediction_output_directory_nomenclature(self):
        self.assertEqual(
            prediction_output_directory(
                Path("evaluation_results"), 102, 90, 720
            ),
            Path("evaluation_results") / "seed_102" / "90_810",
        )

    def test_beta_schedule_latent_support_and_parameter_counts(self):
        self.assertEqual(beta_for_epoch(1, 10, 0.2, 0.3), 0.0)
        self.assertEqual(beta_for_epoch(3, 10, 0.2, 0.3), 0.2)
        for latent_dim in (4, 8, 12):
            self.assertEqual(create_cdmm_ssm(latent_dim=latent_dim).latent_dim, latent_dim)

        cdmm = create_cdmm_ssm()
        gru = create_gru_ar(hidden_dim=64, num_layers=8)
        transformer = create_model_cached(
            n_vars=16,
            d_model=64,
            n_heads=8,
            n_layers=8,
            d_ff=128,
            dropout=0.05,
            T_in=90,
            T_out=90,
        )
        self.assertEqual(cdmm.count_parameters(), 40_824)
        self.assertEqual(gru.count_parameters(), 191_632)
        self.assertEqual(transformer.count_parameters(), 281_616)


if __name__ == "__main__":
    unittest.main()
