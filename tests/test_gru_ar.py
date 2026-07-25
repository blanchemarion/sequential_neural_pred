from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from infer.inference_gru_ar import create_inference_model, evaluate_long_window
from models.model_KV_cached import create_model_cached
from models.model_gru_ar import create_gru_ar


class TestGRUAR(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(101)
        self.model = create_gru_ar(
            hidden_dim=12,
            num_layers=8,
            context_length=90,
            forecast_length=90,
        )
        self.model.eval()
        self.context = torch.randn(2, 90, 16)
        self.target = torch.randn(2, 90, 16)

    def test_teacher_forced_and_autoregressive_shapes_and_first_prediction(self):
        with torch.no_grad():
            teacher_forced = self.model.forward_teacher_forced(
                self.context, self.target
            )
            autoregressive = self.model.forecast_autoregressive(
                self.context, horizon=90
            )
        self.assertEqual(tuple(teacher_forced.shape), (2, 90, 16))
        self.assertEqual(tuple(autoregressive.shape), (2, 90, 16))
        self.assertTrue(torch.equal(teacher_forced[:, 0], autoregressive[:, 0]))

    def test_teacher_forcing_uses_target_shift_without_same_step_leakage(self):
        changed = self.target.clone()
        changed[:, 10, :] += 25.0
        with torch.no_grad():
            original_prediction = self.model.forward_teacher_forced(
                self.context, self.target
            )
            changed_prediction = self.model.forward_teacher_forced(
                self.context, changed
            )

        # target[10] is first consumed while producing prediction[11].
        self.assertTrue(
            torch.equal(original_prediction[:, :11], changed_prediction[:, :11])
        )
        self.assertFalse(
            torch.equal(original_prediction[:, 11], changed_prediction[:, 11])
        )

        changed_last = self.target.clone()
        changed_last[:, -1, :] -= 100.0
        with torch.no_grad():
            changed_last_prediction = self.model.forward_teacher_forced(
                self.context, changed_last
            )
        self.assertTrue(torch.equal(original_prediction, changed_last_prediction))

    def test_autoregressive_interface_has_no_target_and_is_target_independent(self):
        parameters = tuple(
            inspect.signature(self.model.forecast_autoregressive).parameters
        )
        self.assertEqual(parameters, ("context", "horizon"))
        with torch.no_grad():
            before = self.model.forecast_autoregressive(self.context, 90)
            self.target.normal_(mean=1000.0, std=500.0)
            after = self.model.forecast_autoregressive(self.context, 90)
        self.assertTrue(torch.equal(before, after))

    def test_long_blockwise_rollout_shape_and_finiteness(self):
        context = self.context[:1]
        encoded_context_calls = 0

        def count_context_encodes(_module, inputs):
            nonlocal encoded_context_calls
            if inputs[0].shape[1] == 90:
                encoded_context_calls += 1

        hook = self.model.gru.register_forward_pre_hook(count_context_encodes)
        try:
            with torch.no_grad():
                rollout = self.model.forecast_blockwise(context, horizon=720)
        finally:
            hook.remove()
        self.assertEqual(tuple(rollout.shape), (1, 720, 16))
        self.assertEqual(encoded_context_calls, 8)
        self.assertTrue(torch.isfinite(rollout).all().item())

    def test_all_short_outputs_are_finite(self):
        with torch.no_grad():
            teacher_forced = self.model.forward_teacher_forced(
                self.context, self.target
            )
            autoregressive = self.model.forecast_autoregressive(self.context, 90)
        self.assertTrue(torch.isfinite(teacher_forced).all().item())
        self.assertTrue(torch.isfinite(autoregressive).all().item())

    def test_checkpoint_round_trip_preserves_predictions(self):
        configured_model = create_gru_ar(
            hidden_dim=12,
            num_layers=8,
            context_length=90,
            forecast_length=90,
        )
        configured_model.eval()
        config = {
            "model_name": "GRU_AR",
            "n_vars": 16,
            "hidden_dim": 12,
            "num_layers": 8,
            "dropout": 0.05,
            "T_in": 90,
            "T_out": 90,
            "parameter_count": configured_model.count_parameters(),
        }
        with torch.no_grad():
            expected = configured_model.forecast_autoregressive(self.context, 90)

        checkpoint_path = ROOT / "tests" / "_tmp_gru_ar_checkpoint.pt"
        try:
            torch.save(
                {
                    "model_class": "GRUAR",
                    "model_state_dict": configured_model.state_dict(),
                    "parameter_count": configured_model.count_parameters(),
                    "config": config,
                },
                checkpoint_path,
            )
            loaded, loaded_config = create_inference_model(
                checkpoint_path, torch.device("cpu")
            )
            with torch.no_grad():
                actual = loaded.forecast_autoregressive(self.context, 90)
        finally:
            checkpoint_path.unlink(missing_ok=True)

        self.assertEqual(loaded_config["hidden_dim"], 12)
        self.assertTrue(torch.equal(expected, actual))

    def test_inference_output_matches_existing_baseline_layout(self):
        output_dir = ROOT / "tests" / "_tmp_gru_ar_output"
        expected_names = (
            "long_predictions_90_720_GRU_AR.npy",
            "long_ground_truth_90_720.npy",
            "long_predictions_scored_GRU_AR.csv",
            "long_ground_truth_scored.csv",
        )
        try:
            val_examples = np.zeros((1, 16, 180), dtype=np.float32)
            val_sequence_indices = np.array([0], dtype=np.int64)
            evaluate_long_window(
                self.model,
                val_examples,
                val_sequence_indices,
                np.array([0], dtype=np.int64),
                output_dir,
                t_in=90,
                target_pred_length=720,
                device=torch.device("cpu"),
                seed=102,
                n_plot_examples=0,
            )
            self.assertEqual(
                {path.name for path in output_dir.iterdir()}, set(expected_names)
            )
            full_prediction = np.load(
                output_dir / "long_predictions_90_720_GRU_AR.npy"
            )
            self.assertEqual(full_prediction.shape, (1, 810, 16))
            self.assertEqual(full_prediction.dtype, np.float32)

            scored_path = output_dir / "long_predictions_scored_GRU_AR.csv"
            with scored_path.open(encoding="utf-8") as file:
                header = file.readline().strip().split(",")
                row_count = sum(1 for _ in file)
            self.assertEqual(
                header,
                ["sequenceId", "itemPosition"]
                + [f"roi_{index}" for index in range(16)],
            )
            self.assertEqual(row_count, 720)
        finally:
            for name in expected_names:
                (output_dir / name).unlink(missing_ok=True)
            if output_dir.exists():
                output_dir.rmdir()

    def test_small_synthetic_training_decreases_mae(self):
        torch.manual_seed(7)
        model = create_gru_ar(
            hidden_dim=12,
            num_layers=8,
            context_length=6,
            forecast_length=4,
        )
        context = torch.zeros(8, 6, 16)
        target = torch.zeros(8, 4, 16)
        criterion = nn.L1Loss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)

        model.eval()
        with torch.no_grad():
            initial = criterion(model.forward_teacher_forced(context, target), target)
        model.train()
        for _ in range(30):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model.forward_teacher_forced(context, target), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            final = criterion(model.forward_teacher_forced(context, target), target)
        self.assertLess(float(final), float(initial))

    def test_architecture_tracks_applicable_transformer_properties(self):
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
        gru = create_gru_ar(hidden_dim=64, num_layers=8)

        self.assertEqual(gru.hidden_dim, transformer.d_model)
        self.assertEqual(gru.hidden_dim, 64)
        self.assertEqual(gru.num_layers, 8)
        self.assertEqual(gru.dropout_p, 0.05)
        self.assertEqual(len(transformer.blocks), 8)
        self.assertEqual(transformer.blocks[0].attn.n_heads, 8)
        self.assertEqual(transformer.blocks[0].mlp[0].out_features, 128)
        self.assertEqual(gru.count_parameters(), 191_632)
        self.assertEqual(transformer.count_parameters(), 281_616)


if __name__ == "__main__":
    unittest.main()
