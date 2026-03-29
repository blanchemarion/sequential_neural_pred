

"""
Decoder-Only Transformer for Multi-Step Time Series Prediction

4 different training regimes:
- OS: One Shot
- AR: Self Feedback Autoregressive
- TF: Autoregressive with Teacher Forcing
"""

import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    def __init__(
        self,
        n_vars=16,
        d_model=64,
        n_heads=4,
        n_layers=4,
        d_ff=None,
        dropout=0.1,
        T_in=90,
        T_out=90,
    ):
        super().__init__()

        self.n_vars = n_vars
        self.T_in = T_in
        self.T_out = T_out
        self.d_model = d_model
        d_ff = d_ff or 4 * d_model

        # Total token positions per (history + future) block
        self.total_tokens = T_in + T_out

        # Plain token projections
        self.in_proj = nn.Linear(n_vars, d_model)
        self.out_proj = nn.Linear(d_model, n_vars)

        # Positional embeddings (token-level)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.total_tokens, d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.future_start = nn.Parameter(torch.zeros(1, 1, self.d_model))


    def _positions_block_relative(self, L: int, start_in_block: int, device):
        pos = torch.arange(L, device=device) + start_in_block
        if torch.any(pos >= self.total_tokens):
            raise ValueError(
                f"Pos index overflow: max={int(pos.max())} but total_tokens={self.total_tokens}. "
                "Increase total_tokens or reset block/offset earlier."
            )
        return pos

    def _local_causal_mask(self, L: int, device, window: int):
        """
        True = masked (disallowed) for nn.TransformerEncoder.
        window = max lookback length (in tokens).
        """
        idx = torch.arange(L, device=device)
        j = idx.view(1, L)   # keys
        i = idx.view(L, 1)   # queries

        future = j > i
        too_far_past = j < (i - (window - 1))
        return future | too_far_past

    def forward_oneshot(self, x, return_attention=False):
        """
        One-shot: history tokens + zero future tokens, predict all future in parallel.
        """
        B = x.size(0)
        device = x.device

        # Embed history tokens
        input_tokens = self.in_proj(x)              # (B, T_in, d)
        input_tokens = self.dropout(input_tokens)

        # Zero future inputs (same as your oneshot placeholders)
        zeros_tokens = torch.zeros(
            B, self.T_out, self.d_model,
            device=device, dtype=input_tokens.dtype
        )

        all_tokens = torch.cat([input_tokens, zeros_tokens], dim=1)  # (B, T_in+T_out, d)
        L = all_tokens.size(1)
        all_tokens = all_tokens + self.pos_embedding[:, :L, :]

        mask = self._local_causal_mask(L, device=device, window=self.T_in)
        out = self.transformer_encoder(all_tokens, mask=mask)        # (B, L, d)

        # Take future positions and decode
        future_tokens = out[:, self.T_in:, :]                        # (B, T_out, d)
        predictions = self.out_proj(future_tokens)                   # (B, T_out, n_vars)

        if return_attention:
            raise NotImplementedError("Attention extraction not implemented in this rewrite.")
        return predictions
    
    def forward_teacher_forcing(self, x: torch.Tensor, y: torch.Tensor):
        """
        Teacher-forced AR training (GPT/Neuroformer-style), but for continuous values.

        Args:
            x: (B, T_in, n_vars)   ground-truth context / history
            y: (B, T_out, n_vars)  ground-truth future targets

        Returns:
            pred: (B, T_out, n_vars)  predictions for each future step
        """
        B, T_in, V = x.shape
        B2, T_out, V2 = y.shape
        assert B == B2 and V == V2, "x and y must match batch and n_vars"
        assert T_in == self.T_in and T_out == self.T_out, "x/y lengths must match model T_in/T_out"

        device = x.device
        dtype = x.dtype

        # 1) Embed the history tokens
        hist_tokens = self.dropout(self.in_proj(x))  # (B, T_in, d)

        # 2) Build teacher-forced future input tokens:
        #    token 0 is a learned BOS-like vector (self.future_start),
        #    then tokens 1..T_out-1 are embeddings of ground-truth y[0..T_out-2]
        bos = self.future_start.expand(B, 1, self.d_model).to(device=device, dtype=hist_tokens.dtype)

        if T_out > 1:
            y_shift = y[:, :-1, :]                              # (B, T_out-1, n_vars)
            y_shift_tokens = self.dropout(self.in_proj(y_shift)) # (B, T_out-1, d)
            fut_in_tokens = torch.cat([bos, y_shift_tokens], dim=1)  # (B, T_out, d)
        else:
            fut_in_tokens = bos  # (B, 1, d)

        # 3) Concatenate into full sequence [history, teacher-forced future inputs]
        all_tokens = torch.cat([hist_tokens, fut_in_tokens], dim=1)  # (B, T_in+T_out, d)
        L = all_tokens.size(1)

        # 4) Add positional embeddings
        all_tokens = all_tokens + self.pos_embedding[:, :L, :]

        # 5) Causal/local mask
        mask = self._local_causal_mask(L, device=device, window=self.T_in)

        # 6) Transformer pass
        out = self.transformer_encoder(all_tokens, mask=mask)  # (B, L, d)

        # 7) Decode ONLY future positions -> predictions for y[0..T_out-1]
        future_tokens = out[:, self.T_in:, :]                   # (B, T_out, d)
        pred = self.out_proj(future_tokens)                     # (B, T_out, n_vars)

        return pred

    def forward_autoregressive(self, x, block_offset=0):
        """
        AR consistent with forward()/forward_combined():
        future input tokens are [0, in_proj(yhat_0), in_proj(yhat_1), ...]
        """
        B = x.size(0)
        device = x.device

        input_tokens = self.in_proj(x)            # (B, T_in, d)
        input_tokens = self.dropout(input_tokens)

        # shift-style: token_0 = zeros (input for predicting y0)
        future_inputs = [torch.zeros(B, 1, self.d_model, device=device, dtype=input_tokens.dtype)]
        preds = []  # list of (B, 1, n_vars)

        for i in range(self.T_out):
            q = block_offset + self.T_in + i  # absolute position within block

            all_tokens = torch.cat([input_tokens, torch.cat(future_inputs, dim=1)], dim=1)
            # Keep last (T_in + 1) tokens ending at q
            L_keep = min(all_tokens.size(1), self.T_in + 1)
            current = all_tokens[:, -L_keep:, :]  # (B, L_keep, d)

            start_pos = q - (L_keep - 1)
            positions = self._positions_block_relative(L_keep, start_pos, device=device)
            current = current + self.pos_embedding[:, positions, :]

            mask = torch.triu(
                torch.ones(L_keep, L_keep, device=device, dtype=torch.bool),
                diagonal=1
            )
            out = self.transformer_encoder(current, mask=mask)

            #mask = self._local_causal_mask(L_keep, device=device, window=self.T_in)
            #out = self.transformer_encoder(current, mask=mask)  # (B, L_keep, d)

            pred_token = out[:, -1:, :]                         # (B, 1, d)
            pred_val = self.out_proj(pred_token)                # (B, 1, n_vars)
            preds.append(pred_val)

            if i < self.T_out - 1:
                next_in = self.dropout(self.in_proj(pred_val))
                future_inputs.append(next_in)

        return torch.cat(preds, dim=1)                          # (B, T_out, n_vars)

    def forward_combined(
            self,
            x, y,
            p_teacher_start=1.0,
            p_teacher_end=0.1,
            schedule="linear",
            noise_std=0.0,
            detach_self=True,
        ):
            """
            Two-pass scheduled sampling approximation:
            Pass 1: oneshot-ish (future inputs = zeros) -> yhat0
            Build shifted future inputs as mix of shifted_y and shifted_yhat0
            Pass 2: run with mixed future inputs -> yhat (used for loss)

            Cost: ~2 transformer passes (instead of T_out passes).
            """
            B = x.size(0)
            device = x.device

            input_tokens = self.dropout(self.in_proj(x))  # (B,T_in,d)

            # ---------- Pass 1: oneshot context (all future inputs zeros) ----------
            zeros_future = torch.zeros(B, self.T_out, self.d_model, device=device, dtype=input_tokens.dtype)
            all_tokens_1 = torch.cat([input_tokens, zeros_future], dim=1)  # (B, T_in+T_out, d)

            L = all_tokens_1.size(1)
            all_tokens_1 = all_tokens_1 + self.pos_embedding[:, :L, :]

            mask = self._local_causal_mask(L, device=device, window=self.T_in)
            with torch.no_grad():
                out_1 = self.transformer_encoder(all_tokens_1, mask=mask)
                yhat0 = self.out_proj(out_1[:, self.T_in:, :])

            # ---------- Build mixed shifted future inputs ----------
            y_tokens = self.in_proj(y)          # (B,T_out,d)
            yhat0_tokens = self.in_proj(yhat0)  # (B,T_out,d)

            # Shift both (input token t is previous value)
            shifted_y = torch.cat(
                [torch.zeros(B, 1, self.d_model, device=device, dtype=y_tokens.dtype),
                y_tokens[:, :-1, :]],
                dim=1
            )
            shifted_yhat = torch.cat(
                [torch.zeros(B, 1, self.d_model, device=device, dtype=yhat0_tokens.dtype),
                yhat0_tokens[:, :-1, :]],
                dim=1
            )

            if noise_std > 0:
                shifted_y = shifted_y + torch.randn_like(shifted_y) * noise_std

            # Per-timestep teacher probability schedule
            if schedule == "linear":
                t = torch.linspace(0, 1, steps=self.T_out, device=device)  # (T_out,)
                p_teacher_t = p_teacher_start + (p_teacher_end - p_teacher_start) * t
            else:
                raise ValueError("Implement other schedules if needed.")

            # Sample per (B, T_out) whether we use teacher at each step
            use_teacher = (torch.rand(B, self.T_out, device=device) < p_teacher_t.view(1, -1)).float()
            use_teacher = use_teacher.unsqueeze(-1)  # (B,T_out,1)

            future_inputs = use_teacher * shifted_y + (1.0 - use_teacher) * shifted_yhat
            future_inputs = self.dropout(future_inputs)

            # ---------- Pass 2: mixed future inputs ----------
            all_tokens_2 = torch.cat([input_tokens, future_inputs], dim=1)
            L2 = all_tokens_2.size(1)
            all_tokens_2 = all_tokens_2 + self.pos_embedding[:, :L2, :]

            out_2 = self.transformer_encoder(all_tokens_2, mask=mask)
            yhat = self.out_proj(out_2[:, self.T_in:, :])
            return yhat
     

    
    def count_parameters(self):
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(n_vars=16, d_model=64, n_heads=4, n_layers=4, d_ff=None, dropout=0.1,
                 T_in=70, T_out=21, device='cpu'):

    model = TransformerModel(
        n_vars=n_vars,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        T_in=T_in,
        T_out=T_out,
    ).to(device)

    return model



if __name__ == "__main__":
    # Test the model
    print("Testing TransformerModel model...")
    
    # Model parameters
    n_vars = 16
    T_in = 90
    T_out = 90
    batch_size = 4
    
    # Create model
    model = create_model(
        n_vars=n_vars,
        d_model=64,
        n_heads=4,
        n_layers=4,
        T_in=T_in,
        T_out=T_out,
    )
    
    print(f"\nModel created:")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Input shape: (batch, {T_in}, {n_vars})")
    print(f"  Output shape: (batch, {T_out}, {n_vars})")
    
    # Test input
    x = torch.randn(batch_size, T_in, n_vars)
    
    # Test one-shot forward pass
    print(f"\n{'='*60}")
    print("Testing ONE-SHOT inference (original method)")
    print('='*60)
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        import time
        start = time.time()
        output_oneshot = model(x)
        elapsed_oneshot = time.time() - start
        
        print(f"  Output shape: {output_oneshot.shape}")
        print(f"  Expected: ({batch_size}, {T_out}, {n_vars})")
        print(f"  Time: {elapsed_oneshot*1000:.2f} ms")
        
        if output_oneshot.shape == (batch_size, T_out, n_vars):
            print("  [OK] One-shot inference passed!")
        else:
            print("  [ERROR] Shape mismatch!")
    
    # Test autoregressive forward pass
    print(f"\n{'='*60}")
    print("Testing AUTOREGRESSIVE inference (new method)")
    print('='*60)
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        start = time.time()
        output_autoreg = model.forward_autoregressive_old(x)
        elapsed_autoreg = time.time() - start
        
        print(f"  Output shape: {output_autoreg.shape}")
        print(f"  Expected: ({batch_size}, {T_out}, {n_vars})")
        print(f"  Time: {elapsed_autoreg*1000:.2f} ms ({elapsed_autoreg/elapsed_oneshot:.1f}x slower)")
        
        if output_autoreg.shape == (batch_size, T_out, n_vars):
            print("  [OK] Autoregressive inference passed!")
        else:
            print("  [ERROR] Shape mismatch!")
    
    # Compare outputs
    print(f"\n{'='*60}")
    print("Comparing outputs")
    print('='*60)
    print(f"\n  Mean absolute difference: {torch.abs(output_oneshot - output_autoreg).mean():.6f}")
    print(f"  Max absolute difference: {torch.abs(output_oneshot - output_autoreg).max():.6f}")
    print(f"  One-shot output stats: mean={output_oneshot.mean():.4f}, std={output_oneshot.std():.4f}")
    print(f"  Autoregressive output stats: mean={output_autoreg.mean():.4f}, std={output_autoreg.std():.4f}")
    
    print(f"\n{'='*60}")
    print("[OK] All tests passed!")
    print('='*60)