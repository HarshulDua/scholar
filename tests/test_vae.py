"""Tests for Mult-VAE model, loss, and reparameterization."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from scholar.recsys.mult_vae import MultVAE, MultVAELoss


def _make_model(vocab_size: int = 50, latent_dim: int = 16, hidden_dim: int = 64) -> MultVAE:
    return MultVAE(vocab_size=vocab_size, latent_dim=latent_dim, hidden_dim=hidden_dim)


class TestMultVAEForward:
    def test_output_shapes(self):
        model = _make_model(vocab_size=50, latent_dim=16)
        x = torch.rand(4, 50)
        recon, mu, logvar = model(x)
        assert recon.shape == (4, 50)
        assert mu.shape == (4, 16)
        assert logvar.shape == (4, 16)

    def test_recon_is_log_softmax(self):
        model = _make_model()
        x = torch.rand(2, 50)
        recon, _, _ = model(x)
        # log_softmax outputs should be <= 0 and sum to 0 in log domain
        assert (recon <= 0).all()
        exp_sum = recon.exp().sum(dim=-1)
        assert torch.allclose(exp_sum, torch.ones(2), atol=1e-4)

    def test_batch_size_one(self):
        model = _make_model()
        x = torch.rand(1, 50)
        recon, mu, logvar = model(x)
        assert recon.shape == (1, 50)

    def test_encode_output_shape(self):
        model = _make_model(vocab_size=30, latent_dim=8)
        x = torch.rand(3, 30)
        mu, logvar = model.encode(x)
        assert mu.shape == (3, 8)
        assert logvar.shape == (3, 8)

    def test_decode_output_shape(self):
        model = _make_model(vocab_size=30, latent_dim=8)
        z = torch.rand(5, 8)
        out = model.decode(z)
        assert out.shape == (5, 30)


class TestReparameterize:
    def test_deterministic_at_eval(self):
        model = _make_model()
        model.eval()
        mu = torch.randn(2, 16)
        logvar = torch.randn(2, 16)
        z1 = model.reparameterize(mu, logvar)
        z2 = model.reparameterize(mu, logvar)
        assert torch.equal(z1, mu)
        assert torch.equal(z2, mu)

    def test_stochastic_at_train(self):
        model = _make_model()
        model.train()
        mu = torch.randn(10, 16)
        logvar = torch.zeros(10, 16)
        z1 = model.reparameterize(mu, logvar)
        z2 = model.reparameterize(mu, logvar)
        # With non-zero std, samples should differ
        assert not torch.equal(z1, z2)

    def test_eval_returns_mu(self):
        model = _make_model()
        model.eval()
        mu = torch.tensor([[1.0, 2.0, 3.0]])
        logvar = torch.zeros(1, 3)
        z = model.reparameterize(mu, logvar)
        assert torch.allclose(z, mu)


class TestMultVAELoss:
    def test_output_is_scalar(self):
        model = _make_model()
        x = torch.rand(4, 50)
        recon, mu, logvar = model(x)
        criterion = MultVAELoss(beta=0.2)
        loss = criterion(recon, x, mu, logvar)
        assert loss.shape == ()

    def test_loss_is_finite(self):
        model = _make_model()
        x = torch.rand(4, 50)
        recon, mu, logvar = model(x)
        criterion = MultVAELoss(beta=0.1)
        loss = criterion(recon, x, mu, logvar)
        assert torch.isfinite(loss)

    def test_beta_zero_disables_kl(self):
        model = _make_model()
        model.eval()
        x = torch.rand(2, 50)
        recon, mu, logvar = model(x)

        loss_full = MultVAELoss(beta=1.0)(recon, x, mu, logvar)
        loss_no_kl = MultVAELoss(beta=0.0)(recon, x, mu, logvar)
        # With beta=0, only reconstruction loss — should be different from beta=1
        assert loss_full != loss_no_kl

    def test_kl_annealing_schedule(self):
        epochs = list(range(1, 25))
        betas = [min(0.2, e / 20 * 0.2) for e in epochs]
        assert betas[0] == pytest.approx(0.01)
        assert betas[19] == pytest.approx(0.2)
        assert betas[20] == pytest.approx(0.2)
        assert all(0 <= b <= 0.2 for b in betas)
