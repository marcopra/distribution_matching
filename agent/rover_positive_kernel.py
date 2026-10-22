"""ROVER with probability-preserving landmark kernel transitions."""

import math

import torch

from agent.rover_nystrom_subspace import RoverSubspaceCoverageAgent


def positive_kernel_logits(H, E, coeff, prior_mass):
    """Query replay and sink coefficients in the normalized action basis."""
    assert H.ndim == E.ndim == coeff.ndim == 2
    assert H.shape[1] == E.shape[0] == coeff.shape[0] - 1
    assert coeff.shape[1] == 1 and math.isfinite(prior_mass) and prior_mass > 0
    logits = (H @ (coeff[:-1] * E) + prior_mass * coeff[-1]) / (H @ E + prior_mass)
    assert torch.isfinite(logits).all(), "Nonfinite positive-kernel policy logits"
    return logits


class PositiveKernelTransition:
    """Aggregate kernel-weighted replay next states, with an absorbing sink."""

    def __init__(self, H, E, next_landmark, coverage_gram, alpha, gamma, prior_mass):
        n, m = H.shape
        assert n > 0 and m > 0
        assert 0 < gamma < 1 and math.isfinite(prior_mass) and prior_mass > 0
        assert E.ndim == 2 and E.shape[0] == n
        assert next_landmark.shape == (n,) and next_landmark.dtype == torch.long
        assert 0 <= next_landmark.min() and next_landmark.max() < m
        assert coverage_gram.shape == (m + 1, m + 1) and alpha.shape == (m, 1)
        assert torch.isfinite(H).all() and H.min() >= 0
        assert ((E == 0) | (E == 1)).all() and (E.sum(1) == 1).all()
        assert torch.isfinite(coverage_gram).all()
        assert torch.allclose(coverage_gram, coverage_gram.T)
        assert (alpha >= 0).all() and torch.allclose(alpha.sum(), alpha.new_tensor(1.))
        self.H, self.E, self.next_landmark = H, E, next_landmark
        self.gram, self.gamma, self.prior_mass = coverage_gram, gamma, prior_mass
        self.denominator = H.T @ E + prior_mass
        self.alpha = torch.cat([alpha, alpha.new_zeros(1, 1)])
        self.tolerance = max(1e-10, 100 * torch.finfo(H.dtype).eps)

    def policy(self, coeff, logits_offset=None):
        logits = positive_kernel_logits(self.H.T, self.E, coeff, self.prior_mass)
        if logits_offset is not None:
            assert logits_offset.shape == logits.shape
            logits = logits + logits_offset
        return torch.softmax(-logits, dim=1)

    def transition_matrix(self, pi):
        m = self.H.shape[1]
        assert pi.shape == (m, self.E.shape[1])
        assert torch.isfinite(pi).all() and pi.min() >= 0
        assert torch.allclose(pi.sum(1), torch.ones_like(pi[:, 0]),
                              atol=self.tolerance, rtol=self.tolerance)
        weights = self.E @ (pi / self.denominator).T
        weights.mul_(self.H)
        P = self.H.new_zeros(m + 1, m + 1)
        P[:-1, :-1].index_add_(0, self.next_landmark, weights)
        P[-1, :-1] = (pi * (self.prior_mass / self.denominator)).sum(1)
        P[-1, -1] = 1
        assert torch.isfinite(P).all() and P.min() >= 0
        assert torch.allclose(P.sum(0), torch.ones_like(P[0]),
                              atol=self.tolerance, rtol=self.tolerance)
        return P

    def _resolvent(self, pi):
        P = self.transition_matrix(pi)
        S = torch.eye(len(P), device=P.device, dtype=P.dtype) - self.gamma * P
        z = torch.linalg.solve(S, self.alpha)
        occupancy = (1 - self.gamma) * z
        assert torch.isfinite(occupancy).all() and occupancy.min() >= -self.tolerance
        assert torch.allclose(occupancy.sum(), occupancy.new_tensor(1.),
                              atol=self.tolerance, rtol=self.tolerance)
        return z, S

    def loss(self, pi):
        z, _ = self._resolvent(pi)
        return ((1 - self.gamma) ** 2 * (z.T @ self.gram @ z)).squeeze()

    def gradient_coefficients(self, pi):
        z, S = self._resolvent(pi)
        value = torch.linalg.solve(S.T, self.gram @ z)
        coeff = torch.cat([value[:-1][self.next_landmark], value[-1:]])
        return 2 * self.gamma * (1 - self.gamma) ** 2 * coeff

    @torch.no_grad()
    def diagnostics(self, pi):
        """Return exact 1-norm conditioning and real-state transition statistics."""
        P = self.transition_matrix(pi)
        real = P[:-1, :-1]
        negative = real < 0
        diagonal_min = P.diagonal().min()
        # For column-stochastic P, ||(I-gamma P)^-1||_1 is 1/(1-gamma),
        # and column j of I-gamma P has norm 1+gamma-2*gamma*P[j,j].
        condition_1 = (1 + self.gamma - 2 * self.gamma * diagonal_min) / (1 - self.gamma)
        stats = torch.stack((
            condition_1,
            negative.to(P.dtype).mean(),
            real.min(),
            -real.clamp_max(0).sum(),
            real.sum(0).min(),
            real.sum(0).max(),
        )).detach().cpu().tolist()
        return dict(zip((
            "resolvent_condition_1_estimate",
            "negative_real_transition_fraction",
            "negative_real_transition_min",
            "negative_real_transition_l1",
            "real_transition_column_mass_min",
            "real_transition_column_mass_max",
        ), map(float, stats)))


class RoverPositiveKernelAgent(RoverSubspaceCoverageAgent):
    """Use the shared PMD loop with a positive normalized transition estimator."""

    def __init__(self, *args, transition_prior_mass=1., **kwargs):
        super().__init__(*args, **kwargs)
        assert self.kernel_type == "gaussian", "Positive transitions require a Gaussian dynamics kernel"
        assert math.isfinite(transition_prior_mass) and transition_prior_mass > 0
        self.transition_prior_mass = float(transition_prior_mass)

    def _policy_logits_from_H(self, H, coeff=None):
        coeff = self.gradient_coeff if coeff is None else coeff
        if coeff is None:
            return super()._policy_logits_from_H(H, coeff=None)
        return positive_kernel_logits(H, self.E, coeff, self.transition_prior_mass)

    def _prepare_actor_model(self, sink_norm):
        H = self._kernel(self._phi_all_obs, self._phi_sub_next)
        next_states, landmarks = self._phi_all_next, self._phi_sub_next
        if self.dynamics_kernel_input == "states" and not self.uses_state_dynamics_encoder:
            next_states = self.kernel_fn._decode(next_states)
            landmarks = self.kernel_fn._decode(landmarks)
        nearest = self._pmd_nearest_rows(next_states, landmarks)
        del self._phi_all_next
        m = H.shape[1]
        gram = H.new_zeros(m + 1, m + 1)
        features = self._coverage_sub_next_features
        assert features is not None and len(features) == m
        assert math.isfinite(float(sink_norm)) and sink_norm >= 0
        gram[:-1, :-1] = self.coverage_kernel_fn(features, features)
        gram[-1, -1] = float(sink_norm) ** 2
        return PositiveKernelTransition(
            H, self.E, nearest, gram, self._sub_alpha, self.discount, self.transition_prior_mass,
        )

    @torch.no_grad()
    def _cache_encoded_features(self, encoded_full, encoded_sub=None):
        super()._cache_encoded_features(encoded_full, encoded_sub)
        self._phi_all_next = self._append_zero_feature_column(
            encoded_full["phi_next"].to(device=self.device, dtype=self.compute_dtype)
        )

    @torch.no_grad()
    def _cache_features(self, obs, action, next_obs, encoder=None, **kwargs):
        super()._cache_features(obs, action, next_obs, encoder=encoder, **kwargs)
        encoder = self.encoder if encoder is None else encoder
        self._phi_all_next = self._append_zero_feature_column(
            self._encode_with_module(encoder, next_obs, project=True).to(dtype=self.compute_dtype)
        )

    def _debug_visualizer_text(self, step):
        return (super()._debug_visualizer_text(step)
                + f"transition estimator = positive kernel, prior mass = {self.transition_prior_mass}\n")
