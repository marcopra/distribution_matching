from __future__ import annotations

from typing import Optional

import torch

import utils
from agent.rover_sink import (
    nystrom_resolvent_coefficients,
    nystrom_resolvent_diagnostics,
    orthogonal_sink_residual_gram,
)

class DistributionMatcher:
    """Handles mathematical operations for distribution matching via PMD."""

    def __init__(self, 
                 lambda_reg: float,
                 gamma: float = 0.9, 
                 pca_truncation: Optional[int] = None,
                 svd_truncation: Optional[int] = None,
                 kernel_type: str = "inner_product",
                 kernel_bandwidth: Optional[float] = None,
                 device: str = "cpu"):
        
        self.gamma = gamma
        self.lambda_reg = lambda_reg
        self.device = device  
        self.pca_truncation = pca_truncation if pca_truncation is not None else svd_truncation
        self.kernel_type = kernel_type
        self.kernel_bandwidth = kernel_bandwidth
        self.kernel_fn = utils.build_kernel_fn(
            kernel_type,
            bandwidth=self.kernel_bandwidth,
        )
        self.state_kernel_fn = None

    def kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return self.kernel_fn(X, Y)

    def state_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        kernel_fn = self.state_kernel_fn if self.state_kernel_fn is not None else self.kernel_fn
        return kernel_fn(X, Y)

    def state_action_kernel(
            self,
            X_states: torch.Tensor,
            Y_states: torch.Tensor,
            X_actions: torch.Tensor,
            Y_actions: torch.Tensor
        ) -> torch.Tensor:
        """Compute k((s,a),(s',a')) = k_s(s,s') 1[a = a'] for discrete actions."""
        K = self.kernel_fn(X_states, Y_states)
        if X_actions.ndim > 1 and Y_actions.ndim > 1:
            X_actions = X_actions.to(device=K.device, dtype=K.dtype).reshape(K.shape[0], -1)
            Y_actions = Y_actions.to(device=K.device, dtype=K.dtype).reshape(K.shape[1], -1)
            action_mask = X_actions @ Y_actions.T
        else:
            if X_actions.ndim > 1:
                X_actions = torch.argmax(X_actions, dim=-1)
            if Y_actions.ndim > 1:
                Y_actions = torch.argmax(Y_actions, dim=-1)
            X_actions = X_actions.to(device=K.device).reshape(-1)
            Y_actions = Y_actions.to(device=K.device).reshape(-1)
            action_mask = X_actions[:, None] == Y_actions[None, :]
        return K.mul_(action_mask)

    
            
    def compute_nu_pi(
            self, 
            phi_all_next_obs: torch.Tensor, 
            psi_all_obs_action: torch.Tensor,
            K: torch.Tensor,
            M: torch.Tensor,
            alpha: torch.Tensor,
            sink_norm: float
        ) -> torch.Tensor:
        """Compute discounted occupancy: ν = (1-γ)Φᵀ(I - γBM)⁻¹α."""

        N = K.shape[0]
       
        # α̃ augmented to be [α; 1]
        tilde_alpha = torch.ones((alpha.shape[0] + 1, 1), device=alpha.device, dtype=alpha.dtype)
        tilde_alpha[:-1] = alpha

        # ** COMPUTATION STEP **
        # Compute Cholesky decomposition and solve: B̃M̃ = Ã⁻¹M̃
        A = K + self.lambda_reg * torch.eye(N, device=K.device, dtype=K.dtype)
        # L = torch.linalg.cholesky(A)
        # BM = torch.cholesky_solve(M, L)

        BM = torch.linalg.solve(A, M) # [n, n]
        
        # M̃ augmented to be [M 0; 0 1]
        tilde_BM = torch.zeros(BM.shape[0] + 1, BM.shape[1] + 1, device=BM.device, dtype=BM.dtype)
        tilde_BM[:-1, :-1] = BM
        tilde_BM[-1, -1] = 1.0

        inv_term = torch.linalg.solve(torch.eye(N + 1, device=tilde_BM.device, dtype=tilde_BM.dtype) - self.gamma * tilde_BM, tilde_alpha)
        
        sink_state = torch.zeros((phi_all_next_obs.shape[1],1), device=self.device, dtype=phi_all_next_obs.dtype)
        sink_state[-1] = sink_norm

        # Computing Ψ̃ and Φ̃ are now of shape [N+1, d*|A| + 2] and [N+1, d + 2] respectively
        upper_left = phi_all_next_obs.T - sink_state@torch.ones((1, psi_all_obs_action.shape[1]), device=psi_all_obs_action.device, dtype=psi_all_obs_action.dtype)@psi_all_obs_action.T
        tilde_phi_all_next_obs_transposed = torch.zeros((phi_all_next_obs.shape[1]+1, phi_all_next_obs.shape[0]+1), device=phi_all_next_obs.device, dtype=phi_all_next_obs.dtype)
        tilde_phi_all_next_obs_transposed[:upper_left.shape[0], :upper_left.shape[1]] = upper_left
        tilde_phi_all_next_obs_transposed[:sink_state.shape[0], -1:] = sink_state
        # tilde_phi_all_next_obs_transposed[-1, -1] = 1.0 # TODO patch 0.1

        occupancy = (1 - self.gamma) *  tilde_phi_all_next_obs_transposed @ inv_term
        # print(f"Occupancy sum: {occupancy.sum().item()} and occupancy of sink state: {occupancy[-1].item()}")
        return occupancy

    def compute_gradient_coefficient(
            self, 
            M: torch.Tensor, 
            phi_all_next_obs:torch.Tensor, 
            psi_all_obs_action:torch.Tensor, 
            alpha:torch.Tensor,
            sink_norm: float,
            K: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """Compute gradient coefficient for policy update."""
        # Identity matrix
        I_n_plus1 = torch.eye(psi_all_obs_action.shape[0], device=psi_all_obs_action.device, dtype=psi_all_obs_action.dtype)

        # Ã augmented to be [A 0; 0 1]
        # Symmetric positive definite matrix A = ψψᵀ + λI
        K = self.kernel(psi_all_obs_action, psi_all_obs_action) if K is None else K
        A = K + self.lambda_reg * I_n_plus1
        tilde_A = torch.zeros(A.shape[0] + 1, A.shape[1] + 1, device=A.device, dtype=A.dtype)
        tilde_A[:-1, :-1] = A
        tilde_A[-1, -1] = 1.0

        # M̃ augmented to be [M 0; 0 1]
        tilde_M = torch.zeros(M.shape[0] + 1, M.shape[1] + 1, device=M.device, dtype=M.dtype)
        tilde_M[:-1, :-1] = M
        tilde_M[-1, -1] = 1.0

        # α̃ augmented to be [α; 1]
        tilde_alpha = torch.ones((alpha.shape[0] + 1, 1), device=alpha.device, dtype=alpha.dtype)
        tilde_alpha[:-1] = alpha

        # ** COMPUTATION STEP **
        # Compute Cholesky decomposition and solve: BM = A⁻¹M
        # L = torch.linalg.cholesky(A)
        # BM = torch.cholesky_solve(M, L)
        BM = torch.linalg.solve(A, M) # [n, n]
        tilde_B_tilde_M = torch.zeros(BM.shape[0] + 1, BM.shape[1] + 1, device=BM.device, dtype=BM.dtype)
        tilde_B_tilde_M[:-1, :-1] = BM
        tilde_B_tilde_M[-1, -1] = 1.0

        # gradient = 2 γ (1 - γ)² Ã⁻ᵀ (I - γ Ã⁻¹M̃)⁻ᵀΦ̃Φ̃ᵀ(I - γ Ã⁻¹M̃)⁻¹ α̃ 
        state_features = phi_all_next_obs[:, :-1]
        state_gram = self.state_kernel(state_features, state_features)
        phi_kernel = orthogonal_sink_residual_gram(
            state_gram, psi_all_obs_action.sum(dim=1), sink_norm
        )
        I_n_plus1 = torch.eye(tilde_B_tilde_M.shape[0], device=tilde_B_tilde_M.device, dtype=tilde_B_tilde_M.dtype)
        # Left term: Ã⁻ᵀ(I - γB̃M̃)⁻ᵀΦ̃Φ̃ᵀ
        left_term = torch.linalg.solve((I_n_plus1 - self.gamma * tilde_B_tilde_M).T@tilde_A.T, phi_kernel) # [n+1, n+1]

        # (I - γ Ã⁻¹M̃)⁻ᵀΦ̃Φ̃ᵀ
        # inv_term_kernel = torch.linalg.solve((I_n_plus1 - self.gamma * tilde_B_tilde_M).T, phi_kernel) # [n+1, n+1]
        # Solve Ãᵀ x = left_term_without_b using Cholesky
        # L_T = torch.linalg.cholesky(tilde_A.T)
        # left_term = torch.cholesky_solve(inv_term_kernel, L_T)

        # right term: (I - γ Ã⁻¹M̃)⁻¹ α̃
        right_term = torch.linalg.solve((I_n_plus1 - self.gamma * tilde_B_tilde_M), tilde_alpha)

        # gradient = 2 γ (1 - γ)² Ã⁻ᵀ (I - γ Ã⁻¹M̃)⁻ᵀΦ̃Φ̃ᵀ(I - γ Ã⁻¹M̃)⁻¹ α̃
        gradient = 2 * self.gamma * ((1 - self.gamma) ** 2) * left_term @ right_term
      
        return gradient
    


class NystromActorModel:
    """Action-block regression with cached transitions and projected gradients."""

    @torch.no_grad()
    def __init__(self, matcher, states, landmarks, next_landmarks, actions,
                 landmark_actions, E, alpha, coverage_gram, components, block_size=1024):
        n, d = states.shape
        m = len(landmarks)
        assert n > 0 and m > 0
        assert isinstance(block_size, int) and block_size > 0
        assert landmarks.shape == next_landmarks.shape == (m, d)
        assert E.ndim == 2 and E.shape[0] == n and E.shape[1] > 0
        assert alpha.shape == (m, 1) and coverage_gram.shape == (m + 1, m + 1)
        for tensor in (landmarks, next_landmarks, E, alpha, coverage_gram):
            assert tensor.dtype == states.dtype and tensor.device == states.device
            assert torch.isfinite(tensor).all()
        assert torch.isfinite(states).all()
        assert torch.isfinite(states.new_tensor(matcher.lambda_reg)) and matcher.lambda_reg > 0
        assert 0 < matcher.gamma < 1
        assert components is None or (isinstance(components, int) and components > 0)
        assert actions.dtype == landmark_actions.dtype == torch.long
        assert (alpha >= 0).all() and torch.allclose(alpha.sum(), alpha.new_tensor(1.))
        self.actions = actions.to(device=states.device, dtype=torch.long).reshape(-1)
        self.landmark_actions = landmark_actions.to(device=states.device, dtype=torch.long).reshape(-1)
        assert self.actions.shape == (n,) and self.landmark_actions.shape == (m,)
        for ids in (self.actions, self.landmark_actions):
            assert ids.min() >= 0 and ids.max() < E.shape[1]
        assert torch.equal(E.argmax(1), self.actions)
        assert ((E == 0) | (E == 1)).all() and (E.sum(1) == 1).all()
        self.matcher, self.states, self.landmarks = matcher, states, landmarks
        self.next_landmarks, self.E, self.alpha = next_landmarks, E, alpha
        self.gram, self.block_size = coverage_gram, int(block_size)

        rank = m if components is None else min(m, int(components))
        self.blocks, eigenspaces = [], []
        for action in self.landmark_actions.unique(sorted=True).tolist():
            rows = torch.nonzero(self.actions == action, as_tuple=True)[0]
            columns = torch.nonzero(self.landmark_actions == action, as_tuple=True)[0]
            support = landmarks[columns]
            A = states.new_empty(len(columns), len(columns))
            for start in range(0, len(columns), self.block_size):
                section = slice(start, start + self.block_size)
                A[section] = matcher.kernel(support[section], support)
            # Same jitter and CPU float32 eigensolver; retain global top-r below.
            K_cpu = A.to(device="cpu", copy=True)
            K_cpu.diagonal().add_(1e-8)
            values, vectors = torch.linalg.eigh(K_cpu.float())
            keep = min(rank, len(columns))
            eigenspaces.append((values[-keep:].clone(), vectors[:, -keep:].clone()))
            del K_cpu, values, vectors
            A.mul_(matcher.lambda_reg)
            C = states.new_zeros(len(columns), m)
            for start in range(0, len(rows), self.block_size):
                batch = states[rows[start:start + self.block_size]]
                K = matcher.kernel(batch, support)
                H = matcher.state_kernel(batch, next_landmarks)
                A.addmm_(K.T, K)
                C.addmm_(K.T, H)
                del K, H
            A.diagonal().add_(1e-6)
            lu, pivots = torch.linalg.lu_factor(A)
            # For row action a, T(pi) = solve(A_a, K_a.T @ H_a) diag(pi[:, a]).
            self.blocks.append(dict(action=action, rows=rows, columns=columns, lu=lu, pivots=pivots,
                                    transition=torch.linalg.lu_solve(lu, pivots, C)))
            del A, C, support

        values = torch.cat([values for values, _ in eigenspaces])
        selected = torch.argsort(values, stable=True)[-rank:]
        self.U = states.new_zeros(m, rank)
        offset = 0
        for block, (values, vectors) in zip(self.blocks, eigenspaces):
            basis = torch.nonzero((selected >= offset) & (selected < offset + len(values)), as_tuple=True)[0]
            local_U = vectors[:, selected[basis] - offset].to(device=states.device, dtype=states.dtype)
            basis = basis.to(states.device)
            self.U[block["columns"][:, None], basis] = local_U
            block["basis"] = basis
            block["adjoint_projection"] = torch.linalg.lu_solve(block["lu"], block["pivots"], local_U, adjoint=True)
            block["projected_transition"] = local_U.T @ block["transition"]
            offset += len(values)
        # Float32 eigenvectors are only approximately orthonormal.
        self.U_gram = self.U.T @ self.U
        self.K_rr = self.U.T @ self.gram[:-1, :-1] @ self.U
        self.k_re = self.U.T @ self.gram[:-1, -1:]
        self.k_er = self.gram[-1:, :-1] @ self.U
        self.alpha_r = self.U.T @ alpha

    def _rows(self):
        for start in range(0, len(self.states), self.block_size):
            yield slice(start, start + self.block_size)

    def _solve(self, rhs, adjoint=False):
        assert rhs.ndim == 2 and rhs.shape[0] == len(self.landmarks)
        assert rhs.dtype == self.states.dtype and rhs.device == self.states.device
        result = torch.empty_like(rhs)
        for block in self.blocks:
            columns = block["columns"]
            result[columns] = torch.linalg.lu_solve(block["lu"], block["pivots"], rhs[columns], adjoint=adjoint)
        return result

    def _apply_kernel(self, coefficients):
        result = self.states.new_zeros(len(self.states), coefficients.shape[1])
        for block in self.blocks:
            rows, columns = block["rows"], block["columns"]
            support, coeff = self.landmarks[columns], coefficients[columns]
            for start in range(0, len(rows), self.block_size):
                batch = rows[start:start + self.block_size]
                result[batch] = self.matcher.kernel(self.states[batch], support) @ coeff
        return result

    @torch.no_grad()
    def policy(self, coeff, logits_offset=None):
        assert coeff.shape == (len(self.states) + 1, 1)
        logits = self.states.new_zeros(len(self.landmarks), self.E.shape[1])
        for rows in self._rows():
            H = self.matcher.state_kernel(self.states[rows], self.next_landmarks)
            logits.addmm_(H.T, coeff[:-1][rows] * self.E[rows])
        logits.add_(coeff[-1])
        if logits_offset is not None:
            assert logits_offset.shape == logits.shape
            logits.add_(logits_offset)
        assert torch.isfinite(logits).all(), "Nonfinite Nyström policy logits"
        return torch.softmax(-logits, dim=1)

    def _validate_policy(self, pi):
        assert pi.shape == (len(self.landmarks), self.E.shape[1])
        assert pi.dtype == self.states.dtype and pi.device == self.states.device
        assert torch.isfinite(pi).all() and (pi >= 0).all()
        assert torch.allclose(pi.sum(1), torch.ones_like(pi[:, 0]))

    @torch.no_grad()
    def transition_matrix(self, pi):
        self._validate_policy(pi)
        transition = self.states.new_empty(len(self.landmarks), len(self.landmarks))
        for block in self.blocks:
            for start in range(0, len(block["columns"]), self.block_size):
                section = slice(start, start + self.block_size)
                transition[block["columns"][section]] = block["transition"][section] * pi[:, block["action"]]
        return transition

    @torch.no_grad()
    def adjoint(self, q):
        return self._apply_kernel(self._solve(q, adjoint=True))

    @torch.no_grad()
    def loss(self, pi):
        z = nystrom_resolvent_coefficients(self.transition_matrix(pi), self.alpha, self.matcher.gamma)
        return ((1 - self.matcher.gamma) ** 2 * (z.T @ self.gram @ z)).squeeze()

    @torch.no_grad()
    def gradient_coefficients(self, pi):
        gamma, eps = self.matcher.gamma, 1e-6
        self._validate_policy(pi)
        S_r = self.U_gram.clone()
        for block in self.blocks:
            weighted_U = pi[:, block["action"]:block["action"] + 1] * self.U
            S_r[block["basis"]] -= gamma * (block["projected_transition"] @ weighted_U)
        del weighted_U
        z_r = torch.linalg.solve(S_r, self.alpha_r)
        S_r.diagonal().add_(eps)
        z_e = 1 / (1 - gamma)
        h_r = self.K_rr @ z_r + self.k_re * z_e
        h_e = self.k_er @ z_r + self.gram[-1:, -1:] * z_e
        tmp = torch.linalg.solve(S_r.T, h_r)
        solved = self.states.new_empty(len(self.landmarks), 1)
        for block in self.blocks:
            solved[block["columns"]] = block["adjoint_projection"] @ tmp[block["basis"]]
        gradient = self.states.new_empty(len(self.states) + 1, 1)
        gradient[:-1] = self._apply_kernel(solved)
        gradient[-1:] = h_e / (1 - gamma + eps)
        gradient.mul_(2 * gamma * (1 - gamma) ** 2)
        assert torch.isfinite(gradient).all(), "Nonfinite Nyström gradient"
        return gradient

    @torch.no_grad()
    def diagnostics(self, pi):
        return nystrom_resolvent_diagnostics(self.transition_matrix(pi), self.matcher.gamma)
