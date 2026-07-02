from __future__ import annotations

from typing import Optional

import torch

import utils
import matplotlib.pyplot as plt
 

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
        return K * action_mask.to(dtype=K.dtype)

    
            
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
        A = K + self.lambda_reg * torch.eye(N, device=self.device)
        # L = torch.linalg.cholesky(A)
        # BM = torch.cholesky_solve(M, L)

        BM = torch.linalg.solve(A, M) # [n, n]
        
        # M̃ augmented to be [M 0; 0 1]
        tilde_BM = torch.zeros(BM.shape[0] + 1, BM.shape[1] + 1, device=BM.device, dtype=BM.dtype)
        tilde_BM[:-1, :-1] = BM
        tilde_BM[-1, -1] = 1.0

        inv_term = torch.linalg.solve( torch.eye(N+1, device=self.device) - self.gamma * tilde_BM, tilde_alpha)
        
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
        I_n_plus1 = torch.eye(psi_all_obs_action.shape[0], device=self.device)

        sink_state = torch.zeros((phi_all_next_obs.shape[1],1), device=self.device, dtype=phi_all_next_obs.dtype)
        sink_state[-1] = sink_norm

        # Computing Ψ̃ and Φ̃ are now of shape [N+1, d*|A| + 2] and [N+1, d + 2] respectively
        upper_left = phi_all_next_obs.T - sink_state@torch.ones((1, psi_all_obs_action.shape[1]), device=psi_all_obs_action.device, dtype=psi_all_obs_action.dtype)@psi_all_obs_action.T
        tilde_phi_all_next_obs_transposed = torch.zeros((phi_all_next_obs.shape[1]+1, phi_all_next_obs.shape[0]+1), device=phi_all_next_obs.device, dtype=phi_all_next_obs.dtype)
        tilde_phi_all_next_obs_transposed[:upper_left.shape[0], :upper_left.shape[1]] = upper_left
        assert sink_state.shape[0] == upper_left.shape[0], "Sink state and upper left matrix row size mismatch"
        tilde_phi_all_next_obs_transposed[:sink_state.shape[0], -1:] = sink_state
        tilde_phi_all_next_obs = tilde_phi_all_next_obs_transposed.T
        assert torch.all(tilde_phi_all_next_obs_transposed[:sink_state.shape[0], -1:] == sink_state), "Last column of tilde_phi_all_next_obs should be sink_state"

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
        # State and state-action similarities can use different Gaussian bandwidths.
        phi_kernel = self.state_kernel(tilde_phi_all_next_obs, tilde_phi_all_next_obs) # [n+1, n+1]
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
    
    #******* NYSTROM********
    def _regularized_solve_memory_efficient(
            self,
            A: torch.Tensor,
            B: torch.Tensor,
            jitter_scale: float = 1e-10,
        ) -> torch.Tensor:
        """Solve AX=B robustly when A is singular/ill-conditioned."""
        

        idx = torch.arange(A.shape[0], device=A.device)
        A[idx, idx] += jitter_scale

        X = torch.linalg.solve(A, B)
        
        return X 
    
    def compute_B_nystrom(
            self,
            psi_all_obs_action: torch.Tensor,
            psi_sub_obs_action: torch.Tensor,
            svd_truncation: Optional[int] = None,
            phi_all_obs: Optional[torch.Tensor] = None,
            phi_sub_obs: Optional[torch.Tensor] = None,
            all_actions: Optional[torch.Tensor] = None,
            sub_actions: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        N = psi_all_obs_action.shape[0]
        if phi_all_obs is not None and phi_sub_obs is not None and all_actions is not None and sub_actions is not None:
            K_nm = self.state_action_kernel(phi_all_obs, phi_sub_obs, all_actions, sub_actions)
            K_mm = self.state_action_kernel(phi_sub_obs, phi_sub_obs, sub_actions, sub_actions)
        else:
            K_nm = self.kernel(psi_all_obs_action, psi_sub_obs_action) # [n, m]
            K_mm = self.kernel(psi_sub_obs_action, psi_sub_obs_action) # [m, m]
        A_nystrom = K_nm.T@K_nm + self.lambda_reg * K_mm # [m, m] # TODO add N
        inv_A_nystrom = self.pseudo_inverse_low_rank_svd(A_nystrom, svd_rank=svd_truncation)
        return inv_A_nystrom@K_nm.T
    
    def compute_B_and_projections(
            self,
            K_nm: torch.Tensor,
            K_mm: torch.Tensor,
            components: Optional[int] = 10000,
        ) -> tuple[torch.Tensor, torch.Tensor]:
      
        m = K_nm.shape[1]

        # A_nystrom = K_nm.T @ K_nm
        # A_nystrom.add_(self.lambda_reg * K_mm)
        # A_nystrom.diagonal().add_(1e-6)

        A_nystrom = K_nm.T@K_nm + self.lambda_reg * K_mm + 1e-6 * torch.eye(m, dtype=torch.float64, device=self.device)# [m, m] # TODO add N
        B = torch.linalg.solve(A_nystrom, K_nm.T) # [m, n]
        effective_components = m if components is None else min(m, int(components))
        
        # K = K_mm.clone()
        # K.diagonal().add_(1e-8)
        
        eig_vals_r, eig_vecs_r = torch.linalg.eigh(
            K_mm + 1e-8 * torch.eye(m, dtype=K_mm.dtype, device=K_mm.device)
        )

        eig_vals_r, eig_vecs_r = torch.linalg.eigh(K_mm)
        U_r = eig_vecs_r[:, -effective_components:]

        return B, U_r

    def pseudo_inverse_low_rank_svd(self, A, tol=1e-12, svd_rank=None):
        assert svd_rank is not None, "svd_rank must be specified for low-rank pseudo-inverse"
        if svd_rank is None:
            U, S, Vh = torch.linalg.svd(A, full_matrices=False)
            # Inverti solo i valori singolari non nulli
            V= Vh.transpose(-2, -1) 

        else:
            U, S, V = torch.svd_lowrank(A, q=svd_rank)
        S_inv = torch.where(S > tol, 1.0 / S, torch.zeros_like(S))
    
        S_inv_mat = torch.diag(S_inv)
        
        A_pinv = V @ S_inv_mat @ U.transpose(-2, -1)

        #save plot of singular values
        plt.figure()
        plt.plot(S.cpu().numpy(), marker='o')
        plt.yscale('log')
        plt.title('Singular Values of A')
        plt.xlabel('Index')
        plt.ylabel('Singular Value (log scale)')
        plt.grid()
        plt.savefig('singular_values.png')
        
        
        return A_pinv

    def compute_nu_pi_nystrom_memory_efficient(
            self,
            phi_all_obs: torch.Tensor,
            phi_sub_next_obs: torch.Tensor,
            psi_sub_obs_action: torch.Tensor,
            psi_all_obs_action: torch.Tensor,
            H: torch.Tensor,
            pi: torch.Tensor,
            E: torch.Tensor,
            alpha: torch.Tensor,
            sink_norm: float,
            B_nystrom: Optional[torch.Tensor] = None,
            phi_sub_obs: Optional[torch.Tensor] = None,
            all_actions: Optional[torch.Tensor] = None,
            sub_actions: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """Compute discounted occupancy: ν = (1-γ)Φᵀ(I - γBM)⁻¹α."""
        N = psi_all_obs_action.shape[0]

        m = psi_sub_obs_action.shape[0]

        d = phi_sub_next_obs.shape[1]

        # α̃ = [α; 1], but avoid torch.ones
        alpha_tilde = torch.empty(
            (alpha.shape[0] + 1, 1),
            device=alpha.device,
            dtype=alpha.dtype,
        )
        alpha_tilde[:-1] = alpha
        alpha_tilde[-1] = 1.0

        # H = phi_all_obs @ phi_sub_next_obs.T # [n, m] 
        M = H*(E@pi.T) # [n, m]

        if B_nystrom is not None:
            BM = B_nystrom @ M
        else:
            # Nyström matrices
            B = self.compute_B_nystrom(
                psi_all_obs_action,
                psi_sub_obs_action,
                svd_truncation=self.pca_truncation,
                phi_all_obs=phi_all_obs,
                phi_sub_obs=phi_sub_next_obs if phi_sub_obs is None else phi_sub_obs,
                all_actions=all_actions,
                sub_actions=sub_actions,
            )
            BM = B @ M


        # Build S = I - gamma * tilde_BM directly
        # tilde_BM = [BM 0]
        #            [0  1]

        S = torch.empty(
            (BM.shape[0] + 1, BM.shape[1] + 1),
            device=BM.device,
            dtype=BM.dtype,

        )

        S[:-1, :-1] = BM
        S[:-1, :-1].mul_(-self.gamma)
        idx = torch.arange(BM.shape[0], device=BM.device)
        S[idx, idx] += 1.0
        S[:-1, -1] = 0.0
        S[-1, :-1] = 0.0
        S[-1, -1] = 1.0 - self.gamma

        inv_term = torch.linalg.solve(S,alpha_tilde)

        del S, alpha_tilde, BM

        # Build Φ̃ᵀ directly
        tilde_phi_T = torch.zeros(
            (d + 1, m + 1),
            device=phi_sub_next_obs.device,
            dtype=phi_sub_next_obs.dtype,
        )

        tilde_phi_T[:d, :m] = phi_sub_next_obs.T
        # sink_state is zero except at index d-1, so only this row changes
        tilde_phi_T[d - 1, :m] -= sink_norm * psi_sub_obs_action.sum(dim=1)
        tilde_phi_T[d - 1, m] = sink_norm
        occupancy = tilde_phi_T @ inv_term
        occupancy.mul_(1 - self.gamma)

        return occupancy


    
    def compute_gradient_coefficient_nystrom_memory_efficient_and_projection(
            self, 
            phi_sub_next_obs:torch.Tensor,
            psi_sub_obs_action:torch.Tensor,
            H: torch.Tensor,
            pi: torch.Tensor,
            E: torch.Tensor,
            alpha:torch.Tensor,
            sink_norm: float,
            B_nystrom: Optional[torch.Tensor],
            eig_vecs_r: Optional[torch.Tensor],
            
        ) -> torch.Tensor:
        """Compute gradient coefficient for policy update."""
        # Identity matrix
        # I_n_plus1 = torch.eye(psi_all_obs_action.shape[0], device=self.device)
        N = H.shape[0]
        m = phi_sub_next_obs.shape[0]
        d = phi_sub_next_obs.shape[1]

        # Build Phi-tilde directly, without upper_left_sub temporary
        tilde_phi_sub_next_obs_T = torch.zeros(

            (d + 1, m + 1),

            device=phi_sub_next_obs.device,

            dtype=phi_sub_next_obs.dtype,

        )
        tilde_phi_sub_next_obs_T[:d, :m] = phi_sub_next_obs.T
        tilde_phi_sub_next_obs_T[d - 1, :m] -= sink_norm * psi_sub_obs_action.sum(dim=1)
        tilde_phi_sub_next_obs_T[d - 1, m] = sink_norm
        tilde_phi_sub_next_obs = tilde_phi_sub_next_obs_T.T

        sink_state = torch.zeros((d,1), device=self.device, dtype=phi_sub_next_obs.dtype)
        sink_state[-1] = sink_norm

        M = H*(E@pi.T) # [n, m]

        
        BM = B_nystrom @ M
    

        tilde_eig_vecs_r = torch.zeros((m + 1, eig_vecs_r.shape[1]+1), device=eig_vecs_r.device, dtype=eig_vecs_r.dtype)
        tilde_eig_vecs_r[:-1, :-1] = eig_vecs_r
        tilde_eig_vecs_r[-1, -1] = 1.0


        # Build S = I - gamma * tilde_B * tilde_M directly
        S = torch.empty(
            (BM.shape[0] + 1, BM.shape[1] + 1),
            device=BM.device,
            dtype=BM.dtype,
        )

        # 2γ(1 − γ)² B̃ₙᵧᵀ(I − B̃ₙᵧM̃)⁻ᵀΦ̃ₘΦ̃ₘᵀ(I − B̃ₙᵧM̃)⁻¹α̃ₘ

        S[:-1, :-1] = BM
        S[:-1, :-1].mul_(-self.gamma)
        idx = torch.arange(BM.shape[0], device=BM.device)
        S[idx, idx] += 1.0
        S[:-1, -1] = 0.0
        S[-1, :-1] = 0.0
        S[-1, -1] = 1.0 - self.gamma

        tilde_S_r = tilde_eig_vecs_r.T @ S @ tilde_eig_vecs_r
        # del S, BM
        tilde_S_r_reg = tilde_S_r + 1e-6 * torch.eye(tilde_S_r.shape[0], device=tilde_S_r.device, dtype=tilde_S_r.dtype)
        # del tilde_S_r

        tilde_phi_kernel = self.state_kernel(tilde_phi_sub_next_obs, tilde_phi_sub_next_obs) 
        tilde_phi_kernel_r = tilde_eig_vecs_r.T @ tilde_phi_kernel @ tilde_eig_vecs_r
        # del tilde_phi_kernel, tilde_phi_sub_next_obs
        
        tilde_B = torch.zeros(B_nystrom.shape[0] + 1, B_nystrom.shape[1] + 1, device=B_nystrom.device, dtype=B_nystrom.dtype)
        tilde_B[:-1, :-1] = B_nystrom
        tilde_B[-1, -1] = 1.0
        tilde_B_r = tilde_eig_vecs_r.T @  tilde_B #@ tilde_eig_vecs_r
        inv_tilde_S_r_reg = torch.linalg.solve(tilde_S_r_reg, tilde_phi_kernel_r) 
        # del tilde_phi_kernel_r
        left_term_r = tilde_B_r.T @ inv_tilde_S_r_reg
       
        # right_term = symmetric_term.T @ tilde_alpha, without tilde_alpha
        tilde_alpha = torch.ones((alpha.shape[0] + 1, 1), device=alpha.device, dtype=alpha.dtype)
        tilde_alpha[:-1] = alpha
        tilde_alpha_r = tilde_eig_vecs_r.T @ tilde_alpha
        right_term_r = torch.linalg.solve(tilde_S_r_reg, tilde_alpha_r)

        gradient = left_term_r @ right_term_r

        gradient.mul_(2 * self.gamma * ((1 - self.gamma) ** 2))

        # gradient = tilde_eig_vecs_r @ gradient_r
        return gradient
    
    def compute_gradient_coefficient_nystrom_memory_efficient(
            self, 
            phi_all_obs: torch.Tensor,
            phi_all_next_obs:torch.Tensor, 
            phi_sub_next_obs:torch.Tensor,
            psi_all_obs_action:torch.Tensor, 
            psi_sub_obs_action:torch.Tensor,
            H: torch.Tensor,
            pi: torch.Tensor,
            E: torch.Tensor,
            alpha:torch.Tensor,
            sink_norm: float,
            B_nystrom: Optional[torch.Tensor] = None,
            phi_sub_obs: Optional[torch.Tensor] = None,
            all_actions: Optional[torch.Tensor] = None,
            sub_actions: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """Compute gradient coefficient for policy update."""
        # Identity matrix
        # I_n_plus1 = torch.eye(psi_all_obs_action.shape[0], device=self.device)
        N = psi_all_obs_action.shape[0]
        m = phi_sub_next_obs.shape[0]
        d = phi_sub_next_obs.shape[1]

        # Build Phi-tilde directly, without upper_left_sub temporary
        tilde_phi_sub_next_obs_T = torch.zeros(

            (d + 1, m + 1),

            device=phi_sub_next_obs.device,

            dtype=phi_sub_next_obs.dtype,

        )
        tilde_phi_sub_next_obs_T[:d, :m] = phi_sub_next_obs.T
        tilde_phi_sub_next_obs_T[d - 1, :m] -= sink_norm * psi_sub_obs_action.sum(dim=1)
        tilde_phi_sub_next_obs_T[d - 1, m] = sink_norm
        tilde_phi_sub_next_obs = tilde_phi_sub_next_obs_T.T

        sink_state = torch.zeros((phi_all_next_obs.shape[1],1), device=self.device, dtype=phi_all_next_obs.dtype)
        sink_state[-1] = sink_norm

        M = H*(E@pi.T) # [n, m]

        if B_nystrom is not None:
            B = B_nystrom
            BM = B_nystrom @ M
        else:
            # Nyström matrices
            B = self.compute_B_nystrom(
                psi_all_obs_action,
                psi_sub_obs_action,
                svd_truncation=self.pca_truncation,
                phi_all_obs=phi_all_obs,
                phi_sub_obs=phi_sub_next_obs if phi_sub_obs is None else phi_sub_obs,
                all_actions=all_actions,
                sub_actions=sub_actions,
            )
            BM = B @ M
    

        # Build S = I - gamma * tilde_B * tilde_M directly
        S = torch.empty(

            (BM.shape[0] + 1, BM.shape[1] + 1),

            device=BM.device,

            dtype=BM.dtype,

        )

        # 2 γ (1 - γ)² Ã_ny⁻ᵀ (I - γ Ã_ny⁻¹M̃)⁻ᵀΦ̃_m Φ̃_mᵀ(I - γ Ã_ny⁻¹M̃)⁻¹ α̃_m 

        S[:-1, :-1] = BM
        S[:-1, :-1].mul_(-self.gamma)
        idx = torch.arange(BM.shape[0], device=BM.device)
        S[idx, idx] += 1.0
        S[:-1, -1] = 0.0
        S[-1, :-1] = 0.0
        S[-1, -1] = 1.0 - self.gamma

        phi_kernel = self.state_kernel(tilde_phi_sub_next_obs, tilde_phi_sub_next_obs) 

        tilde_B = torch.zeros(B.shape[0] + 1, B.shape[1] + 1, device=B.device, dtype=B.dtype)
        tilde_B[:-1, :-1] = B
        tilde_B[-1, -1] = 1.0
        left_term = tilde_B.T @ torch.linalg.solve(S.T,phi_kernel)   

       

        # right_term = symmetric_term.T @ tilde_alpha, without tilde_alpha
        tilde_alpha = torch.ones((alpha.shape[0] + 1, 1), device=alpha.device, dtype=alpha.dtype)
        tilde_alpha[:-1] = alpha
        right_term = torch.linalg.solve(S, tilde_alpha)

        gradient = left_term @ right_term

        gradient.mul_(2 * self.gamma * ((1 - self.gamma) ** 2))

        return gradient
           


# # Identity matrix
# I_n_plus1 = torch.eye(psi_all_obs_action.shape[0], device=self.device)

# sink_state = torch.zeros((phi_all_next_obs.shape[1],1), device=self.device, dtype=phi_all_next_obs.dtype)
# sink_state[-1] = sink_norm

# # Computing Ψ̃ and Φ̃ are now of shape [N+1, d*|A| + 2] and [N+1, d + 2] respectively
# upper_left = phi_all_next_obs.T - sink_state@torch.ones((1, psi_all_obs_action.shape[1]), device=psi_all_obs_action.device, dtype=psi_all_obs_action.dtype)@psi_all_obs_action.T
# tilde_phi_all_next_obs_transposed = torch.zeros((phi_all_next_obs.shape[1]+1, phi_all_next_obs.shape[0]+1), device=phi_all_next_obs.device, dtype=phi_all_next_obs.dtype)
# tilde_phi_all_next_obs_transposed[:upper_left.shape[0], :upper_left.shape[1]] = upper_left
# assert sink_state.shape[0] == upper_left.shape[0], "Sink state and upper left matrix row size mismatch"
# tilde_phi_all_next_obs_transposed[:sink_state.shape[0], -1:] = sink_state
# tilde_phi_all_next_obs = tilde_phi_all_next_obs_transposed.T
# assert torch.all(tilde_phi_all_next_obs_transposed[:sink_state.shape[0], -1:] == sink_state), "Last column of tilde_phi_all_next_obs should be sink_state"

# # Ã augmented to be [A 0; 0 1]
# # Symmetric positive definite matrix A = ψψᵀ + λI
# K_ref = self.kernel(psi_all_obs_action, psi_all_obs_action) if K is None else K
# A_ref = K_ref + self.lambda_reg * I_n_plus1
# tilde_A_ref = torch.zeros(A_ref.shape[0] + 1, A_ref.shape[1] + 1, device=A_ref.device, dtype=A_ref.dtype)
# tilde_A_ref[:-1, :-1] = A_ref
# tilde_A_ref[-1, -1] = 1.0

# # M̃ augmented to be [M 0; 0 1]
# tilde_M_ref = torch.zeros(M.shape[0] + 1, M.shape[1] + 1, device=M.device, dtype=M.dtype)
# tilde_M_ref[:-1, :-1] = M
# tilde_M_ref[-1, -1] = 1.0

# # α̃ augmented to be [α; 1]
# tilde_alpha_ref = torch.ones((alpha.shape[0] + 1, 1), device=alpha.device, dtype=alpha.dtype)
# tilde_alpha_ref[:-1] = alpha

# # ** COMPUTATION STEP **
# # Compute Cholesky decomposition and solve: BM = A⁻¹M
# # L = torch.linalg.cholesky(A)
# # BM = torch.cholesky_solve(M, L)
# BM_ref= torch.linalg.solve(A_ref, M) # [n, n]
# tilde_B_tilde_M_ref = torch.zeros(BM_ref.shape[0] + 1, BM_ref.shape[1] + 1, device=BM_ref.device, dtype=BM_ref.dtype)
# tilde_B_tilde_M_ref[:-1, :-1] = BM_ref
# tilde_B_tilde_M_ref[-1, -1] = 1.0

# # gradient = 2 γ (1 - γ)² Ã⁻ᵀ (I - γ Ã⁻¹M̃)⁻ᵀΦ̃Φ̃ᵀ(I - γ Ã⁻¹M̃)⁻¹ α̃ 
# # State and state-action similarities can use different Gaussian bandwidths.
# phi_kernel_ref = self.state_kernel(tilde_phi_all_next_obs, tilde_phi_all_next_obs) # [n+1, n+1]
# I_n_plus1 = torch.eye(tilde_B_tilde_M_ref.shape[0], device=tilde_B_tilde_M_ref.device, dtype=tilde_B_tilde_M_ref.dtype)
# # Left term: Ã⁻ᵀ(I - γB̃M̃)⁻ᵀΦ̃Φ̃ᵀ
# left_term_ref = torch.linalg.solve((I_n_plus1 - self.gamma * tilde_B_tilde_M_ref).T@tilde_A_ref.T, phi_kernel_ref) # [n+1, n+1]

# # (I - γ Ã⁻¹M̃)⁻ᵀΦ̃Φ̃ᵀ
# # inv_term_kernel = torch.linalg.solve((I_n_plus1 - self.gamma * tilde_B_tilde_M).T, phi_kernel_ref) # [n+1, n+1]
# # Solve Ãᵀ x = left_term_without_b using Cholesky
# # L_T = torch.linalg.cholesky(tilde_A.T)
# # left_term = torch.cholesky_solve(inv_term_kernel, L_T)

# # right term: (I - γ Ã⁻¹M̃)⁻¹ α̃
# right_term_ref = torch.linalg.solve((I_n_plus1 - self.gamma * tilde_B_tilde_M_ref), tilde_alpha_ref)

# # gradient = 2 γ (1 - γ)² Ã⁻ᵀ (I - γ Ã⁻¹M̃)⁻ᵀΦ̃Φ̃ᵀ(I - γ Ã⁻¹M̃)⁻¹ α̃
# gradient_ref = 2 * self.gamma * ((1 - self.gamma) ** 2) * left_term_ref @ right_term_ref




# # Identity matrix
# # I_n_plus1 = torch.eye(psi_all_obs_action.shape[0], device=self.device)
# N = psi_all_obs_action.shape[0]
# m = phi_sub_next_obs.shape[0]
# d = phi_sub_next_obs.shape[1]

# # Build Phi-tilde directly, without upper_left_sub temporary
# tilde_phi_sub_next_obs_T = torch.zeros(

#     (d + 1, m + 1),

#     device=phi_sub_next_obs.device,

#     dtype=phi_sub_next_obs.dtype,

# )
# tilde_phi_sub_next_obs_T[:d, :m] = phi_sub_next_obs.T
# tilde_phi_sub_next_obs_T[d - 1, :m] -= sink_norm * psi_sub_obs_action.sum(dim=1)
# tilde_phi_sub_next_obs_T[d - 1, m] = sink_norm
# tilde_phi_sub_next_obs = tilde_phi_sub_next_obs_T.T

# sink_state = torch.zeros((phi_all_next_obs.shape[1],1), device=self.device, dtype=phi_all_next_obs.dtype)
# sink_state[-1] = sink_norm

# M = H*(E@pi.T) # [n, m]

# if B_nystrom is not None:
#     B = B_nystrom
#     BM = B_nystrom @ M
# else:
#     # Nyström matrices
#     B = self.compute_B_nystrom(
#         psi_all_obs_action,
#         psi_sub_obs_action,
#         svd_truncation=self.svd_truncation,
#         phi_all_obs=phi_all_obs,
#         phi_sub_obs=phi_sub_next_obs if phi_sub_obs is None else phi_sub_obs,
#         all_actions=all_actions,
#         sub_actions=sub_actions,
#     )
#     BM = B @ M


# # Build S = I - gamma * tilde_B * tilde_M directly
# S = torch.empty(

#     (BM.shape[0] + 1, BM.shape[1] + 1),

#     device=BM.device,

#     dtype=BM.dtype,

# )

# S[:-1, :-1] = BM
# S[:-1, :-1].mul_(-self.gamma)
# idx = torch.arange(BM.shape[0], device=BM.device)
# S[idx, idx] += 1.0
# S[:-1, -1] = 0.0
# S[-1, :-1] = 0.0
# S[-1, -1] = 1.0 - self.gamma

# symmetric_term = torch.linalg.solve(S.T,tilde_phi_sub_next_obs)   

# # left_term = tilde_B.T @ symmetric_term, without tilde_B
# left_term = torch.empty(
#     (B.shape[1] + 1, symmetric_term.shape[1]),
#     device=symmetric_term.device,
#     dtype=symmetric_term.dtype,
# )
# left_term[:-1] = B.T @ symmetric_term[:-1]
# left_term[-1:] = symmetric_term[-1:]

# # right_term = symmetric_term.T @ tilde_alpha, without tilde_alpha

# right_term = symmetric_term[:-1].T @ alpha + symmetric_term[-1:].T

# gradient = left_term @ right_term

# gradient.mul_(2 * self.gamma * ((1 - self.gamma) ** 2))

# def compute_B_nystrom_no_pseudo(
#             self,
#             psi_all_obs_action: torch.Tensor,
#             psi_sub_obs_action: torch.Tensor,
#             svd_truncation: Optional[int] = None,
#             phi_all_obs: Optional[torch.Tensor] = None,
#             phi_sub_obs: Optional[torch.Tensor] = None,
#             all_actions: Optional[torch.Tensor] = None,
#             sub_actions: Optional[torch.Tensor] = None
#         ) -> torch.Tensor:
#         N = psi_all_obs_action.shape[0]
#         if phi_all_obs is not None and phi_sub_obs is not None and all_actions is not None and sub_actions is not None:
#             K_nm = self.state_action_kernel(phi_all_obs, phi_sub_obs, all_actions, sub_actions)
#             K_mm = self.state_action_kernel(phi_sub_obs, phi_sub_obs, sub_actions, sub_actions)
#         else:
#             K_nm = self.kernel(psi_all_obs_action, psi_sub_obs_action) # [n, m]
#             K_mm = self.kernel(psi_sub_obs_action, psi_sub_obs_action) # [m, m]
#         A_nystrom = K_nm.T@K_nm + self.lambda_reg * K_mm # [m, m] # TODO add N
#         return torch.linalg.solve(A_nystrom, K_nm.T)
