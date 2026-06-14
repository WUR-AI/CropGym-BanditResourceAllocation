import datetime
import os
import math
from dataclasses import dataclass
from collections import deque
from typing import List, Tuple, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from cropgym import _DEFAULT_MODEL_DIR, _DEFAULT_LOGDIR


#  Utilities

def positive_param(raw: torch.Tensor) -> torch.Tensor:
    # Softplus with small beta for stable gradients
    return F.softplus(raw) + 1e-6


def add_jitter(K: torch.Tensor, jitter: float = 1e-6) -> torch.Tensor:
    return K + jitter * torch.eye(K.shape[-1], device=K.device, dtype=K.dtype)


#  Kernels (scalar)

class RBF(nn.Module):
    """
    ARD RBF on R^d: k(x,x') = exp(-0.5 * ||(x-x') / ℓ||^2)
    lengthscale: (d,) or scalar (learned, constrained positive)
    """
    def __init__(self, d: int, init_log_ell: float = math.log(0.5)):
        super().__init__()
        self.raw_lengthscale = nn.Parameter(torch.full((d,), init_log_ell))

    @property
    def lengthscale(self) -> torch.Tensor:
        return positive_param(self.raw_lengthscale)

    def forward(self, X: torch.Tensor, Xp: torch.Tensor) -> torch.Tensor:
        # X: (n,d), Xp: (m,d) -> (n,m)
        ell = self.lengthscale  # (d,)
        Xs = X / ell
        Ys = Xp / ell
        xx = (Xs**2).sum(dim=1, keepdim=True)
        yy = (Ys**2).sum(dim=1, keepdim=True).T
        # dist^2 = ||x||^2 + ||y||^2 - 2x·y
        d2 = xx + yy - 2.0 * (Xs @ Ys.T)
        return torch.exp(-0.5 * d2.clamp_min_(0.0))

    def diag(self, X: torch.Tensor) -> torch.Tensor:
        return torch.ones(X.shape[0], device=X.device, dtype=X.dtype)


class Matern32(nn.Module):
    """
    Matérn-3/2 with ARD: k(r)= (1+sqrt(3) r) exp(-sqrt(3) r), r = ||(x-x')/ℓ||_2
    """
    def __init__(self, d: int, init_log_ell: float = math.log(0.7)):
        super().__init__()
        self.raw_lengthscale = nn.Parameter(torch.full((d,), init_log_ell))

    @property
    def lengthscale(self) -> torch.Tensor:
        return positive_param(self.raw_lengthscale)

    def forward(self, X: torch.Tensor, Xp: torch.Tensor) -> torch.Tensor:
        ell = self.lengthscale
        Xs = X / ell
        Ys = Xp / ell
        # pairwise Euclidean
        xx = (Xs**2).sum(1, keepdim=True)
        yy = (Ys**2).sum(1, keepdim=True).T
        d2 = xx + yy - 2.0 * (Xs @ Ys.T)
        # Add small epsilon before sqrt to avoid infinite/NaN gradients at r=0
        eps = 1e-12
        r = (d2.clamp_min(0.0) + eps).sqrt()
        sqrt3r = math.sqrt(3.0) * r
        return (1.0 + sqrt3r) * torch.exp(-sqrt3r)

    def diag(self, X: torch.Tensor) -> torch.Tensor:
        return torch.ones(X.shape[0], device=X.device, dtype=X.dtype)


# --------------------------- Neural feature map g(θ) ---------------------------

class FeatureNet(nn.Module):

    def __init__(
        self,
        d_theta: int,
        m: int,
        hidden: int = 64,
        depth: int = 1,
        normalize: bool = True,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = d_theta
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            in_dim = hidden
        layers += [nn.Linear(in_dim, m)]
        self.net = nn.Sequential(*layers)

        # Normalizing g(θ) is a very effective stabilizer for NN-AGP.
        # It prevents the collaborative kernel magnitude from blowing up.
        self.normalize = normalize
        self.g_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, Theta: torch.Tensor) -> torch.Tensor:
        # Accept (N,d) only. If a sequence arrives, use the last step by default.
        if Theta.dim() == 3:
            Theta = Theta[:, -1, :]  # (N,d)
        elif Theta.dim() != 2:
            raise ValueError(f"Theta must be (N,d) (or (N,T,d) which will be last-stepped), got {Theta.shape}")

        G = self.net(Theta)
        if self.normalize:
            G = F.normalize(G, p=2, dim=-1)
        return self.g_scale * G

class LSTMFeatureNet(nn.Module):

    def __init__(
        self,
        d_theta: int,
        m: int,
        hidden: int = 128,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=d_theta,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden * self.num_directions
        self.proj = nn.Linear(out_dim, m)
        self.normalize = True
        self.g_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, Theta: torch.Tensor) -> torch.Tensor:
        """
        Theta: (N, dθ) or (N, T, dθ)
        Returns: g ∈ (N, m), using the last layer hidden state.
        """
        if Theta.dim() == 2:
            Theta = Theta.unsqueeze(1)  # (N, 1, dθ)
        elif Theta.dim() != 3:
            raise ValueError(f"Theta must be (N,d) or (N,T,d), got {Theta.shape}")

        # LSTM
        pool = False
        raw_output = True

        if pool:
            output, _ = self.lstm(Theta)

            # mean pooling over time (uses all timesteps)
            pooled = output.mean(dim=1)  # (N, out_dim)

            G = self.proj(output)  # (N, m)

        elif raw_output:
            output, (hn, _) = self.lstm(Theta)

            # hn: (num_layers * num_directions, N, hidden)
            hn = hn.view(self.num_layers, self.num_directions, Theta.size(0), self.hidden)

            last_layer = hn[-1]  # (num_dirs, N, hidden)

            if self.num_directions == 2:
                last = torch.cat([last_layer[0], last_layer[1]], dim=-1)
            else:
                last = last_layer[0]

            G = self.proj(last)  # (N, m)

        else:

            _, (hn, _) = self.lstm(Theta)  # hn: (num_layers*num_dirs, N, hidden)

            # Reshape to (num_layers, num_dirs, N, hidden)
            hn = hn.view(self.num_layers, self.num_directions, Theta.size(0), self.hidden)

            # Take last layer
            last_layer = hn[-1]  # (num_dirs, N, hidden)

            # If bidirectional, concat directions; else just squeeze
            if self.num_directions == 2:
                last = torch.cat([last_layer[0], last_layer[1]], dim=-1)  # (N, 2*hidden)
            else:
                last = last_layer[0]  # (N, hidden)

            G = self.proj(last)  # (N, m)
        if self.normalize:
            G = F.normalize(G, p=2, dim=-1)
        return self.g_scale * G

# --------------------------- Collaborative multi-output GP K(x,x') ---------------------------
# p(x) = [p_1(x),...,p_m(x)]^T with covariance:
#   K(x,x') = Σ_q A_q k_q(x,x') + Diag( k~_1(x,x'),...,k~_m(x,x') )
# where A_q = L_q L_q^T ensures PSD. (Eq. (4))

class CollaborativeMOGP(nn.Module):
    def __init__(
        self,
        d_x: int,
        m: int,
        Q: int = 1,
        shared_kernel: Literal["rbf", "matern32"] = "rbf",
        indep_kernel: Literal["rbf", "matern32"] = "rbf",
    ):
        super().__init__()
        self.m = m
        self.Q = Q

        # Shared latent kernels k_q
        KernelCls = RBF if shared_kernel == "rbf" else Matern32
        self.k_shared = nn.ModuleList([KernelCls(d_x) for _ in range(Q)])

        # PSD mixing matrices A_q = L_q L_qᵀ
        self.raw_L = nn.ParameterList([nn.Parameter(0.05 * torch.randn(m, m)) for _ in range(Q)])

        # Independent per-output kernels k~_l
        KernelIndCls = RBF if indep_kernel == "rbf" else Matern32
        self.k_indep = nn.ModuleList([KernelIndCls(d_x) for _ in range(m)])

    def A_q(self, q: int) -> torch.Tensor:
        L = self.raw_L[q]
        # force lower-triangular for stability
        L = torch.tril(L)
        return L @ L.T + 1e-5 * torch.eye(self.m, device=L.device, dtype=L.dtype)

    # Build K̃((X,Θ),(X',Θ')) = g(Θ)ᵀ K(X,X') g(Θ')  (Prop. 1)
    def K_tilde(
        self,
        X: torch.Tensor, Theta: torch.Tensor,
        Xp: torch.Tensor, Thetap: torch.Tensor,
        g_theta: torch.Tensor, g_thetap: torch.Tensor
    ) -> torch.Tensor:
        """
        X:(n,d), Theta:(n,dθ), g_theta:(n,m)
        Xp:(m_,d), Thetap:(m_,dθ), g_thetap:(m_,m)
        returns (n,m_)
        """
        n, m_ = X.shape[0], Xp.shape[0]
        device = X.device
        out = torch.zeros(n, m_, device=device, dtype=X.dtype)

        # Shared Σ_q  [ k_q(X,X') ∘ (G A_q G'^T) ]
        for q in range(self.Q):
            Kxx = self.k_shared[q](X, Xp)  # (n,m_)
            A = self.A_q(q)                # (m,m)
            S = g_theta @ A @ g_thetap.T   # (n,m_)
            out = out + Kxx * S

        # Independent Σ_l [ k~_l(X,X') ∘ (g_l g_l'^T) ]
        for l in range(self.m):
            Kll = self.k_indep[l](X, Xp)  # (n,m_)
            gl = g_theta[:, l:l+1]        # (n,1)
            glp = g_thetap[:, l:l+1]      # (m_,1)
            out = out + Kll * (gl @ glp.T)

        return out

    def K_tilde_diag(
        self, X: torch.Tensor, Theta: torch.Tensor, g_theta: torch.Tensor
    ) -> torch.Tensor:
        # diag term: g(θ)ᵀ K(x,x) g(θ) = Σ_q k_q(x,x) * gᵀ A_q g + Σ_l k~_l(x,x) * g_l^2
        n = X.shape[0]
        device = X.device
        diag = torch.zeros(n, device=device, dtype=X.dtype)

        for q in range(self.Q):
            kdiag = self.k_shared[q].diag(X)         # (n,)
            A = self.A_q(q)
            quad = (g_theta @ A * g_theta).sum(dim=1)  # diag of g A gᵀ
            diag = diag + kdiag * quad

        for l in range(self.m):
            kdiag = self.k_indep[l].diag(X)          # (n,)
            diag = diag + kdiag * (g_theta[:, l] ** 2)

        return diag


# --------------------------- NN-AGP model ---------------------------

class NNAGP(nn.Module):
    """
    End-to-end NN-AGP:
      - g(θ)
      - collaborative MOGP for K(x,x')
      - exact GP posterior over f(x;theta) using K̃
    """
    def __init__(
            self,
            d_theta: int,
            d_x: int,
            m: int,
            Q: int = 1,
            kernel='matern',
            device: Optional[torch.device] = None,
            g_net: Optional[nn.Module] = None,
            use_lstm: bool = False,
    ):
        super().__init__()
        self.g_net = g_net
        if g_net is None and use_lstm:
            self.g_net = LSTMFeatureNet(d_theta, m, hidden=128, num_layers=3)
        elif g_net is None and not use_lstm:
            self.g_net = FeatureNet(d_theta, m, hidden=64, depth=2, normalize=True)
        self.mogp = CollaborativeMOGP(d_x, m, Q=Q, shared_kernel=kernel, indep_kernel=kernel)
        self.raw_noise = nn.Parameter(torch.tensor(-2.0))  # σ_ε ≈ 0.12
        self.jitter = 1e-3
        self.device = device or torch.device("cpu")
        self.to(self.device)

        # cache
        self._train_cache = {}  # keys: ("Theta","X","y","L","alpha","K̃")

    @property
    def noise(self) -> torch.Tensor:
        sigma = positive_param(self.raw_noise)
        return torch.clamp(sigma, min=3e-3)  # floor the noise

    def _clear_cache(self):
        self._train_cache.clear()

    def _robust_cholesky(self, K: torch.Tensor, base_jitter: float | None = None):
        """Symmetrize + escalating jitter + eigen fallback to get a stable Cholesky."""
        K = 0.5 * (K + K.T)  # enforce symmetry
        n = K.shape[0]
        eye = torch.eye(n, device=K.device, dtype=K.dtype)

        jitter = self.jitter if base_jitter is None else base_jitter
        # try escalating jitter a few times
        for _ in range(7):
            try:
                return torch.linalg.cholesky(K + (self.noise ** 2) * eye + jitter * eye)
            except RuntimeError:
                jitter *= 10.0

        # final fallback: eig cleanup (make PSD), then Cholesky
        w, V = torch.linalg.eigh(K)
        w = w.clamp_min(1e-12)  # push any tiny negatives to 0+
        K_psd = (V * w) @ V.T
        return torch.linalg.cholesky(K_psd + (self.noise ** 2 + jitter) * eye)

    # ---- training objective: negative log-marginal likelihood (Eq. (5))
    def nll(self, act: torch.Tensor, context_in: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # act, context_in, y = act.to(self.device), context_in.to(self.device), y.to(self.device)
        G = self.g_net(context_in)                        # (n,m)
        Kt = self.mogp.K_tilde(act, context_in, act, context_in, G, G)  # (n,n)

        # Use robust Cholesky that adds noise^2 I and jitter internally
        L = self._robust_cholesky(Kt)
        alpha = torch.cholesky_solve(y.unsqueeze(1), L).squeeze(1)  # (n,)

        log_det = 2.0 * torch.log(torch.diag(L)).sum()
        quad = 0.5 * (y @ alpha)
        const = 0.5 * act.shape[0] * math.log(2.0 * math.pi)
        nll = quad + 0.5 * log_det + const

        # cache for prediction if inputs match
        self._train_cache = {
            "Theta": context_in.detach(),
            "X": act.detach(),
            "y": y.detach(),
            "L": L.detach(),
            "alpha": alpha.detach(),
            "G": G.detach(),
        }
        return nll

    # ---- exact posterior μ, σ at batch of (X*,Θ*) (Eq. (2))
    @torch.no_grad()
    def predict(self, X_star: torch.Tensor, Theta_star: torch.Tensor,
                train_X: torch.Tensor, train_Theta: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.device
        X_star = X_star.to(device)
        Theta_star = Theta_star.to(device)
        train_X = train_X.to(device)
        train_Theta = train_Theta.to(device)
        y = y.to(device)

        # reuse cached Cholesky based on a few flags
        need_refit = not self._train_cache or \
                     self._train_cache["X"].data_ptr() != train_X.data_ptr() or \
                     self._train_cache["Theta"].data_ptr() != train_Theta.data_ptr()

        if need_refit:
            G_tr = self.g_net(train_Theta)
            Kt = self.mogp.K_tilde(train_X, train_Theta, train_X, train_Theta, G_tr, G_tr)
            # Use robust Cholesky that adds noise^2 I and jitter internally
            L = self._robust_cholesky(Kt)
            alpha = torch.cholesky_solve(y.unsqueeze(1), L).squeeze(1)
            self._train_cache = {"Theta": train_Theta, "X": train_X, "y": y, "L": L, "alpha": alpha, "G": G_tr}
        else:
            L = self._train_cache["L"]
            alpha = self._train_cache["alpha"]

        G_tr = self._train_cache["G"]
        G_st = self.g_net(Theta_star)

        # cross-covariances and prior diagonal
        K_star = self.mogp.K_tilde(train_X, train_Theta, X_star, Theta_star, G_tr, G_st)          # (n,m)
        mu = K_star.T @ alpha                                                                      # (m,)

        v = torch.cholesky_solve(K_star, L)                                                        # (n,m)
        prior_diag = self.mogp.K_tilde_diag(X_star, Theta_star, G_st)                              # (m,)
        var = (prior_diag - (K_star * v).sum(dim=0)).clamp_min(0.0)
        return mu, var.sqrt()

    # posterior over a candidate set for one context theta_t (UCB and TS)
    @torch.no_grad()
    def posterior_on_candidates(
            self,
            Xc: torch.Tensor,
            theta: torch.Tensor,
            train_X: torch.Tensor,
            train_Theta: torch.Tensor,
            y: torch.Tensor,
            calculate_covariance: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:

        M = Xc.shape[0]

        # ---- normalize theta shape to either (1,d) or (1,T,d) ----
        if theta.dim() == 1:
            theta = theta.view(1, -1)  # (1,d)
        elif theta.dim() == 2:
            # could be (1,d) or (T,d) -> treat (T,d) as sequence
            if theta.shape[0] == 1:
                pass  # (1,d)
            else:
                theta = theta.unsqueeze(0)  # (1,T,d)
        elif theta.dim() == 3:
            # (1,T,d) already
            if theta.shape[0] != 1:
                raise ValueError(f"theta 3D must be (1,T,d), got {theta.shape}")
        else:
            raise ValueError(f"Unsupported theta shape {theta.shape}")

        # ---- build Theta_star to match candidates ----
        if theta.dim() == 2:
            # (1,d) -> (M,d)
            Theta_star = theta.expand(M, -1)
        else:
            # (1,T,d) -> (M,T,d)
            Theta_star = theta.expand(M, -1, -1)

        # compute posterior mean/std (predict handles Theta_star shape fine)
        mu, std = self.predict(Xc, Theta_star, train_X, train_Theta, y)
        if not calculate_covariance:
            return mu, std, None

        # returns μ_c (M,), σ_c (M,), and (optional) full covariance Σ_c (M,M)
        mu, std = self.predict(Xc, theta.expand(Xc.shape[0], -1), train_X, train_Theta, y)
        if not calculate_covariance:
            return mu, std, None


        # Build covariance only when needed (used for TS)
        # Σ = K_cand - K_*^T (K+σ^2I)^{-1} K_*
        G_tr = self._train_cache["G"]

        # candidate embeddings
        if theta.dim() == 2:
            G_c = self.g_net(theta.expand(M, -1))  # (M,m)
            theta_rep = theta.expand(M, -1)  # (M,d)
        else:
            G_c = self.g_net(theta.expand(M, -1, -1))  # (M,m)
            theta_rep = theta.expand(M, -1, -1)

        Kcand = self.mogp.K_tilde(Xc, theta.expand(Xc.shape[0], -1), Xc, theta.expand(Xc.shape[0], -1), G_c, G_c)
        K_star = self.mogp.K_tilde(train_X, train_Theta, Xc, theta.expand(Xc.shape[0], -1), G_tr, G_c)

        L = self._train_cache["L"]
        V = torch.cholesky_solve(K_star, L)                # (n,M)
        cov = Kcand - K_star.T @ V
        cov = (cov + cov.T) * 0.5                          # symmetrize
        cov = add_jitter(cov, 1e-10)
        return mu, std, cov


#  Acquisition rules

def beta_finite_candidates(t: int, M: int, delta: float = 0.1) -> float:
    # Finite-arm NNAGP-UCB schedule
    return 2.0 * math.log((M * (t**2) * (math.pi**2)) / (6.0 * delta))


@dataclass
class SelectionInfo:
    mu: torch.Tensor
    std: torch.Tensor
    ucb: Optional[torch.Tensor] = None
    sampled_vals: Optional[torch.Tensor] = None
    beta_t: Optional[float] = None
    rule: str = "ucb"


def ucb_components(mu: torch.Tensor, std: torch.Tensor, beta_t: float):
    """
    Inputs
      mu, std: shape (M,) over candidates
      beta_t:  scalar > 0
    Returns
      exploit = mu
      explore = sqrt(beta_t) * std
      ucb     = exploit + explore
    """
    scale = torch.sqrt(torch.as_tensor(beta_t, dtype=mu.dtype, device=mu.device))
    explore = scale * std
    exploit = mu
    ucb = exploit + explore
    return exploit, explore, ucb


class NNAGPBandit:
    def __init__(
            self,
            d_theta: int,
            d_x: int,
            m: int = 8,
            Q: int = 1,
            lr: float = 1e-4,
            device: Optional[torch.device] = None,
            posterior_type: Literal["gp", "neural_linear"] = "gp",
            buffer_size: int = 256,
            ridge_lambda: float = 1.0,
            coreset_size=256,
            coreset_mode="diverse",
            action_mode: Literal["joint", "factored"] = "factored",
            n_fields: Optional[int] = None,
            d_theta_per_field: Optional[int] = None,
            m_sub: int = 5,
            share_g_net: bool = False,
            use_farm_budget: bool = False,
            budget_tick: float = 5.0,
            budget_slack: float = 1e-6,
            use_lstm: bool = False,
    ):
        self.action_mode = action_mode
        self.n_fields = n_fields

        self.use_farm_budget = bool(use_farm_budget)
        self.budget_tick = float(budget_tick)
        self.budget_slack = float(budget_slack)
        self.use_lstm = use_lstm

        # ---- FACTORED MODE: orchestrator only ----
        if self.action_mode == "factored":
            if n_fields is None or n_fields <= 0:
                raise ValueError("factored mode requires n_fields > 0")
            if d_theta_per_field is None or d_theta_per_field <= 0:
                raise ValueError("factored mode requires d_theta_per_field > 0")

            self.share_g_net = bool(share_g_net)

            self.shared_g_net = None
            self.opt_g = None
            dev = device or torch.device("cpu")

            self.seq_len = 5
            self.seq_stride = 5
            self._ctx_step = 0
            self._theta_seq = deque(maxlen=self.seq_len)  # stores (n_fields, d_theta)
            self._last_theta_seq_fields = None

            if self.share_g_net:
                self.shared_g_net = FeatureNet(
                    int(d_theta_per_field),
                    int(m_sub),
                    hidden=64,
                    depth=2,
                    normalize=True
                ).to(dev) if not use_lstm else (
                    LSTMFeatureNet(
                        d_theta=int(d_theta_per_field),
                        m=int(m_sub),
                        hidden=128,
                        num_layers=1,
                        bidirectional=False,
                        dropout=0.0
                    ).to(dev)
                )
                self.opt_g = torch.optim.AdamW(self.shared_g_net.parameters(), lr=lr, weight_decay=1e-4)

            # Create sub-bandits normally first
            self.sub_bandits = [
                NNAGPBandit(
                    d_theta=int(d_theta_per_field),
                    d_x=1,
                    m=int(m_sub),
                    Q=Q,
                    lr=lr,
                    device=device,
                    posterior_type=posterior_type,
                    buffer_size=buffer_size,
                    ridge_lambda=ridge_lambda,
                    coreset_size=coreset_size,
                    coreset_mode=coreset_mode,
                    action_mode="joint",
                    share_g_net=False,  # prevent recursion
                    use_lstm=use_lstm
                )
                for _ in range(int(n_fields))
            ]

            # If sharing: inject shared g-net and rebuild sub optimizers excluding g params
            if self.share_g_net:
                for sb in self.sub_bandits:
                    sb.model = NNAGP(
                        d_theta=int(d_theta_per_field),
                        d_x=1,
                        m=int(m_sub),
                        Q=Q,
                        kernel="matern",
                        device=dev,
                        g_net=self.shared_g_net,
                        use_lstm=use_lstm
                    )

                    # optimizer should NOT update shared g-net
                    non_g_params = [p for n, p in sb.model.named_parameters() if not n.startswith("g_net.")]
                    sb.opt = torch.optim.AdamW(non_g_params, lr=lr, weight_decay=1e-4)

            self.t = 1
            self.model = None
            return
        self.posterior_type = posterior_type
        self.model = NNAGP(d_theta, d_x, m=m, Q=Q, device=device or torch.device("cpu"), use_lstm=use_lstm)

        self.coreset_size = int(coreset_size)
        self.coreset_mode = coreset_mode

        self.theta_hist: deque[torch.Tensor] = deque(maxlen=self.coreset_size)
        self.x_hist: deque[torch.Tensor] = deque(maxlen=self.coreset_size)
        self.y_hist: deque[torch.Tensor] = deque(maxlen=self.coreset_size)
        # store concatenated z=[x,theta] for coreset maintenance (exact GP path)
        self._z_gp_hist: deque[torch.Tensor] = deque(maxlen=self.coreset_size)
        self.t = 1
        # Cache stacked (X, Theta, y) tensors so that selection does NOT recreate
        # new tensors each call. This allows NNAGP.predict() to reuse its cached
        # Cholesky (data_ptr stays identical across calls).
        self._stack_cache_dirty = True
        self._stack_cache = {}

        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-4,
        )

        self.ridge_lambda = float(ridge_lambda)

        # --- Neural-Linear posterior (optional) ---
        # Features depend on BOTH (x, theta): phi = phi_net([x, theta])
        self.phi_net = nn.Sequential(
            nn.Linear(d_x + d_theta, 256),
            nn.ReLU(),
            nn.Linear(256, m),
        ).to(self.model.device)

        # Sufficient statistics for Bayesian linear regression: A = λI + Σ phi phi^T, b = Σ phi y
        self._A = (self.ridge_lambda * torch.eye(m, device=self.model.device, dtype=torch.get_default_dtype()))
        self._b = torch.zeros(m, device=self.model.device, dtype=torch.get_default_dtype())

        # Optional: small buffer for training phi_net (stores concatenated z=[x,theta])
        self._z_hist: deque[torch.Tensor] = deque(maxlen=buffer_size)
        self._yl_hist: deque[torch.Tensor] = deque(maxlen=buffer_size)

        self.opt_phi = torch.optim.AdamW(self.phi_net.parameters(), lr=lr, weight_decay=1e-4)

    def _seq_reset_if_needed(self):
        if (self._ctx_step % self.seq_stride) == 0:
            self._theta_seq.clear()

    def _seq_push(self, theta_fields: torch.Tensor):
        """
        Push ONE context snapshot per step into the internal sequence buffer.

        Expected: (n_fields, d)
        If given: (n_fields, T, d), keep only the latest step ([:, -1, :]).
        """
        th = theta_fields
        if th.dim() == 3:
            th = th[:, -1, :]  # (n_fields, T, d) -> (n_fields, d)
        elif th.dim() != 2:
            raise ValueError(f"_seq_push expected (n_fields,d) or (n_fields,T,d), got {tuple(th.shape)}")
        self._theta_seq.append(th.detach().cpu())

    def _seq_build(self, device):
        # returns (n_fields, T, d_theta)
        H = torch.stack(list(self._theta_seq), dim=0)

        # Defensive: if something 3D slipped through, collapse to latest step.
        # Desired: H is (L, n_fields, d)
        if H.dim() == 4:
            H = H[:, :, -1, :]  # (L, n_fields, T, d) -> (L, n_fields, d)
        if H.dim() != 3:
            raise ValueError(f"_seq_build expected H to be 3D (L,n_fields,d), got {tuple(H.shape)}")

        L, n_fields, d = H.shape
        T = self.seq_len

        if L < T:
            pad = H[0:1].expand(T - L, n_fields, d)
            H = torch.cat([pad, H], dim=0)

        H = H[-T:].permute(1, 0, 2).contiguous()  # (n_fields, T, d)
        return H.to(device)

    def _phi(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Compute neural-linear features phi([x,theta]) -> (m,) or (B,m)."""
        if x.dim() == 2 and x.shape[0] == 1:
            x = x.squeeze(0)
        if theta.dim() == 2 and theta.shape[0] == 1:
            theta = theta.squeeze(0)
        z = torch.cat([x, theta], dim=-1)
        return self.phi_net(z)

    def _w_mean_and_Ainv(self):
        """Return posterior mean weights and A^{-1}."""
        # A is PSD; use solve for stability
        Ainv = torch.linalg.inv(self._A)
        w_mean = Ainv @ self._b
        return w_mean, Ainv

    def _linear_posterior(self, X_candidates: torch.Tensor, theta_t: torch.Tensor):
        """Compute (mu, std) for each candidate under neural-linear posterior."""
        device = self.model.device
        X_candidates = X_candidates.to(device)
        theta_t = theta_t.to(device)
        if theta_t.dim() == 1:
            theta_rep = theta_t.unsqueeze(0).expand(X_candidates.shape[0], -1)  # (M, d)
        elif theta_t.dim() == 2:
            theta_rep = theta_t.unsqueeze(0).expand(X_candidates.shape[0], -1, -1)  # (M, T, d)
        else:
            raise ValueError(theta_t.shape)
        Phi = self._phi(X_candidates, theta_rep)  # (M, m)

        w_mean, Ainv = self._w_mean_and_Ainv()
        mu = Phi @ w_mean  # (M,)

        # predictive variance: sigma^2 * phi^T A^{-1} phi
        sigma2 = float(self.model.noise.detach().cpu().item() ** 2)
        quad = (Phi @ Ainv * Phi).sum(dim=1).clamp_min(0.0)
        var = sigma2 * quad
        std = var.sqrt()
        return mu, std, Phi, Ainv

    # ---- training step: maximize log marginal likelihood (Eq. (5))
    def train_step(self, steps: int = 200, lr: float = 3e-3) -> float:
        if getattr(self, "action_mode", "joint") == "factored":
            # If sharing g(θ), do one backward pass over mean sub-loss, then step:
            # - each sub-bandit optimizer (non-g params)
            # - shared g optimizer
            if getattr(self, "share_g_net", False) and getattr(self, "opt_g", None) is not None:
                # If neural-linear, g_net isn't used -> fallback
                if self.sub_bandits and getattr(self.sub_bandits[0], "posterior_type", "gp") == "neural_linear":
                    losses = [b.train_step(steps=steps, lr=lr) for b in self.sub_bandits]
                    losses = [float(x) for x in losses if x is not None]
                    return float(sum(losses) / max(len(losses), 1))

                # one joint update per outer step
                loss_val = 0.0
                for _ in range(steps):
                    self.opt_g.zero_grad()
                    for sb in self.sub_bandits:
                        sb.opt.zero_grad()

                    used = 0
                    total = None
                    for sb in self.sub_bandits:
                        if len(getattr(sb, "y_hist", [])) == 0:
                            continue
                        x, theta, y = sb._get_stacked_history(device=sb.model.device)
                        sb.model._clear_cache()
                        li = sb.model.nll(x, theta, y)
                        total = li if total is None else (total + li)
                        used += 1

                    if used == 0 or total is None:
                        return 0.0

                    loss = total / float(used)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.shared_g_net.parameters(), max_norm=5.0)
                    for sb in self.sub_bandits:
                        torch.nn.utils.clip_grad_norm_(sb.opt.param_groups[0]["params"], max_norm=5.0)

                    self.opt_g.step()
                    for sb in self.sub_bandits:
                        sb.opt.step()

                    loss_val = float(loss.detach().cpu())

                return loss_val

            # Default: independent sub-bandits
            losses = [b.train_step(steps=steps, lr=lr) for b in self.sub_bandits]
            losses = [float(x) for x in losses if x is not None]
            return float(sum(losses) / max(len(losses), 1))


        if self.posterior_type == "neural_linear":
            if len(self._z_hist) == 0:
                return 0.0

            # Train phi_net via differentiable ridge regression on mini-batches
            device = self.model.device
            Z = torch.vstack(tuple(self._z_hist)).to(device)  # (N, d_x+d_theta)
            y = torch.hstack(tuple(self._yl_hist)).to(device)  # (N,)

            loss_val = 0.0
            # simple batching
            batch_size = min(128, Z.shape[0])
            for _ in range(steps):
                idx = torch.randint(0, Z.shape[0], (batch_size,), device=device)
                Zb = Z[idx]
                yb = y[idx]

                Phi = self.phi_net(Zb)  # (B, m)
                # ridge solution: w = (Phi^T Phi + λI)^{-1} Phi^T y
                A = Phi.T @ Phi + self.ridge_lambda * torch.eye(Phi.shape[1], device=device, dtype=Phi.dtype)
                b = Phi.T @ yb
                w = torch.linalg.solve(A, b)
                pred = Phi @ w
                loss = F.mse_loss(pred, yb)

                self.opt_phi.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.phi_net.parameters(), max_norm=5.0)
                self.opt_phi.step()
                loss_val = float(loss.detach().cpu())

            return loss_val

        if len(self.y_hist) == 0:
            return 0.0
        device = self.model.device
        x, theta, y = self._get_stacked_history(device=device)
        self.model._clear_cache()

        loss_val = 0.0
        for _ in range(steps):
            self.opt.zero_grad()
            loss = self.model.nll(x, theta, y)
            # n = x.shape[0]
            # print(f"[diag] nll={float(loss):.3f}  per_samp={float(loss) / n:.4f}  sigma={float(self.model.noise):.4g}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.opt.step()
            loss_val = float(loss.detach().cpu())
        return loss_val

    # ---- choose x_t by UCB over a finite candidate set for current θ_t (Eq. (3))
    @torch.no_grad()
    def select_ucb(
            self,
            theta_t: torch.Tensor,
            X_candidates: torch.Tensor,
            delta: float = 0.1,
            deterministic: bool = False,
    ) -> Tuple[torch.Tensor, SelectionInfo]:
        if len(self.y_hist) == 0:
            # cold-start: pick random
            idx = torch.randint(0, X_candidates.shape[0], (1,)).item()
            mu = torch.zeros(X_candidates.shape[0])
            std = torch.ones(X_candidates.shape[0])
            return X_candidates[idx], SelectionInfo(mu=mu, std=std, ucb=None, beta_t=None, rule="ucb")

        if self.posterior_type == "neural_linear":
            mu, std, _, _ = self._linear_posterior(X_candidates, theta_t)
            beta_t = beta_finite_candidates(self.t, X_candidates.shape[0], delta) if not deterministic else 0
            ucb = mu + (beta_t ** 0.5) * std
            idx = int(torch.argmax(ucb).item())
            return X_candidates[idx], SelectionInfo(mu=mu.detach().cpu(), std=std.detach().cpu(),
                                                    ucb=ucb.detach().cpu(), beta_t=beta_t, rule="ucb")

        X, Theta, y = self._get_stacked_history(device=self.model.device)
        mu, std, _ = self.model.posterior_on_candidates(
            X_candidates,
            theta_t.unsqueeze(0),
            X,
            Theta,
            y,
            calculate_covariance=False
        )
        beta_t = beta_finite_candidates(
            self.t,
            X_candidates.shape[0],
            delta
        ) if not deterministic else 0
        ucb = mu + (beta_t ** 0.5) * std
        idx = int(torch.argmax(ucb).item())
        return X_candidates[idx], SelectionInfo(mu=mu.cpu(), std=std.cpu(), ucb=ucb.cpu(), beta_t=beta_t, rule="ucb")

    @torch.no_grad()
    def select_ts(
        self,
        theta_t: torch.Tensor,
        X_candidates: torch.Tensor,
        deterministic: bool = False,
        seed: Optional[int] = None,
        temperature: float | bool = False,
    ) -> Tuple[torch.Tensor, SelectionInfo]:
        """Thompson sampling over a finite candidate set.

        For the current context theta_t and a finite set of candidate actions X_candidates,
        draw a sample from the joint posterior over {f(x; theta_t)} for x in X_candidates
        and pick the maximizer.
        """
        # Cold start: no data yet, pick a random arm
        if len(self.y_hist) == 0:
            idx = torch.randint(0, X_candidates.shape[0], (1,)).item()
            mu = torch.zeros(X_candidates.shape[0])
            std = torch.ones(X_candidates.shape[0])
            return X_candidates[idx], SelectionInfo(mu=mu, std=std, rule="ts")

        if self.posterior_type == "neural_linear":
            mu, std, Phi, Ainv = self._linear_posterior(X_candidates, theta_t)

            # Temperature / tempering: scale posterior sampling noise.
            # - temperature=False (default) -> no tempering
            # - temperature=True            -> use self.ts_temperature if present else 0.2
            # - temperature=float           -> use that value as tau
            if temperature is True:
                tau = float(getattr(self, "ts_temperature", 0.2))
            elif temperature is False or temperature is None:
                tau = 0.0
            else:
                tau = float(temperature)
            tau = float(max(0.0, min(1.0, tau)))

            # Deterministic default = greedy on posterior mean.
            # If deterministic=True AND tau>0, we do a *tempered* Thompson draw around the mean
            # using a fixed seed (so it is reproducible), then pick argmax.
            if deterministic and tau <= 0.0:
                sample = mu
            else:
                # sample weights: w ~ N(w_mean, sigma^2 A^{-1})
                device = mu.device
                w_mean, _ = self._w_mean_and_Ainv()
                sigma = float(self.model.noise.detach().cpu().item())
                L = torch.linalg.cholesky(add_jitter(Ainv, 1e-10))

                # If we are in deterministic+tempered mode, require a seed for reproducibility.
                # If none is provided, fall back to 0.
                if seed is None:
                    seed_use = 0
                else:
                    seed_use = int(seed)

                g = torch.Generator(device=device)
                g.manual_seed(seed_use)
                z = torch.randn(w_mean.shape[0], generator=g, device=device)

                # Tempered sampling: scale noise by tau (tau=1 -> standard TS)
                w_samp = w_mean + (tau if (deterministic or temperature) else 1.0) * sigma * (L @ z)
                sample = Phi @ w_samp

            idx = int(torch.argmax(sample).item())
            return X_candidates[idx], SelectionInfo(mu=mu.detach().cpu(), std=std.detach().cpu(),
                                                    sampled_vals=sample.detach().cpu(), rule="ts")

        # Stack history (cached)
        X, Theta, y = self._get_stacked_history(device=self.model.device)

        # Get posterior over candidates, including the covariance
        mu, std, cov = self.model.posterior_on_candidates(
            X_candidates,
            theta_t.unsqueeze(0),
            X,
            Theta,
            y,
            calculate_covariance=True,
        )

        mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        std = torch.nan_to_num(std, nan=0.0, posinf=0.0, neginf=0.0)

        # Temperature / tempering: scale posterior sampling noise.
        # - temperature=False (default) -> deterministic uses mean; stochastic uses standard TS
        # - temperature=True            -> use self.ts_temperature if present else 0.2
        # - temperature=float           -> use that value as tau
        if temperature is True:
            tau = float(getattr(self, "ts_temperature", 0.2))
        elif temperature is False or temperature is None:
            tau = 0.0
        else:
            tau = float(temperature)
        tau = float(max(0.0, min(1.0, tau)))

        # Deterministic default = greedy on posterior mean.
        # If deterministic=True AND tau>0, we do a *tempered* Thompson draw around the mean
        # using a fixed seed (so it is reproducible), then pick argmax.
        if deterministic and tau <= 0.0:
            sample = mu
        else:
            # Sample a function draw over the candidate set (tempered by tau)
            if seed is None:
                seed_use = 0
            else:
                seed_use = int(seed)

            g = torch.Generator(device=mu.device)
            g.manual_seed(seed_use)

            if cov is not None:
                Lc = self.model._robust_cholesky(cov, base_jitter=1e-10)
                z = torch.randn(mu.shape[0], generator=g, device=mu.device)
                # tau=1 -> standard TS; tau in (0,1) -> limited deviation around mean
                sample = mu + (tau if (deterministic or temperature) else 1.0) * (Lc @ z)
            else:
                z = torch.randn(mu.shape, generator=g, device=mu.device, dtype=mu.dtype)
                sample = mu + (tau if (deterministic or temperature) else 1.0) * (std * z)

        idx = int(torch.argmax(sample).item())
        return X_candidates[idx], SelectionInfo(mu=mu.cpu(), std=std.cpu(), sampled_vals=sample.cpu(), rule="ts")

    @torch.no_grad()
    def select_ucb_factored(
            self,
            theta_fields: torch.Tensor,  # (n_fields, d_theta_per_field)
            X_candidates_list: list[torch.Tensor],  # each (M_i, 1) or (M_i,)
            delta: float = 0.1,
            global_budget: float | None = None,
            max_budgets: torch.Tensor | None = None,
            deterministic: bool = False,
    ) -> tuple[torch.Tensor, list["SelectionInfo"] | dict]:
        """Factored UCB selection.

        Two modes:
          1) Independent per-field argmax (default): picks the best candidate per field.
          2) Budget-coupled (if self.use_farm_budget=True): solves a multi-choice knapsack
             to pick one candidate per field maximizing sum(UCB_i) subject to
             sum(applied_i) <= global_budget, where applied_i = max_budget_i - reduction_i.

        Notes
        -----
        - Each X_candidates_list[i] should contain the per-field reduction candidates (kg/ha).
        - This method computes posterior mean/std for ALL candidates per field (small grids).
        """
        if self.action_mode != "factored":
            raise ValueError("select_ucb_factored called but action_mode != 'factored'")

        if theta_fields.shape[0] != len(self.sub_bandits):
            raise ValueError("theta_fields first dimension must equal n_fields")

        n_fields = len(self.sub_bandits)
        device = theta_fields.device

        # sequence mode (Option A)
        self._seq_reset_if_needed()
        self._seq_push(theta_fields)
        theta_seq_fields = self._seq_build(device=device)  # (n_fields, T, d)
        self._last_theta_seq_fields = theta_seq_fields

        # --- helpers (defined inline to avoid extra dependencies) ---
        def _applied_costs_from_reductions(
                X_list: list[torch.Tensor],
                max_budgets_: torch.Tensor,
        ) -> list[torch.Tensor]:
            costs = []
            for i, Xi in enumerate(X_list):
                x = Xi.view(-1).to(dtype=torch.float32, device=device)
                max_i = max_budgets_[i].to(dtype=torch.float32, device=device)
                costs.append(max_i - x * 10)
            return costs

        def _solve_multi_choice_knapsack_dp(
                scores_list: list[torch.Tensor],
                costs_list: list[torch.Tensor],
                B: float,
                tick: float,
        ) -> tuple[list[int], float]:
            """DP multi-choice knapsack: pick 1 per group maximizing sum(scores) s.t. sum(cost)<=B."""
            if len(scores_list) == 0:
                return [], 0.0
            # discretize budget
            Bn = int(round(float(B) / float(tick)))
            NEG = -1e18
            dp = torch.full((Bn + 1,), NEG, dtype=torch.float64)
            dp[0] = 0.0

            choice = [[-1] * (Bn + 1) for _ in range(len(scores_list))]
            prev_b = [[-1] * (Bn + 1) for _ in range(len(scores_list))]

            for i in range(len(scores_list)):
                s = scores_list[i].detach().to(dtype=torch.float64).cpu()
                c = costs_list[i].detach().to(dtype=torch.float64).cpu()
                c_int = torch.round(c / float(tick)).to(dtype=torch.int64)

                new_dp = torch.full((Bn + 1,), NEG, dtype=torch.float64)
                for b in range(Bn + 1):
                    base = float(dp[b])
                    if base <= NEG / 2:
                        continue
                    for k in range(s.numel()):
                        nb = b + int(c_int[k])
                        if 0 <= nb <= Bn:
                            val = base + float(s[k])
                            if val > float(new_dp[nb]):
                                new_dp[nb] = val
                                choice[i][nb] = int(k)
                                prev_b[i][nb] = int(b)
                dp = new_dp

            best_b = int(torch.argmax(dp).item())

            picks = [0] * len(scores_list)
            b = best_b
            for i in range(len(scores_list) - 1, -1, -1):
                k = choice[i][b]
                if k < 0:
                    # infeasible DP path -> greedy fallback
                    picks = [int(torch.argmax(scores_list[j]).item()) for j in range(len(scores_list))]
                    total_cost = float(torch.stack([costs_list[j][picks[j]] for j in range(len(scores_list))]).sum().item())
                    return picks, total_cost
                picks[i] = k
                b = prev_b[i][b]

            total_cost = float(torch.stack([costs_list[i][picks[i]] for i in range(len(scores_list))]).sum().item())
            return picks, total_cost

        # --- compute per-field posterior and UCB scores ---
        infos: list["SelectionInfo"] = []
        scores_list: list[torch.Tensor] = []

        # Important: sub-bandits maintain their own histories and devices
        for i, b in enumerate(self.sub_bandits):
            Xi = X_candidates_list[i]
            Xi = Xi.to(dtype=torch.float32, device=b.model.device)
            th_raw = theta_seq_fields[i]  # (T, d)
            th_i = th_raw.to(dtype=torch.float32, device=b.model.device).unsqueeze(0)  # (1, T, d)

            if len(b.y_hist) == 0:
                # cold-start for this field: pretend mu=0, std=1
                mu_i = torch.zeros(Xi.shape[0], device=b.model.device, dtype=torch.float32)
                std_i = torch.ones(Xi.shape[0], device=b.model.device, dtype=torch.float32)
                beta_t = 0.0 if deterministic else beta_finite_candidates(self.t, Xi.shape[0], delta)
                ucb_i = mu_i + (beta_t ** 0.5) * std_i
                infos.append(SelectionInfo(mu=mu_i.detach().cpu(), std=std_i.detach().cpu(), ucb=ucb_i.detach().cpu(), beta_t=beta_t, rule="ucb"))
                scores_list.append(ucb_i.to(device))
                continue

            if getattr(b, "posterior_type", None) == "neural_linear":
                mu_i, std_i, _, _ = b._linear_posterior(Xi, th_i.squeeze(0))
                beta_t = 0.0 if deterministic else beta_finite_candidates(self.t, Xi.shape[0], delta)
                ucb_i = mu_i + (beta_t ** 0.5) * std_i
                infos.append(SelectionInfo(mu=mu_i.detach().cpu(), std=std_i.detach().cpu(), ucb=ucb_i.detach().cpu(), beta_t=beta_t, rule="ucb"))
                scores_list.append(ucb_i.to(device))
            else:
                X_hist, Theta_hist, y_hist = b._get_stacked_history(device=b.model.device)
                mu_i, std_i, _ = b.model.posterior_on_candidates(
                    Xi,
                    th_i,
                    X_hist,
                    Theta_hist,
                    y_hist,
                    calculate_covariance=False,
                )
                beta_t = 0.0 if deterministic else beta_finite_candidates(self.t, Xi.shape[0], delta)
                ucb_i = mu_i + (beta_t ** 0.5) * std_i
                infos.append(SelectionInfo(mu=mu_i.detach().cpu(), std=std_i.detach().cpu(), ucb=ucb_i.detach().cpu(), beta_t=beta_t, rule="ucb"))
                scores_list.append(ucb_i.to(device))

        # --- choose one candidate per field ---
        if getattr(self, "use_farm_budget", False):
            if global_budget is None or max_budgets is None:
                raise ValueError("use_farm_budget=True requires global_budget and max_budgets")

            tick = float(getattr(self, "budget_tick", 0.5))
            slack = float(getattr(self, "budget_slack", 1e-6))

            max_budgets = max_budgets.to(dtype=torch.float32, device=device).view(-1)
            # Fast-path: unconstrained budget => skip DP.
            # applied_i = max_budget_i - reduction_i, so max feasible applied is sum(max_budgets).
            B_max = float(max_budgets.sum().item())
            slack = float(getattr(self, "budget_slack", 1e-6))  # ensure slack exists here

            if float(global_budget) >= (B_max - slack):
                xs = []
                picks = []
                for i, ucb_i in enumerate(scores_list):
                    idx = int(torch.argmax(ucb_i).item())
                    picks.append(idx)
                    xs.append(float(torch.as_tensor(X_candidates_list[i]).view(-1)[idx].item()))

                x_vec = torch.tensor(xs, dtype=torch.get_default_dtype(), device=device)
                self.t += 1
                return x_vec, {
                    "rule": "ucb_factored_unconstrained",
                    "picks": picks,
                    "total_applied": B_max,
                    "infos": infos,
                }
            costs_list = _applied_costs_from_reductions(X_candidates_list, max_budgets)

            picks, total_applied = _solve_multi_choice_knapsack_dp(
                scores_list=scores_list,
                costs_list=costs_list,
                B=float(global_budget) + slack,
                tick=tick,
            )

            xs = [float(torch.as_tensor(X_candidates_list[i]).view(-1)[picks[i]].item()) for i in range(n_fields)]
            x_vec = torch.tensor(xs, dtype=torch.get_default_dtype(), device=device)
            self.t += 1
            self._ctx_step += 1
            return x_vec, {"rule": "ucb_factored_budget", "picks": picks, "total_applied": float(total_applied), "infos": infos}

        # Independent per-field
        xs = []
        for i, ucb_i in enumerate(scores_list):
            idx = int(torch.argmax(ucb_i).item())
            xs.append(float(torch.as_tensor(X_candidates_list[i]).view(-1)[idx].item()))

        x_vec = torch.tensor(xs, dtype=torch.get_default_dtype(), device=device)
        self.t += 1
        return x_vec, infos

    @torch.no_grad()
    def select_ts_factored(
            self,
            theta_fields: torch.Tensor,
            X_candidates_list: list[torch.Tensor],
            global_budget: float | None = None,
            max_budgets: torch.Tensor | None = None,
            deterministic: bool = False,
            seed: "Optional[int]" = None,
            temperature: float | bool = False,
    ) -> tuple[torch.Tensor, list["SelectionInfo"] | dict]:
        """Factored Thompson sampling.

        Two modes:
          1) Independent per-field argmax of sampled values (default).
          2) Budget-coupled (if self.use_farm_budget=True): multi-choice knapsack with
             sampled scores and applied-cost constraint.
        """
        if self.action_mode != "factored":
            raise ValueError("select_ts_factored called but action_mode != 'factored'")

        if theta_fields.shape[0] != len(self.sub_bandits):
            raise ValueError("theta_fields first dimension must equal n_fields")

        n_fields = len(self.sub_bandits)
        device = theta_fields.device

        # sequence mode (Option A)
        self._seq_reset_if_needed()
        self._seq_push(theta_fields)
        theta_seq_fields = self._seq_build(device=device)  # (n_fields, T, d)
        self._last_theta_seq_fields = theta_seq_fields

        # Temperature / tempering: scale posterior sampling noise.
        # - temperature=False (default) -> stochastic TS uses full noise scale=1.0; deterministic uses mean.
        # - temperature=True            -> use self.ts_temperature if present else 0.2
        # - temperature=float           -> use that value as tau
        if temperature is True:
            tau = float(getattr(self, "ts_temperature", 0.2))
        elif temperature is False or temperature is None:
            tau = 0.0
        else:
            tau = float(temperature)
        tau = float(max(0.0, min(1.0, tau)))

        # Noise scale used when we DO sample:
        # - if temperature is not provided (False/None): use 1.0 (standard TS)
        # - otherwise: use tau (tempered TS)
        if temperature is False or temperature is None:
            noise_scale = 1.0
        else:
            noise_scale = tau

        # --- helpers ---
        def _applied_costs_from_reductions(
                X_list: list[torch.Tensor],
                max_budgets_: torch.Tensor,
        ) -> list[torch.Tensor]:
            costs = []
            for i, Xi in enumerate(X_list):
                x = Xi.view(-1).to(dtype=torch.float32, device=device)
                max_i = max_budgets_[i].to(dtype=torch.float32, device=device)
                costs.append(max_i - x * 10)
            return costs

        def _solve_multi_choice_knapsack_dp(
                scores_list: list[torch.Tensor],
                costs_list: list[torch.Tensor],
                B: float,
                tick: float,
        ) -> tuple[list[int], float]:
            if len(scores_list) == 0:
                return [], 0.0
            Bn = int(round(float(B) / float(tick)))
            NEG = -1e18
            dp = torch.full((Bn + 1,), NEG, dtype=torch.float64)
            dp[0] = 0.0

            choice = [[-1] * (Bn + 1) for _ in range(len(scores_list))]
            prev_b = [[-1] * (Bn + 1) for _ in range(len(scores_list))]

            for i in range(len(scores_list)):
                s = scores_list[i].detach().to(dtype=torch.float64).cpu()
                c = costs_list[i].detach().to(dtype=torch.float64).cpu()
                c_int = torch.round(c / float(tick)).to(dtype=torch.int64)

                new_dp = torch.full((Bn + 1,), NEG, dtype=torch.float64)
                for b in range(Bn + 1):
                    base = float(dp[b])
                    if base <= NEG / 2:
                        continue
                    for k in range(s.numel()):
                        nb = b + int(c_int[k])
                        if 0 <= nb <= Bn:
                            val = base + float(s[k])
                            if val > float(new_dp[nb]):
                                new_dp[nb] = val
                                choice[i][nb] = int(k)
                                prev_b[i][nb] = int(b)
                dp = new_dp

            best_b = int(torch.argmax(dp).item())

            picks = [0] * len(scores_list)
            b = best_b
            for i in range(len(scores_list) - 1, -1, -1):
                k = choice[i][b]
                if k < 0:
                    picks = [int(torch.argmax(scores_list[j]).item()) for j in range(len(scores_list))]
                    total_cost = float(torch.stack([costs_list[j][picks[j]] for j in range(len(scores_list))]).sum().item())
                    return picks, total_cost
                picks[i] = k
                b = prev_b[i][b]

            total_cost = float(torch.stack([costs_list[i][picks[i]] for i in range(len(scores_list))]).sum().item())
            return picks, total_cost

        infos: list["SelectionInfo"] = []
        sampled_scores: list[torch.Tensor] = []

        # RNG for TS
        # If `seed` is provided, TS becomes reproducible across runs.
        # Use per-field generators so results do not depend on loop ordering.
        base_seed = None if seed is None else int(seed)

        def _make_gen(field_idx: int):
            if base_seed is None:
                return None
            g = torch.Generator(device=device)
            # decorrelate fields while staying deterministic
            g.manual_seed(base_seed + field_idx * 10007)
            return g

        for i, b in enumerate(self.sub_bandits):
            Xi = X_candidates_list[i]
            Xi = Xi.to(dtype=torch.float32, device=b.model.device)
            th_raw = theta_seq_fields[i]  # (T, d)
            th_i = th_raw.to(dtype=torch.float32, device=b.model.device).unsqueeze(0)  # (1, T, d)

            if len(b.y_hist) == 0:
                mu_i = torch.zeros(Xi.shape[0], device=b.model.device, dtype=torch.float32)
                std_i = torch.ones(Xi.shape[0], device=b.model.device, dtype=torch.float32)
            else:
                if getattr(b, "posterior_type", None) == "neural_linear":
                    mu_i, std_i, _, _ = b._linear_posterior(Xi, th_i.squeeze(0))
                else:
                    X_hist, Theta_hist, y_hist = b._get_stacked_history(device=b.model.device)
                    mu_i, std_i, _ = b.model.posterior_on_candidates(
                        Xi,
                        th_i,
                        X_hist,
                        Theta_hist,
                        y_hist,
                        calculate_covariance=False,
                    )

            mu_i = torch.nan_to_num(mu_i, nan=0.0, posinf=0.0, neginf=0.0)
            std_i = torch.nan_to_num(std_i, nan=0.0, posinf=0.0, neginf=0.0)

            # Deterministic default: greedy on posterior mean.
            # If deterministic=True AND tau>0 (temperature enabled), do a reproducible tempered draw around mean.
            if deterministic and tau <= 0.0:
                samp_i = mu_i
            else:
                g_i = _make_gen(i)
                if g_i is not None:
                    z = torch.randn(mu_i.shape, device=mu_i.device, dtype=mu_i.dtype, generator=g_i)
                else:
                    z = torch.randn(mu_i.shape, device=mu_i.device, dtype=mu_i.dtype)
                # Standard TS when temperature is False/None (noise_scale=1.0);
                # tempered TS otherwise (noise_scale=tau in (0,1]).
                samp_i = mu_i + (noise_scale * std_i) * z

            infos.append(SelectionInfo(mu=mu_i.detach().cpu(), std=std_i.detach().cpu(), sampled_vals=samp_i.detach().cpu(), rule="ts"))
            sampled_scores.append(samp_i.to(device))

        if getattr(self, "use_farm_budget", False):
            if global_budget is None or max_budgets is None:
                raise ValueError("use_farm_budget=True requires global_budget and max_budgets")

            tick = float(getattr(self, "budget_tick", 0.5))
            slack = float(getattr(self, "budget_slack", 1e-6))

            max_budgets = max_budgets.to(dtype=torch.float32, device=device).view(-1)
            # Fast-path: unconstrained budget => skip DP.
            B_max = float(max_budgets.sum().item())
            slack = float(getattr(self, "budget_slack", 1e-6))  # ensure slack exists here

            if float(global_budget) >= (B_max - slack):
                xs = []
                picks = []
                for i, s_i in enumerate(sampled_scores):  # scores_list holds the sampled/TS scores per candidate
                    idx = int(torch.argmax(s_i).item())
                    picks.append(idx)
                    xs.append(float(torch.as_tensor(X_candidates_list[i]).view(-1)[idx].item()))

                x_vec = torch.tensor(xs, dtype=torch.get_default_dtype(), device=device)
                self.t += 1
                return x_vec, {
                    "rule": "ts_factored_unconstrained",
                    "picks": picks,
                    "total_applied": B_max,
                    "infos": infos,
                }
            costs_list = _applied_costs_from_reductions(X_candidates_list, max_budgets)

            picks, total_applied = _solve_multi_choice_knapsack_dp(
                scores_list=sampled_scores,
                costs_list=costs_list,
                B=float(global_budget) + slack,
                tick=tick,
            )

            xs = [float(torch.as_tensor(X_candidates_list[i]).view(-1)[picks[i]].item()) for i in range(n_fields)]
            x_vec = torch.tensor(xs, dtype=torch.get_default_dtype(), device=device)
            self.t += 1
            self._ctx_step += 1
            return x_vec, {"rule": "ts_factored_budget", "picks": picks, "total_applied": float(total_applied), "infos": infos}

        xs = []
        for i, samp_i in enumerate(sampled_scores):
            idx = int(torch.argmax(samp_i).item())
            xs.append(float(torch.as_tensor(X_candidates_list[i]).view(-1)[idx].item()))

        x_vec = torch.tensor(xs, dtype=torch.get_default_dtype(), device=device)
        self.t += 1
        return x_vec, infos

    def update_factored(
            self,
            theta_fields: torch.Tensor,  # (n_fields, d) OR (n_fields, T, d)
            x_vec: torch.Tensor,         # (n_fields,)
            y_fields: np.ndarray | torch.Tensor  # (n_fields,)
    ):
        if self.action_mode != "factored":
            raise ValueError("update_factored called but action_mode != 'factored'")

        # y_fields -> tensor
        if isinstance(y_fields, np.ndarray):
            y_fields = torch.from_numpy(y_fields.astype(np.float32))
        y_fields = y_fields.view(-1)

        # theta_fields -> tensor
        if not torch.is_tensor(theta_fields):
            theta_fields = torch.as_tensor(theta_fields, dtype=torch.float32)

        if len(self.sub_bandits) != theta_fields.shape[0] or len(self.sub_bandits) != y_fields.shape[0]:
            raise ValueError("Mismatch: n_fields between sub_bandits / theta_fields / y_fields")

        # Support both MLP g(θ) and LSTM g(θ)
        # - MLP expects (d,)
        # - LSTM expects (T, d) (or (1, T, d) at call sites)
        # Here we normalize each field context to (T, d) and store it for later.
        self._last_theta_seq_fields = []
        for i in range(theta_fields.shape[0]):
            th_i = theta_fields[i]
            if th_i.dim() == 1:
                th_i = th_i.unsqueeze(0)  # (1, d)
            elif th_i.dim() != 2:
                raise ValueError(
                    f"theta_fields[{i}] must be 1D (d,) or 2D (T,d), got shape {tuple(th_i.shape)}"
                )
            self._last_theta_seq_fields.append(th_i.detach().cpu())  # keep a cheap CPU copy

        # Ensure x_vec tensor
        if not torch.is_tensor(x_vec):
            x_vec = torch.as_tensor(x_vec, dtype=torch.get_default_dtype())
        x_vec = x_vec.view(-1)

        for i, b in enumerate(self.sub_bandits):
            device = b.model.device
            th_i = self._last_theta_seq_fields[i].to(device=device, dtype=torch.float32)  # (T, d)
            x_i = torch.tensor([float(x_vec[i].item())], device=device, dtype=torch.get_default_dtype())
            b.update(th_i, x_i, float(y_fields[i].item()))

    def _coreset_maybe_replace_gp(self, z_new: torch.Tensor) -> int | None:
        """For exact-GP mode: decide whether to replace an existing coreset point.

        Returns the index to replace (0..K-1) or None to discard.
        Strategy:
          - fifo: always replace oldest (deque maxlen handles it)
          - diverse: keep points diverse in (x,theta) space by replacing the nearest point
                    only if the new point is sufficiently far from the set.
        """
        if self.coreset_mode == "fifo":
            return None

        # diverse mode
        if len(self._z_gp_hist) == 0:
            return None

        Z = torch.vstack(tuple(self._z_gp_hist))  # (K, d_x+d_theta)
        # compute distance to nearest existing point
        d2 = ((Z - z_new) ** 2).sum(dim=1)
        j = int(torch.argmin(d2).item())
        min_dist = float(torch.sqrt(d2[j]).detach().cpu())

        # adaptive threshold: median nearest-neighbor distance within current coreset
        # (cheap approx using distances to the mean)
        center = Z.mean(dim=0, keepdim=True)
        spread = torch.sqrt(((Z - center) ** 2).sum(dim=1)).median()
        thresh = float(spread.detach().cpu()) * 0.25  # keep if "meaningfully" far

        if min_dist > thresh:
            return j
        return None

    # ---- log an observation and advance time
    def update(self, theta_t: torch.Tensor, x_t: torch.Tensor, y_t: float):
        device = self.model.device

        theta_t = theta_t.detach().to(device)
        x_t = x_t.detach().to(device)
        y_val = torch.tensor(float(y_t), device=device, dtype=torch.get_default_dtype())

        # For LSTM g(θ), theta_t may be (T, d). For coreset bookkeeping we need a 1D
        # vector to concatenate with x_t, so we summarize theta_t with the LAST step.
        # (This does NOT change what gets stored in theta_hist for GP training.)
        if theta_t.dim() == 2:
            theta_z = theta_t[-1]  # (d,)
        elif theta_t.dim() == 3 and theta_t.shape[0] == 1:
            # Be tolerant if a caller passes (1, T, d)
            theta_z = theta_t[0, -1]
        else:
            theta_z = theta_t

        x_z = x_t.view(-1)
        theta_z = theta_z.view(-1)

        if self.posterior_type == "neural_linear":
            # update sufficient stats
            phi = self._phi(x_t, theta_t)  # (m,)
            if phi.dim() == 1:
                phi_col = phi.unsqueeze(1)  # (m,1)
            else:
                phi_col = phi.T

            self._A = self._A + (phi_col @ phi_col.T)
            self._b = self._b + phi * y_val

            # optional training buffer for representation learning
            z = torch.cat([x_z, theta_z], dim=0)
            self._z_hist.append(z.unsqueeze(0))
            self._yl_hist.append(y_val.unsqueeze(0))

            self.t += 1
            return

        # Default GP path
        # Default GP path (exact GP on a bounded coreset)
        z_new = torch.cat([x_z, theta_z], dim=0)

        if len(self.y_hist) < self.coreset_size:
            # still filling
            self.theta_hist.append(theta_t.unsqueeze(0))
            self.x_hist.append(x_t.unsqueeze(0))
            self.y_hist.append(torch.tensor([float(y_t)], dtype=torch.get_default_dtype(), device=device))
            self._z_gp_hist.append(z_new.unsqueeze(0))
        else:
            # coreset is full
            j = self._coreset_maybe_replace_gp(z_new)
            if j is None:
                # fifo mode: deque maxlen will drop oldest if we append
                if self.coreset_mode == "fifo":
                    self.theta_hist.append(theta_t.unsqueeze(0))
                    self.x_hist.append(x_t.unsqueeze(0))
                    self.y_hist.append(torch.tensor([float(y_t)], dtype=torch.get_default_dtype(), device=device))
                    self._z_gp_hist.append(z_new.unsqueeze(0))
                # diverse mode: discard
            else:
                # replace point j in-place (convert deques to list, replace, then rebuild deques)
                th = list(self.theta_hist)
                xx = list(self.x_hist)
                yy = list(self.y_hist)
                zz = list(self._z_gp_hist)

                th[j] = theta_t.unsqueeze(0)
                xx[j] = x_t.unsqueeze(0)
                yy[j] = torch.tensor([float(y_t)], dtype=torch.get_default_dtype(), device=device)
                zz[j] = z_new.unsqueeze(0)

                self.theta_hist = deque(th, maxlen=self.coreset_size)
                self.x_hist = deque(xx, maxlen=self.coreset_size)
                self.y_hist = deque(yy, maxlen=self.coreset_size)
                self._z_gp_hist = deque(zz, maxlen=self.coreset_size)
        # coreset changed -> clear cached Cholesky/alpha
        self.model._clear_cache()
        self.t += 1
        self._stack_cache_dirty = True

    def export_posterior(self):
        if getattr(self, "action_mode", "joint") == "factored":
            return {
                "action_mode": "factored",
                "n_fields": int(self.n_fields or len(self.sub_bandits)),
                "sub": [b.export_posterior() for b in self.sub_bandits],
                "t": int(getattr(self, "t", 1)),
            }
        """Return everything needed for posterior predictions."""
        X = torch.vstack(tuple(self.x_hist)) if self.x_hist else torch.empty(0, self.model.mogp.k_shared[
            0].raw_lengthscale.numel())
        Theta = torch.vstack(tuple(self.theta_hist)) if self.theta_hist else torch.empty(0,
                                                                                  self.model.g_net.net[0].in_features)
        y = torch.hstack(tuple(self.y_hist)) if self.y_hist else torch.empty(0)
        cache = None
        if self.model._train_cache:
            # optional speed-up: store Cholesky and alpha so we don’t recompute
            cache = {
                "L": self.model._train_cache["L"].cpu(),
                "alpha": self.model._train_cache["alpha"].cpu(),
                "G": self.model._train_cache["G"].cpu(),
            }

        nl = None
        if self.posterior_type == "neural_linear":
            nl = {
                "posterior_type": "neural_linear",
                "phi_state": self.phi_net.state_dict(),
                "A": self._A.detach().cpu(),
                "b": self._b.detach().cpu(),
                "ridge_lambda": self.ridge_lambda,
            }
        return {
            "model_state": self.model.state_dict(),
            "X": X.cpu(),
            "Theta": Theta.cpu(),
            "y": y.cpu(),
            "cache": cache,
            "neural_linear": nl,
        }

    def import_posterior(self, blob: dict, map_location="cpu"):
        if blob.get("action_mode") == "factored":
            subs = blob["sub"]
            n_fields = int(blob.get("n_fields", len(subs)))
            if len(subs) == 0:
                raise ValueError("Cannot import factored posterior: empty sub list")

            # infer d_theta_per_field from sub[0]["Theta"] if possible
            first = subs[0]
            Theta0 = first.get("Theta", None)
            if Theta0 is None or (hasattr(Theta0, "numel") and Theta0.numel() == 0):
                raise ValueError(
                    "Cannot infer d_theta_per_field from saved posterior because Theta is empty. "
                    "Store d_theta_per_field explicitly in the blob if you expect empty history."
                )
            d_theta_per_field = int(Theta0.shape[1])

            self.action_mode = "factored"
            self.n_fields = n_fields
            self.sub_bandits = [
                NNAGPBandit(
                    d_theta=d_theta_per_field,
                    d_x=1,
                    m=8,
                    Q=1,
                    device=torch.device(map_location),
                    action_mode="joint",
                )
                for _ in range(n_fields)
            ]
            for b, sb in zip(self.sub_bandits, subs):
                b.import_posterior(sb, map_location=map_location)
            self.t = int(blob.get("t", 1))
            return
        self.model.load_state_dict(blob["model_state"])
        self.x_hist = [blob["X"].to(map_location)]
        self.theta_hist = [blob["Theta"].to(map_location)]
        self.y_hist = [blob["y"].to(map_location)]
        # rebuild cache (optional)
        self.model._clear_cache()

        nl = blob.get("neural_linear")
        if nl and nl.get("posterior_type") == "neural_linear":
            self.posterior_type = "neural_linear"
            self.phi_net.load_state_dict(nl["phi_state"])
            self._A = nl["A"].to(map_location)
            self._b = nl["b"].to(map_location)
            self.ridge_lambda = float(nl.get("ridge_lambda", 1.0))

        if blob.get("cache"):
            c = blob["cache"]
            self.model._train_cache = {
                "X": blob["X"].to(map_location),
                "Theta": blob["Theta"].to(map_location),
                "y": blob["y"].to(map_location),
                "L": c["L"].to(map_location),
                "alpha": c["alpha"].to(map_location),
                "G": c["G"].to(map_location),
            }

    # Save and load learned bandits
    def save(
            self,
            seed: int = None,
            t=None,
            name: str = None,
            args = None,
            rms = None,
            farm_id: str = None,
            method: str = "ucb",
    ):
        if farm_id is None:
            file_dir = os.path.join(_DEFAULT_LOGDIR, name)
        else:
            file_dir = os.path.join(_DEFAULT_LOGDIR, name, farm_id)
        os.makedirs(file_dir, exist_ok=True)
        name_file = f"s{seed}_{method}_model{t if t is not None else ''}.pth"
        file_dir = os.path.join(
                file_dir,
                name_file,
        )

        dict_to_save: dict = self.export_posterior()
        dict_to_save["context_rms"] = rms
        dict_to_save["args"] = args

        torch.save(
            dict_to_save,
            file_dir,
        )
        return file_dir

    def load(self, load_dir: str = None):
        state = torch.load(
            os.path.join(
                _DEFAULT_MODEL_DIR,
                load_dir
            )
        )
        self.import_posterior(state)
        return state

    def _get_stacked_history(self, device: torch.device | None = None):
        """Return stacked (X, Theta, y) cached until history changes.

        Important: this keeps tensor storage stable across calls so NNAGP.predict()
        can reuse its cached Cholesky instead of refitting every selection.
        """
        device = device or self.model.device

        # Detect whether g_net expects sequences (LSTMFeatureNet) or flat vectors (FeatureNet)
        g_net = getattr(self.model, "g_net", None)
        use_lstm = bool(getattr(self, "use_lstm", False))
        if g_net is not None:
            use_lstm = use_lstm or (hasattr(g_net, "lstm") and isinstance(getattr(g_net, "lstm"), torch.nn.LSTM))

        if (not self._stack_cache_dirty) and self._stack_cache:
            X = self._stack_cache["X"]
            Theta = self._stack_cache["Theta"]
            y = self._stack_cache["y"]

            if X.device != device:
                X = X.to(device)
                Theta = Theta.to(device)
                y = y.to(device)
                self._stack_cache["X"] = X
                self._stack_cache["Theta"] = Theta
                self._stack_cache["y"] = y

            return X, Theta, y

        # Rebuild stacked history
        X = torch.vstack(tuple(self.x_hist)).to(device)
        y = torch.hstack(tuple(self.y_hist)).to(device)

        if not use_lstm:
            # Flat theta: elements typically (1,d) => vstack => (N,d)
            Theta = torch.vstack(tuple(self.theta_hist)).to(device)
        else:
            # Sequence theta: normalize to (1,T,d), then pad to common T and cat => (N,T,d)
            seqs = []
            max_T = 1
            d = None

            for th in self.theta_hist:
                th = th.to(device)

                if th.dim() == 2:
                    # (1,d) -> (1,1,d)
                    th = th.unsqueeze(1)
                elif th.dim() != 3:
                    raise ValueError(f"theta_hist element must be 2D (1,d) or 3D (1,T,d); got {tuple(th.shape)}")

                if d is None:
                    d = int(th.shape[-1])
                elif int(th.shape[-1]) != d:
                    raise ValueError(f"Inconsistent d in theta_hist: expected {d}, got {int(th.shape[-1])}")

                max_T = max(max_T, int(th.shape[1]))
                seqs.append(th)

            # Pad by repeating the last step (keeps scale stable vs zero-padding)
            padded = []
            for th in seqs:
                Ti = int(th.shape[1])
                if Ti < max_T:
                    last = th[:, -1:, :].expand(-1, max_T - Ti, -1)
                    th = torch.cat([th, last], dim=1)
                padded.append(th)

            Theta = torch.cat(padded, dim=0)  # (N,max_T,d)

        self._stack_cache = {"X": X, "Theta": Theta, "y": y}
        self._stack_cache_dirty = False
        return X, Theta, y

    @staticmethod
    def _applied_costs_from_reductions(
            X_list: list[torch.Tensor],
            max_budgets: torch.Tensor,
    ) -> list[torch.Tensor]:
        """cost_i(x) = applied_N = max_budget_i - reduction."""
        costs = []
        for i, Xi in enumerate(X_list):
            x = Xi.view(-1)  # (M_i,)
            max_i = max_budgets[i].to(dtype=x.dtype, device=x.device)
            costs.append(max_i - x)
        return costs

    @staticmethod
    def _solve_multi_choice_knapsack_dp(
            scores_list: list[torch.Tensor],
            costs_list: list[torch.Tensor],
            B: float,
            tick: float,
    ) -> tuple[list[int], float]:
        """
        Pick exactly one item per group i maximizing sum(scores) s.t. sum(costs) <= B.

        scores_list[i]: (M_i,)
        costs_list[i]:  (M_i,)
        Returns: (picked_indices, total_cost)
        """
        n = len(scores_list)
        if n == 0:
            return [], 0.0

        Bn = int(round(float(B) / float(tick)))
        NEG = -1e18

        # dp[b] = best score after processing i groups
        dp = torch.full((Bn + 1,), NEG, dtype=torch.float64)
        dp[0] = 0.0

        choice = [[-1] * (Bn + 1) for _ in range(n)]
        prev_b = [[-1] * (Bn + 1) for _ in range(n)]

        for i in range(n):
            s = scores_list[i].detach().to(dtype=torch.float64).cpu()
            c = costs_list[i].detach().to(dtype=torch.float64).cpu()
            c_int = torch.round(c / float(tick)).to(dtype=torch.int64)

            new_dp = torch.full((Bn + 1,), NEG, dtype=torch.float64)

            for b in range(Bn + 1):
                base = float(dp[b])
                if base <= NEG / 2:
                    continue
                for k in range(s.numel()):
                    nb = b + int(c_int[k])
                    if 0 <= nb <= Bn:
                        val = base + float(s[k])
                        if val > float(new_dp[nb]):
                            new_dp[nb] = val
                            choice[i][nb] = int(k)
                            prev_b[i][nb] = int(b)

            dp = new_dp

        best_b = int(torch.argmax(dp).item())

        picks = [0] * n
        b = best_b
        for i in range(n - 1, -1, -1):
            k = choice[i][b]
            if k < 0:
                # no feasible solution -> greedy fallback (may violate if B is impossible)
                picks = [int(torch.argmax(scores_list[j]).item()) for j in range(n)]
                total_cost = float(torch.stack([costs_list[j][picks[j]] for j in range(n)]).sum().item())
                return picks, total_cost
            picks[i] = k
            b = prev_b[i][b]

        total_cost = float(torch.stack([costs_list[i][picks[i]] for i in range(n)]).sum().item())
        return picks, total_cost

