import datetime
import os
import math
from dataclasses import dataclass
from collections import deque
from typing import List, Tuple, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from cropgymzoo import _DEFAULT_MODEL_DIR, _DEFAULT_LOGDIR


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

    def __init__(self, d_theta: int, m: int, hidden: int = 256, depth: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = d_theta
        for d in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            in_dim = hidden
        layers += [nn.Linear(in_dim, m)]
        self.net = nn.Sequential(*layers)

    def forward(self, Theta: torch.Tensor) -> torch.Tensor:
        # (n,d_theta) -> (n,m)
        return self.net(Theta)

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

    def forward(self, Theta: torch.Tensor) -> torch.Tensor:
        """
        Theta: (N, dθ) or (N, T, dθ)
        Returns: g ∈ (N, m), using the last time step (or last hidden).
        """
        if Theta.dim() == 2:
            Theta = Theta.unsqueeze(1)  # (N, 1, dθ)

        # LSTM
        _, (hn, _) = self.lstm(Theta)  # hn: (layers*dirs, N, hidden)
        layers = self.num_layers * self.num_directions
        hn = hn.view(layers, Theta.size(0), self.hidden)
        last = hn[-1]  # (N, hidden) -> top layer, forward (if bidir, torch uses concat internally in hn)
        return self.proj(last)

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
    def __init__(self, d_theta: int, d_x: int, m: int, Q: int = 1, kernel='matern', device: Optional[torch.device] = None):
        super().__init__()
        self.g_net = FeatureNet(d_theta, m)
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
        act, context_in, y = act.to(self.device), context_in.to(self.device), y.to(self.device)
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
        self._train_cache = {"Theta": context_in, "X": act, "y": y, "L": L.detach(), "alpha": alpha.detach(), "G": G.detach()}
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
        # returns μ_c (M,), σ_c (M,), and (optional) full covariance Σ_c (M,M)
        mu, std = self.predict(Xc, theta.expand(Xc.shape[0], -1), train_X, train_Theta, y)
        if not calculate_covariance:
            return mu, std, None
        # Build covariance only when needed (used for TS)
        # Σ = K_cand - K_*^T (K+σ^2I)^{-1} K_*
        G_tr = self._train_cache["G"]
        G_c = self.g_net(theta.expand(Xc.shape[0], -1))
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
    ):
        self.posterior_type = posterior_type
        self.model = NNAGP(d_theta, d_x, m=m, Q=Q, device=device or torch.device("cpu"))

        self.coreset_size = int(coreset_size)
        self.coreset_mode = coreset_mode

        self.theta_hist: deque[torch.Tensor] = deque(maxlen=self.coreset_size)
        self.x_hist: deque[torch.Tensor] = deque(maxlen=self.coreset_size)
        self.y_hist: deque[torch.Tensor] = deque(maxlen=self.coreset_size)
        # store concatenated z=[x,theta] for coreset maintenance (exact GP path)
        self._z_gp_hist: deque[torch.Tensor] = deque(maxlen=self.coreset_size)
        self.t = 1

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
        theta_rep = theta_t.unsqueeze(0).expand(X_candidates.shape[0], -1)
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
        x = torch.vstack(tuple(self.x_hist))
        theta = torch.vstack(tuple(self.theta_hist))
        y = torch.hstack(tuple(self.y_hist))
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

        X = torch.vstack(tuple(self.x_hist))
        Theta = torch.vstack(tuple(self.theta_hist))
        y = torch.hstack(tuple(self.y_hist))
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

    # Use sparsely!
    @torch.no_grad()
    def select_ucb_streaming(
            self,
            theta_t,
            all_actions,
            chunk=16000,
            delta=0.1,
            deterministic=False,
    )->Tuple[torch.Tensor, dict | None]:
        """
        all_actions: (M, d_x) tensor (can be on disk-mapped or memmap if huge)
        Returns the best action under UCB without ever materializing all M scores.
        """
        # pull the posterior data once
        if not self.y_hist:
            # cold start: pick random
            idx = torch.randint(0, all_actions.shape[0], (1,)).item()
            return all_actions[idx], None

        X = torch.vstack(tuple(self.x_hist))
        Theta = torch.vstack(tuple(self.theta_hist))
        y = torch.hstack(tuple(self.y_hist))
        beta_t = beta_finite_candidates(
            self.t,
            chunk, # use chunk size conservatively
            delta
        )  if not deterministic else 0

        best_ucb = -float("inf")
        best_x = None

        for i in range(0, all_actions.shape[0], chunk):
            Xc = all_actions[i:i + chunk]
            # only need mean & std; DO NOT build candidate covariance
            mu, std, _ = self.model.posterior_on_candidates(Xc, theta_t.unsqueeze(0), X, Theta, y, calculate_covariance=False)
            ucb = mu + (beta_t ** 0.5) * std
            j = int(torch.argmax(ucb))
            if ucb[j].item() > best_ucb:
                best_ucb = ucb[j].item()
                best_x = Xc[j].clone()

        return best_x, {"best_ucb": best_ucb, "beta_t": beta_t}

    @torch.no_grad()
    def select_ts(
        self,
        theta_t: torch.Tensor,
        X_candidates: torch.Tensor,
        deterministic: bool = False,
        seed: Optional[int] = None,
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
            if deterministic:
                sample = mu
            else:
                # sample weights: w ~ N(w_mean, sigma^2 A^{-1})
                device = mu.device
                w_mean, _ = self._w_mean_and_Ainv()
                sigma = float(self.model.noise.detach().cpu().item())
                L = torch.linalg.cholesky(add_jitter(Ainv, 1e-10))
                if seed is not None:
                    g = torch.Generator(device=device)
                    g.manual_seed(seed)
                    z = torch.randn(w_mean.shape[0], generator=g, device=device)
                else:
                    z = torch.randn(w_mean.shape[0], device=device)
                w_samp = w_mean + sigma * (L @ z)
                sample = Phi @ w_samp

            idx = int(torch.argmax(sample).item())
            return X_candidates[idx], SelectionInfo(mu=mu.detach().cpu(), std=std.detach().cpu(),
                                                    sampled_vals=sample.detach().cpu(), rule="ts")

        # Stack history
        X = torch.vstack(tuple(self.x_hist))
        Theta = torch.vstack(tuple(self.theta_hist))
        y = torch.hstack(tuple(self.y_hist))

        # Get posterior over candidates, including the covariance
        mu, std, cov = self.model.posterior_on_candidates(
            X_candidates,
            theta_t.unsqueeze(0),
            X,
            Theta,
            y,
            calculate_covariance=True,
        )

        # Deterministic = greedy on posterior mean; otherwise Thompson sampling
        if deterministic:
            sample = mu
        else:
            # Sample a function draw over the candidate set
            if seed is not None:
                # Make sampling reproducible if a seed is provided
                g = torch.Generator(device=mu.device)
                g.manual_seed(seed)
                if cov is not None:
                    Lc = torch.linalg.cholesky(cov)
                    z = torch.randn(mu.shape[0], generator=g, device=mu.device)
                    sample = mu + Lc @ z
                else:
                    z = torch.randn_like(mu, generator=g)
                    sample = mu + std * z
            else:
                if cov is not None:
                    Lc = torch.linalg.cholesky(cov)
                    z = torch.randn(mu.shape[0], device=mu.device)
                    sample = mu + Lc @ z
                else:
                    # Fallback: assume independence if covariance is not provided
                    z = torch.randn_like(mu)
                    sample = mu + std * z

        idx = int(torch.argmax(sample).item())
        return X_candidates[idx], SelectionInfo(mu=mu.cpu(), std=std.cpu(), sampled_vals=sample.cpu(), rule="ts")

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
            z = torch.cat([x_t, theta_t], dim=-1)
            self._z_hist.append(z.unsqueeze(0))
            self._yl_hist.append(y_val.unsqueeze(0))

            self.t += 1
            return

        # Default GP path
        # Default GP path (exact GP on a bounded coreset)
        z_new = torch.cat([x_t, theta_t], dim=-1)

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

    def export_posterior(self):
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

