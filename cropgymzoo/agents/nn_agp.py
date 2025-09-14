import os
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from cropgymzoo import _DEFAULT_MODEL_DIR


# --------------------------- Utilities ---------------------------

def positive_param(raw: torch.Tensor) -> torch.Tensor:
    # Softplus with small beta for stable gradients
    return F.softplus(raw) + 1e-6


def add_jitter(K: torch.Tensor, jitter: float = 1e-6) -> torch.Tensor:
    return K + jitter * torch.eye(K.shape[-1], device=K.device, dtype=K.dtype)


# --------------------------- Kernels (scalar) ---------------------------

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
        r = d2.clamp_min_(0).sqrt()
        sqrt3r = math.sqrt(3.0) * r
        return (1.0 + sqrt3r) * torch.exp(-sqrt3r)

    def diag(self, X: torch.Tensor) -> torch.Tensor:
        return torch.ones(X.shape[0], device=X.device, dtype=X.dtype)


# --------------------------- Neural feature map g(θ) ---------------------------

class FeatureNet(nn.Module):
    """Small MLP: θ -> g(θ) in R^m."""
    def __init__(self, d_theta: int, m: int, hidden: int = 64, depth: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = d_theta
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            in_dim = hidden
        layers += [nn.Linear(in_dim, m)]
        self.net = nn.Sequential(*layers)

    def forward(self, Theta: torch.Tensor) -> torch.Tensor:
        # (n,d_theta) -> (n,m)
        return self.net(Theta)


# --------------------------- Collaborative multi-output GP K(x,x') ---------------------------
# p(x) = [p_1(x),...,p_m(x)]ᵀ with covariance:
#   K(x,x') = Σ_q A_q k_q(x,x') + Diag( k~_1(x,x'),...,k~_m(x,x') )
# where A_q = L_q L_qᵀ ensures PSD. (Eq. (4))   [oai_citation:5‡9244_Contextual_Gaussian_Proce.pdf](file-service://file-TsvLCc4k6gDpym1pL6Qi5r)

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
        return L @ L.T + 1e-6 * torch.eye(self.m, device=L.device, dtype=L.dtype)

    # Build K̃((X,Θ),(X',Θ')) = g(Θ)ᵀ K(X,X') g(Θ')  (Prop. 1)   [oai_citation:6‡9244_Contextual_Gaussian_Proce.pdf](file-service://file-TsvLCc4k6gDpym1pL6Qi5r)
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
      - g(θ) via FeatureNet
      - collaborative MOGP for K(x,x')
      - exact GP posterior over f(x;θ) using K̃
    """
    def __init__(self, d_theta: int, d_x: int, m: int, Q: int = 1, device: Optional[torch.device] = None):
        super().__init__()
        self.g_net = FeatureNet(d_theta, m)
        self.mogp = CollaborativeMOGP(d_x, m, Q=Q)
        self.raw_noise = nn.Parameter(torch.tensor(-2.0))  # σ_ε ≈ 0.12
        self.jitter = 1e-4
        self.device = device or torch.device("cpu")
        self.to(self.device)

        # cache
        self._train_cache = {}  # keys: ("Theta","X","y","L","alpha","K̃")

    @property
    def noise(self) -> torch.Tensor:
        sigma = positive_param(self.raw_noise)
        return torch.clamp(sigma, min=1e-3)  # floor the noise

    def _clear_cache(self):
        self._train_cache.clear()

    # ---- training objective: negative log-marginal likelihood (Eq. (5))   [oai_citation:7‡9244_Contextual_Gaussian_Proce.pdf](file-service://file-TsvLCc4k6gDpym1pL6Qi5r)
    def nll(self, act: torch.Tensor, context_in: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        act, context_in, y = act.to(self.device), context_in.to(self.device), y.to(self.device)
        G = self.g_net(context_in)                        # (n,m)
        Kt = self.mogp.K_tilde(act, context_in, act, context_in, G, G)  # (n,n)
        Kt = 0.5 * (Kt + Kt.T)
        Kn = add_jitter(Kt, self.jitter) + self.noise**2 * torch.eye(act.shape[0], device=act.device)
        L = torch.linalg.cholesky(Kn)
        alpha = torch.cholesky_solve(y.unsqueeze(1), L).squeeze(1)  # (n,)

        log_det = 2.0 * torch.log(torch.diag(L)).sum()
        quad = 0.5 * (y @ alpha)
        const = 0.5 * act.shape[0] * math.log(2.0 * math.pi)
        nll = quad + 0.5 * log_det + const

        # cache for prediction if inputs match
        self._train_cache = {"Theta": context_in, "X": act, "y": y, "L": L.detach(), "alpha": alpha.detach(), "G": G.detach()}
        return nll

    # ---- exact posterior μ, σ at batch of (X*,Θ*) (Eq. (2))   [oai_citation:8‡9244_Contextual_Gaussian_Proce.pdf](file-service://file-TsvLCc4k6gDpym1pL6Qi5r)
    @torch.no_grad()
    def predict(self, X_star: torch.Tensor, Theta_star: torch.Tensor,
                train_X: torch.Tensor, train_Theta: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.device
        X_star = X_star.to(device)
        Theta_star = Theta_star.to(device)
        train_X = train_X.to(device)
        train_Theta = train_Theta.to(device)
        y = y.to(device)

        # reuse cached Cholesky if consistent with latest parameters/inputs
        need_refit = not self._train_cache or \
                     self._train_cache["X"].data_ptr() != train_X.data_ptr() or \
                     self._train_cache["Theta"].data_ptr() != train_Theta.data_ptr()

        if need_refit:
            G_tr = self.g_net(train_Theta)
            Kt = self.mogp.K_tilde(train_X, train_Theta, train_X, train_Theta, G_tr, G_tr)
            Kn = add_jitter(Kt, self.jitter) + self.noise**2 * torch.eye(train_X.shape[0], device=device)
            L = torch.linalg.cholesky(Kn)
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

    # ---- posterior over a finite candidate set for one context θ_t (UCB, TS use this)
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
        # Build covariance only when needed (e.g., TS)
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


# --------------------------- Acquisition rules ---------------------------

def beta_finite_candidates(t: int, M: int, delta: float = 0.1) -> float:
    # Finite-arm CGP-UCB schedule (cf. paper’s finite set analysis): β_t = 2 log(|X_t| t^2 π^2 / (6 δ))
    # Safe, increasing, and practical for candidate sets.   [oai_citation:9‡9244_Contextual_Gaussian_Proce.pdf](file-service://file-TsvLCc4k6gDpym1pL6Qi5r)
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
    def __init__(self, d_theta: int, d_x: int, m: int = 8, Q: int = 1, lr: float = 3e-3, device: Optional[torch.device] = None):
        self.model = NNAGP(d_theta, d_x, m=m, Q=Q, device=device or torch.device("cpu"))
        self.theta_hist: List[torch.Tensor] = []
        self.x_hist: List[torch.Tensor] = []
        self.y_hist: List[torch.Tensor] = []
        self.t = 1

        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

    # ---- training step: maximize log marginal likelihood (Eq. (5))   [oai_citation:10‡9244_Contextual_Gaussian_Proce.pdf](file-service://file-TsvLCc4k6gDpym1pL6Qi5r)
    def train_step(self, steps: int = 200, lr: float = 3e-3) -> float:
        if len(self.y_hist) == 0:
            return 0.0
        x = torch.vstack(self.x_hist)
        theta = torch.vstack(self.theta_hist)
        y = torch.hstack(self.y_hist)
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

    # ---- choose x_t by UCB over a finite candidate set for current θ_t (Eq. (3))   [oai_citation:11‡9244_Contextual_Gaussian_Proce.pdf](file-service://file-TsvLCc4k6gDpym1pL6Qi5r)
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

        X = torch.vstack(self.x_hist)
        Theta = torch.vstack(self.theta_hist)
        y = torch.hstack(self.y_hist)
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

        X = torch.vstack(self.x_hist)
        Theta = torch.vstack(self.theta_hist)
        y = torch.hstack(self.y_hist)
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

    # ---- choose x_t by Thompson Sampling over candidates (supp §6.1)   [oai_citation:12‡9244_Contextual_Gaussian_Proce_Supplementary Material.pdf](file-service://file-VVr9GXm9MSGcjuonPbh1BE)
    @torch.no_grad()
    def select_ts(self, theta_t: torch.Tensor, X_candidates: torch.Tensor) -> Tuple[torch.Tensor, SelectionInfo]:
        if len(self.y_hist) == 0:
            idx = torch.randint(0, X_candidates.shape[0], (1,)).item()
            mu = torch.zeros(X_candidates.shape[0])
            std = torch.ones(X_candidates.shape[0])
            return X_candidates[idx], SelectionInfo(mu=mu, std=std, rule="ts")
        X = torch.vstack(self.x_hist); Theta = torch.vstack(self.theta_hist); y = torch.hstack(self.y_hist)
        mu, std, cov = self.model.posterior_on_candidates(X_candidates, theta_t.unsqueeze(0), X, Theta, y)
        # exact correlated sample
        Lc = torch.linalg.cholesky(add_jitter(cov))
        z = torch.randn(mu.shape[0], device=mu.device)
        sample = mu + Lc @ z
        idx = int(torch.argmax(sample).item())
        return X_candidates[idx], SelectionInfo(mu=mu.cpu(), std=std.cpu(), sampled_vals=sample.cpu(), rule="ts")

    # ---- log an observation and advance time
    def update(self, theta_t: torch.Tensor, x_t: torch.Tensor, y_t: float):
        self.theta_hist.append(theta_t.detach().unsqueeze(0))
        self.x_hist.append(x_t.detach().unsqueeze(0))
        self.y_hist.append(torch.tensor([y_t], dtype=torch.get_default_dtype()))
        self.t += 1

    def export_posterior(self):
        """Return everything needed for posterior predictions."""
        X = torch.vstack(self.x_hist) if self.x_hist else torch.empty(0, self.model.mogp.k_shared[
            0].raw_lengthscale.numel())
        Theta = torch.vstack(self.theta_hist) if self.theta_hist else torch.empty(0,
                                                                                  self.model.g_net.net[0].in_features)
        y = torch.hstack(self.y_hist) if self.y_hist else torch.empty(0)
        cache = None
        if self.model._train_cache:
            # optional speed-up: store Cholesky and alpha so we don’t recompute
            cache = {
                "L": self.model._train_cache["L"].cpu(),
                "alpha": self.model._train_cache["alpha"].cpu(),
                "G": self.model._train_cache["G"].cpu(),
            }
        return {
            "model_state": self.model.state_dict(),
            "X": X.cpu(),
            "Theta": Theta.cpu(),
            "y": y.cpu(),
            "cache": cache,
        }

    def import_posterior(self, blob: dict, map_location="cpu"):
        self.model.load_state_dict(blob["model_state"])
        self.x_hist = [blob["X"].to(map_location)]
        self.theta_hist = [blob["Theta"].to(map_location)]
        self.y_hist = [blob["y"].to(map_location)]
        # rebuild cache (optional)
        self.model._clear_cache()

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

    def save(
            self,
            seed: int = None,
            t=None,
            name: str = None,
            args = None,
            rms = None,
    ):
        os.makedirs(os.path.join(_DEFAULT_MODEL_DIR, f"NN-ACGP-Bandit{'-streaming' if args.streaming else ''}"), exist_ok=True)
        name_file = f"s{seed}_model{t if t is not None else ''}.pth" if name is None else f"{name}.pth"
        file_dir = os.path.join(
                _DEFAULT_MODEL_DIR,
                f"NN-ACGP-Bandit{'-streaming' if args.streaming else ''}",
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

