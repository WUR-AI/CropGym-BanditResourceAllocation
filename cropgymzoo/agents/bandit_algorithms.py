import math, torch, numpy as np
from collections import defaultdict

from cropgymzoo.agents.networks import UCBNetwork


class FeatureNet(torch.nn.Module):
    """Shared representation Φ(x;θ)."""
    def __init__(self, ctx_dim: int, hidden: int = 64, out_dim: int = 32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(ctx_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, out_dim))
    def forward(self, x):                       # x: (B, ctx_dim)
        return self.net(x)                      # (B, d)

class NeuralC2UCB:
    """
    One instance handles *all* base arms.
    heads:  W  shape (n_arms, d)
    per-arm A_inv  shape (n_arms, d, d)
    """
    def __init__(self, ctx_dim, n_arms, d=32,    # d = Φ output size
                 lambda_=1.0, sigma=1.0, delta=0.01, lr=1e-3):
        self.d = d
        self.sigma, self.delta = sigma, delta
        self.phinet = FeatureNet(ctx_dim, out_dim=d)
        self.opt = torch.optim.Adam(self.phinet.parameters(), lr=lr)

        self.W = torch.zeros(n_arms, d)            # last-layer weights
        self.A_inv = torch.stack([torch.eye(d)/lambda_ for _ in range(n_arms)])
        self.b = torch.zeros(n_arms, d)
        self.t = 0

    # ────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def ucb(self, ctx, beta):
        """ctx: (ctx_dim,) numpy  →  UCB scores (n_arms,)"""
        phi = self.phinet(torch.tensor(ctx, dtype=torch.float32)).detach()  # (d,)
        # einsum trick: variance = φᵀ A_inv φ for every arm
        var = torch.einsum("ad, d -> a", (self.A_inv @ phi), phi)
        mean = (self.W @ phi)
        return (mean + beta * torch.sqrt(var)).cpu().numpy(), phi

    # ────────────────────────────────────────────────────────────────
    def update(self, arm_indices, phi, reward):
        """
        Update only the arms that actually appeared in the chosen super-arm.
        arm_indices : list[int]       base-arm IDs
        phi         : torch.Tensor(d)  shared feature of this season
        reward      : float
        """
        self.t += 1
        beta = self.sigma * math.sqrt(self.d * math.log((1 + self.t) / self.delta))
        # 1) SGD step on shared net (regression to reward)
        self.opt.zero_grad()
        pred = (self.W[arm_indices] @ phi).mean()
        loss = (pred - reward) ** 2
        loss.backward(); self.opt.step()

        # 2) Bayesian linear update for each involved arm
        phi_np = phi.detach().cpu().numpy()
        for a in arm_indices:
            Ainv = self.A_inv[a].numpy()
            # Sherman–Morrison rank-1 update
            Avphi = Ainv @ phi_np
            fac   = 1.0 / (1.0 + phi_np @ Avphi)
            Ainv_updated = Ainv - fac * np.outer(Avphi, Avphi)
            self.A_inv[a].copy_(torch.tensor(Ainv_updated))
            # weight update
            self.b[a] += reward * phi.detach()
            self.W[a] = self.A_inv[a] @ self.b[a]

    # helper for β_t used outside
    def beta(self):          # compute on demand
        return self.sigma * math.sqrt(self.d * math.log((1 + self.t) / self.delta))


class NeuralUCBDiag:
    def __init__(self, dim, lamdba=1, nu=1, hidden=100):
        self.func = UCBNetwork(dim, hidden_size=hidden).cuda()
        self.context_list = []
        self.reward = []
        self.lamdba = lamdba
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        self.U = lamdba * torch.ones((self.total_param,)).cuda()
        self.nu = nu

    def select(self, context):
        tensor = torch.from_numpy(context).float().cuda()
        mu = self.func(tensor)
        g_list = []
        sampled = []
        ave_sigma = 0
        ave_rew = 0
        for fx in mu:
            self.func.zero_grad()
            fx.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() for p in self.func.parameters()])
            g_list.append(g)
            sigma2 = self.lamdba * self.nu * g * g / self.U
            sigma = torch.sqrt(torch.sum(sigma2))

            sample_r = fx.item() + sigma.item()

            sampled.append(sample_r)
            ave_sigma += sigma.item()
            ave_rew += sample_r
        arm = np.argmax(sampled)
        self.U += g_list[arm] * g_list[arm]
        return arm, g_list[arm].norm().item(), ave_sigma, ave_rew

    def train(self, context, reward):
        self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
        self.reward.append(reward)
        optimizer = torch.optim.SGD(self.func.parameters(), lr=1e-2, weight_decay=self.lamdba)
        length = len(self.reward)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            for idx in index:
                c = self.context_list[idx]
                r = self.reward[idx]
                optimizer.zero_grad()
                delta = self.func(c.cuda()) - r
                loss = delta * delta
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 1000:
                    return tot_loss / 1000
            if batch_loss / length <= 1e-3:
                return batch_loss / length
