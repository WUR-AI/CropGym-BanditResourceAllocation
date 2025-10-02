import torch
import numpy as np
from cropgymzoo.agents.nn_agp import NNAGPBandit

import matplotlib.pyplot as plt

def reward_R1(x: torch.Tensor, theta: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    R1(x, theta) = - sqrt( | sin(||x||) * ||theta||^3 * exp( cos(||x|| + ||theta||) ) | )

    Args
    ----
    x:     (..., d_x)      action vector(s)
    theta: (..., d_theta)  context vector(s)
    eps:   small constant to avoid sqrt(0) / log(0) edge cases

    Returns
    -------
    r:     (...) scalar reward per input sample
    """
    # ensure tensors
    x = torch.as_tensor(x)
    theta = torch.as_tensor(theta)

    # norms
    # nx = torch.linalg.vector_norm(x, dim=dim, keepdim=True)
    # nt = torch.linalg.vector_norm(theta, dim=dim, keepdim=True)
    nx = torch.linalg.norm(x, dim=-1)
    nt = torch.linalg.norm(theta, dim=-1)

    # inside the absolute value
    inner = torch.sin(nx) * (theta ** 3) * torch.exp(torch.cos(nx + nt))

    # numerically safe sqrt of absolute value, then negative sign
    r = -torch.sqrt(torch.abs(inner) + eps)
    r = r.sum(dim=-1)
    return r

def reward_R2(x: torch.Tensor, theta: torch.Tensor, dim = -1, elementwise = True, eps = 1e-12) -> torch.Tensor:
    # ensure tensors
    x = torch.as_tensor(x)
    theta = torch.as_tensor(theta)

    # L2 norms along the feature dimension
    # norm_x = torch.linalg.vector_norm(x, dim=dim, keepdim=True)
    # norm_theta = torch.linalg.vector_norm(theta, dim=dim, keepdim=True)
    norm_x = torch.linalg.norm(x, dim=dim)
    norm_theta = torch.linalg.norm(theta, dim=dim)

    # Scalar scale factor (broadcastable across x's feature dim)
    scale = torch.abs(torch.sin(norm_x)) * torch.exp(norm_x + torch.cos(norm_theta))

    if elementwise:
        inside = torch.abs(scale * (x ** 3))
        r = -torch.sqrt(inside + eps)
        return r.sum(dim=dim)
    else:
        # Collapse x^3 to a scalar per sample via its norm, then sqrt
        x3_norm = torch.linalg.vector_norm(x ** 3, dim=dim, keepdim=True)
        inside = torch.abs(scale * x3_norm)
        out = -torch.sqrt(inside + eps)
        return out.squeeze(dim)  # drop the feature dimension




# toy environment
def reward_fn(x, theta, r: int = 2):
    # unknown to the bandit; just for simulation
    if r == 1:
        return reward_R1(x, theta)  # already returns a scalar tensor
    elif r == 2:
        return reward_R2(x, theta)
    else:
        raise ValueError(f"Unknown reward function r={r}")

def reward_fn_noisy(x, theta, noise_std=0.05):
    return reward_R1(x, theta) + noise_std * torch.randn((), dtype=x.dtype)

def oracle_x_star(theta, d_x=2, steps=400, lr=0.05, restarts=5, seed=0):
    best_x, best_r = None, -1e18
    g = torch.Generator().manual_seed(seed)
    for k in range(restarts):
        x = (torch.rand(d_x, generator=g) * 2 - 1).requires_grad_(True)
        opt = torch.optim.Adam([x], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss = -reward_fn(x, theta)
            loss.backward()
            opt.step()
            with torch.no_grad():
                x.clamp_(-1.0, 1.0)
        with torch.no_grad():
            r = reward_fn(x, theta).item()
            if r > best_r:
                best_r, best_x = r, x.detach().clone()
    return best_x, torch.as_tensor(best_r)

def sample_unit_ball(d, n=1):
    # Gaussian trick: sample normal, normalize, then scale with radius^(1/d)
    x = torch.randn(n, d)                               # normal distribution
    x /= torch.linalg.norm(x, dim=1, keepdim=True)      # project to unit sphere
    r = torch.rand(n, 1).pow(1/d)                       # radius ~ U[0,1]^(1/d)
    return r * x                                        # uniform in ball

def run_one(d_theta, d_x, seed, T=300, noise_std=0.1, r=2):
    from cropgymzoo.agents.nn_agp import NNAGPBandit
    torch.manual_seed(seed); np.random.seed(seed)
    bandit = NNAGPBandit(d_theta, d_x, m=5, Q=1, lr=1e-4, device=torch.device("cpu"))

    inst_regs = []
    avg_regs = []
    cum_reg = 0.0

    for t in range(1, T+1):
        theta_t = torch.clamp(torch.randn(d_theta), -1, 1)
        Xcand = torch.clamp(torch.randn(1024, d_x), -1, 1)

        # small surrogate update
        loss = bandit.train_step(steps=100)

        # pick action with UCB over the candidate set
        x_t, info = bandit.select_ucb(theta_t, Xcand, delta=0.1)

        # observe noisy reward for learning
        r_true = reward_fn(x_t, theta_t, r)
        y_t = float(r_true + noise_std * torch.randn(()))
        bandit.update(theta_t, x_t, y_t)

        # # --- regret w.r.t. continuous oracle ---
        # _, r_star = oracle_x_star(theta_t, d_x=d_x, steps=300, lr=0.05,
        #                           restarts=5, seed=seed+t)
        inst = float(0 - r_true)
        cum_reg += inst
        inst_regs.append(inst)
        avg_regs.append(cum_reg / t)
        print(f"Round {t}, Instant regrets:{inst}")

    return np.array(inst_regs), np.array(avg_regs)


def moving_average(x, window_size=10):
    return np.convolve(x, np.ones(window_size)/window_size, mode="valid")

def moving_average_full(x, window_size=10):
    ma = []
    for i in range(len(x)):
        # use fewer values for early rounds
        start = max(0, i - window_size + 1)
        ma.append(np.mean(x[start:i+1]))
    return np.array(ma)

if __name__ == "__main__":

    S, T = 3, 600
    all_avg = []
    r = 2  # Reward function to test
    if r==1:
        d_theta, d_x = 3, 2
    elif r==2:
        d_theta, d_x = 15, 5
    else:
        raise ValueError(f"Unknown reward function r={r}")
    for s in range(S):
        seed = 1236 + s
        inst, avg = run_one(d_theta, d_x, seed=seed, T=T, noise_std=0.1, r=r)
        print(f"seed: {seed}")
        inst = moving_average_full(inst, window_size=20)
        all_avg.append(inst)
    all_avg = np.stack(all_avg)  # (S, T)

    mean = all_avg.mean(0)
    q10 = np.quantile(all_avg, 0.05, axis=0)
    q90 = np.quantile(all_avg, 0.95, axis=0)

    plt.figure()
    plt.plot(range(1, T + 1), mean, label="NN-AGP-UCB (m=5)")
    plt.fill_between(range(1, T + 1), q10, q90, alpha=0.2)
    plt.xlabel("Rounds")
    plt.ylabel("Average regret")
    plt.title(f"Average regret of using R_{r}(x;theta) with X=[-1,1]^{d_x}, theta=[-1,1]^{d_theta}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # torch.set_default_dtype(torch.float64)
    #
    # d_theta, d_x = 3, 2  # context and action dims
    # bandit = NNAGPBandit(d_theta, d_x, m=3, Q=1, device=torch.device("cpu"))
    #
    # T = 500
    # losses = []
    # rewards = []
    # noisy_rewards = []
    #
    # regrets = []
    # cum_regrets = []
    # cum_regret = 0.0
    #
    # for t in range(1, T + 1):
    #     theta_t = torch.randn(d_theta) * 2 - 1  # observe context
    #     # candidate set for actions (or sample from a box)
    #     Xcand = torch.rand(1024, d_x) * 2 - 1
    #
    #     # train the surrogate a bit on accumulated data
    #     loss = bandit.train_step(steps=50, lr=5e-3)
    #     losses.append(float(loss))
    #     print(f"round {t}, loss: {loss}")
    #
    #     # pick by UCB (or switch to bandit.select_ts(...))
    #     x_t, info = bandit.select_ucb(theta_t, Xcand, delta=0.1)
    #
    #     reward = reward_fn(x_t, theta_t)
    #     rewards.append(float(reward))
    #     print(f"reward {reward}")
    #
    #     # observe noisy reward
    #     y_t = float(reward + 0.05 * torch.randn(()))
    #     noisy_rewards.append(y_t)
    #     print(f"noisy reward {y_t}")
    #     bandit.update(theta_t, x_t, y_t)
    #
    #     with torch.no_grad():
    #         r_all = reward_fn(Xcand, theta_t)  # (256,)
    #         x_star = Xcand[torch.argmax(r_all)]
    #         r_star = torch.max(r_all)
    #
    #     inst_regret = float(r_star - reward)
    #     cum_regret += inst_regret
    #     regrets.append(inst_regret)
    #     cum_regrets.append(cum_regret)
    #
    #     print(f"inst_regret {inst_regret:.6f} | cum_regret {cum_regret:.6f}")
    #
    # # ---- plots ----
    # # 1) Loss vs round
    # plt.figure()
    # plt.plot(range(1, T + 1), losses)
    # plt.xlabel("Round")
    # plt.ylabel("Training loss")
    # plt.title("NN-AGP Training Loss per Round")
    # plt.tight_layout()
    # plt.show()
    #
    # # 2) Reward vs round (true & noisy)
    # plt.figure()
    # plt.plot(range(1, T + 1), rewards, label="True reward")
    # plt.plot(range(1, T + 1), noisy_rewards, label="Noisy reward", alpha=0.7)
    # plt.xlabel("Round")
    # plt.ylabel("Reward")
    # plt.title("Reward per Round")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    #
    # plt.figure()
    # plt.plot(range(1, T + 1), regrets, label="Instantaneous regret")
    # plt.xlabel("Round")
    # plt.ylabel("Regret")
    # plt.title("Instantaneous Regret per Round")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    #
    # plt.figure()
    # plt.plot(range(1, T + 1), cum_regrets, label="Cumulative regret", color="red")
    # plt.xlabel("Round")
    # plt.ylabel("Cumulative Regret")
    # plt.title("Cumulative Regret over Time")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()