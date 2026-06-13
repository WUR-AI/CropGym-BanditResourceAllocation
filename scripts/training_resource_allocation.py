import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cropgym.baselines import resolve_baseline
from cropgym.train_allocator import train_allocator, train_allocator_for_farm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=107)
    parser.add_argument("--render", action='store_true', help="render", dest='render')

    # Meta stuff
    parser.add_argument("--use_model", action='store_true')
    parser.add_argument("--baseline", type=str, choices=["ROT", "random"], default=None)
    parser.add_argument("--model_dir", type=str, default=None, help="deprecated alias for --baseline")
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--bandit_lr", type=float, default=3e-3)
    parser.add_argument("--bandit_epochs", type=int, default=100)
    parser.add_argument("--action_candidate_length", type=int, default=30_000)

    # Elite candidate memory (persist good actions + neighbors)
    parser.add_argument("--elite_enabled", action="store_true",
                        help="Enable persistent elite candidates (best action + neighbors)")
    parser.add_argument("--elite_neighbors", type=int, default=50,
                        help="How many neighbors to keep around best action")
    parser.add_argument("--elite_keep_max", type=int, default=5000,
                        help="Max elite candidates stored per scenario (full/reduced)")
    parser.add_argument("--elite_top_k", type=int, default=10,
                        help="Number of elite centers to keep per scenario")
    parser.add_argument("--elite_inject_max", type=int, default=2000,
                        help="Max elite rows to inject into the candidate set each round")

    parser.add_argument("--model_name", type=str, default='bandit')
    parser.add_argument("--no_comet", action='store_false', dest='use_comet')
    parser.add_argument("--streaming", action='store_true', dest='streaming')
    parser.add_argument("--q", type=int, default=1)
    parser.add_argument("--farm", type=int, default=None)
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--train_every", type=int, default=1)
    parser.add_argument("--method", type=str, default='ucb')
    parser.add_argument("--kernel", type=str, default='matern')
    parser.add_argument(
        "--bandit_posterior",
        type=str,
        default="gp",
        choices=["gp", "neural_linear"],
        help="Posterior type for bandit. gp=exact NN-AGP GP posterior, neural_linear=Bayesian linear head on NN features."
    )
    parser.add_argument(
        "--bandit_buffer",
        type=int,
        default=256,
        help="Buffer size for representation training (neural_linear)."
    )
    parser.add_argument("--lstm", action='store_true', dest='lstm')
    parser.add_argument("--train_multi_campaign", action='store_true', dest='train_multi_campaign')
    parser.add_argument("--coreset_size", type=int, default=300)
    parser.add_argument("--coreset_mode", type=str, default="fifo", choices=["fifo", "diverse"])
    parser.add_argument("--bandit_action_mode", type=str, default="factored", choices=["joined", "factored"])
    parser.add_argument("--online_update_mode", action='store_true', default=False)
    parser.set_defaults(
        use_model=True,
        use_comet=True,
        streaming=False,
        render=False,
        elite_enabled=True,
        train_multi_campaign=False,
        lstm=False,
    )
    args = parser.parse_args()
    args.baseline = resolve_baseline(
        baseline=args.baseline,
        deprecated_value=args.model_dir,
        deprecated_name="model_dir",
    )
    args.model_dir = args.baseline

    if args.farm is not None:
        train_allocator_for_farm(args)
    else:
        train_allocator(args)
