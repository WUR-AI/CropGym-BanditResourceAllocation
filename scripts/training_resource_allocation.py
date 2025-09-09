import argparse

from cropgymzoo.train_allocator import train_allocator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=107)

    # Meta stuff
    parser.add_argument("--use_model", action='store_true')
    parser.add_argument("--model_dir", type=str, default='MLP_PPO')
    parser.add_argument("--rounds", type=int, default=5000)
    parser.add_argument("--bandit_lr", type=float, default=5e-3)
    parser.add_argument("--bandit_epochs", type=int, default=50)
    parser.add_argument("--action_candidate_length", type=int, default=2048)
    parser.add_argument("--model_name", type=str, default='s107_model50')
    parser.add_argument("--no-comet", action='store_false', dest='use_comet')
    parser.set_defaults(
        use_model=True,
        use_comet=True,
    )
    args = parser.parse_args()

    train_allocator(args)