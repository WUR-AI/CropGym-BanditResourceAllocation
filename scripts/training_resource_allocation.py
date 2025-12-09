import argparse

from cropgymzoo.train_allocator import train_allocator, train_allocator_for_farm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=107)
    parser.add_argument("--render", action='store_true', help="render", dest='render')

    # Meta stuff
    parser.add_argument("--use_model", action='store_true')
    parser.add_argument("--model_dir", type=str, default='GRU_PPO')
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--bandit_lr", type=float, default=3e-3)
    parser.add_argument("--bandit_epochs", type=int, default=50)
    parser.add_argument("--action_candidate_length", type=int, default=30_000)
    parser.add_argument("--model_name", type=str, default='bandit')
    parser.add_argument("--no_comet", action='store_false', dest='use_comet')
    parser.add_argument("--streaming", action='store_true', dest='streaming')
    parser.add_argument("--q", type=int, default=1)
    parser.add_argument("--farm", type=int, default=None)
    parser.add_argument("--method", type=str, default='ucb')
    parser.add_argument("--kernel", type=str, default='matern')
    parser.set_defaults(
        use_model=True,
        use_comet=True,
        streaming=False,
        render=False
    )
    args = parser.parse_args()

    if args.farm is not None:
        train_allocator_for_farm(args)
    else:
        train_allocator(args)