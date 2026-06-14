import numpy as np

from collections import defaultdict

def run_aec_till_terminate(env):

    cumulative_step_rewards = []  # farm-level reward per “parallel step”
    cumulative_global_rewards = []
    running_sum = 0.0  # holds rewards until the last parcel acts
    running_global = 0.0

    for agent in env.agent_iter():
        print("step {}".format(env.current_step))
        obs, rew, term, trunc, info = env.last()  # :contentReference[oaicite:0]{index=0}

        individual_reward = env.rewards[agent]

        running_sum += rew  # each agent got the same scalar
        running_global += individual_reward
        is_last = (agent == env.agents[-1])

        # dead-step required by the API
        # env.step(None)\
        #     if (term or trunc)\
        #     else
        env.step(env.action_space(agent).sample(mask=np.array(env._get_mask(agent), dtype=np.int8)))

        if is_last:
            cumulative_step_rewards.append(running_sum)
            cumulative_global_rewards.append(running_global)
            running_sum = 0.0

    return env, cumulative_step_rewards, cumulative_global_rewards

def run_aec_step(env, action: int=None):
    print("step {}".format(env.current_step))
    obs, rew, term, trunc, info = env.last()  # :contentReference[oaicite:0]{index=0}

    # dead-step required by the API
    env.step(None) \
        if (term or trunc) \
        else env.step(
            action if action is not None
            else np.random.choice(range(7), p=[0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    )

    return obs, rew, term, trunc, info, env
