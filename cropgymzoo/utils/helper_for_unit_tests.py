import numpy as np

from collections import defaultdict

def run_aec_till_terminate(env):

    cumulative_step_rewards = []  # farm-level reward per “parallel step”
    running_sum = 0.0  # holds rewards until the last parcel acts

    for agent in env.agent_iter():
        print("step {}".format(env.current_step))
        obs, rew, term, trunc, info = env.last()  # :contentReference[oaicite:0]{index=0}

        running_sum += rew  # each agent got the same scalar
        is_last = (agent == env.agents[-1])

        # dead-step required by the API
        env.step(None)\
            if (term or trunc)\
            else env.step(np.random.choice(range(7), p=[0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]))

        if is_last:
            cumulative_step_rewards.append(running_sum)
            running_sum = 0.0

    return env, cumulative_step_rewards, running_sum

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
