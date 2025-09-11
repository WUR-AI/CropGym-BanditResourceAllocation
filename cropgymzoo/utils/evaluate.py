import torch

from tianshou.env import PettingZooEnv
from tianshou.policy import MultiAgentPolicyManager
from tianshou.policy.modelfree.ppo import PPOTrainingStats
from tianshou.utils import RunningMeanStd
from tianshou.data.batch import Batch

from cropgymzoo.envs.multi_field_env import MultiFieldEnv


def evaluate_policy(
        env: MultiFieldEnv,
        policy: MultiAgentPolicyManager,
        obs_rms: RunningMeanStd,
        years: list[int],
        agents: list[str] = None,
) -> dict[int, dict[str, dict[str, list]]]:
    """Evaluate a policy on CropGymZoo."""

    policy.eval()
    for p in policy.policies:
        p.deterministic_eval = True

    if agents is None:
        agents = env.possible_agents

    info_dict = {}
    for i, year in enumerate(years):
        info_dict[year] = {}

        next_states = {
            ag: None for ag in agents
        }

        env.reset(options={'year': year})

        for agent in env.agent_iter():
            obs, rew, term, trunc, info = env.last()

            # get appropriate info shape for policy
            processed_info = Batch({k: [i[-1]] for k, i in info.items()})
            processed_info['env_id'] = [0]

            with torch.no_grad():
                out = policy.policies[agent](
                    batch=Batch(
                        {
                            'obs': {
                                'obs': obs_rms.norm(obs['observation']),
                                'mask': env._get_mask(agent),
                            },
                            'info': processed_info,
                        }
                    ),
                    state=Batch(next_states[agent]),
                )

            action = out.act.item()
            state = None if not hasattr(out, 'state') else out.state

            next_states[agent] = state

            if env.terminations[agent]:
                info_dict[year][agent] = info  # grab info before agent dies
                env.step(None)
            else:
                env.step(action)

    return info_dict