from abc import ABC, abstractmethod


SUPPORTED_BASELINES = ("ROT", "random")


def normalize_baseline(value: str | None, *, default: str = "ROT") -> str:
    baseline = default if value is None else str(value)
    if baseline.lower() == "rot":
        return "ROT"
    if baseline.lower() == "random":
        return "random"
    raise ValueError(
        f"Unsupported baseline '{value}'. Supported paper baselines are: {', '.join(SUPPORTED_BASELINES)}."
    )


def resolve_baseline(
    *,
    baseline: str | None = None,
    deprecated_value: str | None = None,
    deprecated_name: str = "deprecated",
    default: str = "ROT",
) -> str:
    normalized_baseline = normalize_baseline(baseline, default=default) if baseline is not None else None
    normalized_deprecated = (
        normalize_baseline(deprecated_value, default=default) if deprecated_value is not None else None
    )

    if normalized_baseline is not None and normalized_deprecated is not None:
        if normalized_baseline != normalized_deprecated:
            raise ValueError(
                f"Conflicting baseline values: --baseline={normalized_baseline} and "
                f"--{deprecated_name}={normalized_deprecated}."
            )
        return normalized_baseline

    if normalized_baseline is not None:
        return normalized_baseline
    if normalized_deprecated is not None:
        return normalized_deprecated
    return normalize_baseline(default, default=default)


class BaseBaseline(ABC):
    def __init__(self, env, render: bool = False):
        self.env = env
        self.render = render
        self.agents = self.env.possible_agents

    @abstractmethod
    def get_action(self, agent: str, env=None, scenario: str | None = None):
        raise NotImplementedError

    def run(
        self,
        years: list,
        year_key: bool = True,
        scenario: str = "max",
        *,
        reset_options: dict | None = None,
        reset_each_year: bool = True,
    ) -> dict:
        if not reset_each_year:
            if reset_options is None:
                raise ValueError("reset_each_year=False requires reset_options")
            self.env.reset(options=reset_options)
            info_dict: dict = {}
            for agent in self.env.agent_iter():
                _, _, _, _, info = self.env.last()
                action = self.get_action(agent, env=self.env, scenario=scenario)
                if self.env.terminations[agent]:
                    info_dict[agent] = info
                    self.env.step(None)
                else:
                    self.env.step(action)
            if self.render:
                self.env.render()
            return info_dict

        info_dict = {}
        for year in years:
            if year_key:
                info_dict[year] = {}

            options = {"year": year} if reset_options is None else {**reset_options, "year": year}
            self.env.reset(options=options)

            for agent in self.env.agent_iter():
                _, _, _, _, info = self.env.last()
                action = self.get_action(agent, env=self.env, scenario=scenario)

                if self.env.terminations[agent]:
                    if year_key:
                        info_dict[year][agent] = info
                    else:
                        info_dict[agent] = info
                    self.env.step(None)
                else:
                    self.env.step(action)
            if self.render:
                self.env.render()
        return info_dict


class RoTAgent(BaseBaseline):
    def get_action(self, agent: str, env=None, scenario: str | None = None):
        env = self.env if env is None else env
        return env.rule_of_thumb(agent)


class RandomAgent(BaseBaseline):
    def get_action(self, agent: str, env=None, scenario: str | None = None):
        env = self.env if env is None else env
        return env.random_fertilization(agent)


def make_baseline_runner(baseline: str, env, render: bool = False) -> BaseBaseline:
    baseline = normalize_baseline(baseline)
    if baseline == "ROT":
        return RoTAgent(env=env, render=render)
    if baseline == "random":
        return RandomAgent(env=env, render=render)
    raise ValueError(f"Unsupported baseline '{baseline}'.")
