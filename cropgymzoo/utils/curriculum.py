from dataclasses import dataclass


@dataclass
class RandomiseStage:
    """
    Manager for environment randomisation. Designed to used with curriculum learning.
    """
    stages: list[dict[str, bool]]
    stage: int = 0  # current stage index

    def set_stage(self, i: int) -> None:
        if not (0 <= i < len(self.stages)):
            raise IndexError(f"stage {i} out of range")
        self.stage = i

    def get_max_stage(self) -> int:
        return len(self.stages) - 1

    def __getattr__(self, name: str):
        """Look up keys in the current stage dict."""
        if name in self.stages[self.stage]:
            return self.stages[self.stage][name]
        raise AttributeError(f"{name} not found")

    def __setattr__(self, name: str, value):
        if name in {"stages", "stage"}:
            object.__setattr__(self, name, value)
        elif (
            "stages" in self.__dict__
            and name in self.stages[self.stage]
        ):
            self.stages[self.stage][name] = value
        else:
            object.__setattr__(self, name, value)


def make_default_stage_manager():
    return RandomiseStage(
        stages=make_default_stages()
    )

def make_default_stages():
    return [
        {'sowing': False, 'weather': False, 'budget': False, 'co2': False, 'initial_n': False, 'parameters': False},
        {'sowing': True, 'weather': False, 'budget': False, 'co2': False, 'initial_n': False, 'parameters': False},
        {'sowing': True, 'weather': True, 'budget': False, 'co2': False, 'initial_n': False, 'parameters': False},
        {'sowing': True, 'weather': True, 'budget': False, 'co2': True, 'initial_n': False, 'parameters': False},
        {'sowing': True, 'weather': True, 'budget': True, 'co2': True, 'initial_n': False, 'parameters': False},
    ]


class CurriculumCallbackManager:
    def __init__(
        self,
        *,
        beta: float = 0.1,                 # EMA smoothing
        start_stage: int = 0,
        min_epochs_per_stage: int = 250,   # gate for stages >= 1
        first_stage_reward: float = 2500,
        require_ema_and_inst: bool = True,  # "consistent": both EMA and instant > threshold
        max_stage: bool = 4,
    ):
        self.beta = beta
        self.stage = start_stage
        self.ema: float | None = None
        self.last_score: float | None = None
        self.epochs_in_stage = 0

        self.min_epochs_per_stage = int(min_epochs_per_stage)
        self.first_stage_reward = float(first_stage_reward)
        self.require_ema_and_inst = bool(require_ema_and_inst)
        self.max_stage = max_stage

    def update(self, score: float) -> float:
        """Call once per epoch with your eval metric (avg reward).
        Also increments epoch counter for the current stage.
        """
        self.last_score = float(score)
        self.ema = score if self.ema is None else (1 - self.beta) * self.ema + self.beta * score
        self.epochs_in_stage += 1
        return self.ema

    def _stage_zero_gate(self) -> bool:
        """Stage 0 -> 1 advancement rule: reward > first_stage_reward (instant),
        and optionally EMA > threshold too for consistency."""
        if self.last_score is None or self.ema is None:
            return False
        if self.require_ema_and_inst:
            return (self.last_score > self.first_stage_reward) and (self.ema > self.first_stage_reward)
        else:
            return self.last_score > self.first_stage_reward

    def _epoch_gate(self) -> bool:
        """Stages >=1 advancement rule: spend at least N epochs in the current stage."""
        return self.epochs_in_stage >= self.min_epochs_per_stage

    def should_advance(self) -> bool:
        if self.stage >= self.max_stage:
            return False
        if self.stage == 0:
            return self._stage_zero_gate()
        # stages >= 1
        return self._epoch_gate()

    def advance(self) -> None:
        if self.should_advance():
            self.stage += 1
            self._reset_stage_counters()
            print(f"Curriculum learning stage advanced to {self.stage}")

    def _reset_stage_counters(self):
        # keep EMA to remain stable across stages, or reset if you prefer:
        # self.ema = None
        self.epochs_in_stage = 0

    # (optional) handy introspection helpers
    @property
    def epochs_left(self) -> int:
        if self.stage == 0:
            return 0  # epoch count doesn't gate stage 0
        return max(0, self.min_epochs_per_stage - self.epochs_in_stage)

    def status(self) -> dict:
        return {
            "stage": self.stage,
            "ema": self.ema,
            "last_score": self.last_score,
            "epochs_in_stage": self.epochs_in_stage,
            "epochs_left": self.epochs_left,
            "first_stage_reward": self.first_stage_reward,
            "min_epochs_per_stage": self.min_epochs_per_stage,
        }
