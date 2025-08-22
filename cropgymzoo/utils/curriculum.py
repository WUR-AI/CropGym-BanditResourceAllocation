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
        stages=[
            {'budget': False, 'sowing': False, 'weather': False, 'co2': False, 'initial_n': False, 'parameters': False},
            {'budget': True, 'sowing': False, 'weather': False, 'co2': False, 'initial_n': False, 'parameters': False},
            {'budget': True, 'sowing': True, 'weather': False, 'co2': False, 'initial_n': False, 'parameters': False},
            {'budget': True, 'sowing': True, 'weather': True, 'co2': False, 'initial_n': False, 'parameters': False},
            {'budget': True, 'sowing': True, 'weather': True, 'co2': True, 'initial_n': False, 'parameters': False},
        ]
    )
