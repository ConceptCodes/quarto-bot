from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Tuple, Any

import yaml


def _default_xml_path() -> str:
    """Return absolute path to the default MuJoCo scene."""
    return str(
        (
            Path(__file__).resolve().parents[1] / "core" / "scenes" / "pick_scene.xml"
        ).resolve()
    )


def _default_device() -> str:
    """Pick the best available accelerator on Apple Silicon first."""
    try:
        import torch

        if torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


@dataclass
class PickAndPlaceRandomization:
    """Domain randomization parameters for spawn + placement targets."""

    piece_xy_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (0.25, 0.35),
        (-0.05, 0.05),
    )
    piece_z_bounds: Tuple[float, float] = (0.04, 0.06)
    placement_xy_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (0.15, 0.35),
        (-0.12, 0.12),
    )
    placement_z: float = 0.04
    board_grid_size: Tuple[int, int] = (4, 4)
    board_square_spacing: float = 0.045
    randomize_rotation: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PickAndPlaceEnvConfig:
    """Configuration for the MuJoCo pick-and-place task."""

    xml_path: str = field(default_factory=_default_xml_path)
    control_dt: float = 1 / 60.0
    action_scale: float = 0.05
    max_episode_steps: int = 120
    reward_success_bonus: float = 150.0
    reward_distance_scale: float = -2.0
    reward_energy_scale: float = -0.05
    randomization: PickAndPlaceRandomization = field(
        default_factory=PickAndPlaceRandomization
    )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["randomization"] = self.randomization.to_dict()
        return payload


@dataclass
class PickAndPlaceTrainingConfig:
    """High-level training parameters for PufferLib rollouts."""

    seed: int = 7
    num_envs: int = 8
    rollout_steps: int = 256
    minibatch_size: int = 2048
    update_epochs: int = 4
    device: str = field(default_factory=_default_device)
    checkpoint_dir: str = "models/puffer_pick"
    log_dir: str = "data/puffer_pick"
    env: PickAndPlaceEnvConfig = field(default_factory=PickAndPlaceEnvConfig)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["env"] = self.env.to_dict()
        return payload


def load_pick_training_config(path: str | Path) -> PickAndPlaceTrainingConfig:
    """Parse a YAML config into a dataclass tree."""
    yaml_path = Path(path)
    data = yaml.safe_load(yaml_path.read_text()) if yaml_path.exists() else {}
    env_data = (data or {}).get("env", {})
    rand_data = env_data.pop("randomization", None)

    randomization = (
        PickAndPlaceRandomization(**rand_data)
        if rand_data
        else PickAndPlaceRandomization()
    )
    env_config = PickAndPlaceEnvConfig(randomization=randomization, **env_data)
    train_kwargs = {k: v for k, v in (data or {}).items() if k != "env"}
    return PickAndPlaceTrainingConfig(env=env_config, **train_kwargs)


def dump_pick_training_config(
    config: PickAndPlaceTrainingConfig, path: str | Path, overwrite: bool = True
) -> None:
    """Write the config to disk for reproducibility."""
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
