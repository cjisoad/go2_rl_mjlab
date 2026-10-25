"""Fixed, compact terrain courses used only for interactive policy playback."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np
from mjlab.terrains import BoxFlatTerrainCfg
from mjlab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGeneratorCfg,
    TerrainOutput,
)

PlayTerrain = Literal["auto", "original", "strip"]


@dataclass(kw_only=True)
class StripTerrainCfg(SubTerrainCfg):
    sections: dict[str, SubTerrainCfg]
    start_length: float = 6.0
    section_length: float = 8.0
    shoulder_width: float = 1.0

    @property
    def layout(self):
        x = -self.size[0] / 2
        result = [("start", x, x + self.start_length)]
        x += self.start_length
        for name in self.sections:
            result.append((name, x, x + self.section_length))
            x += self.section_length
        return result

    def section_at(self, x):
        return next(
            (name for name, start, end in self.layout if start <= x < end), "outside"
        )

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        geometries = []
        # Keep the original 8x8 patches: the native box-grid terrain requires
        # square patches. Two flat shoulders make the course 10 m wide.
        patches = [("start", BoxFlatTerrainCfg(), self.start_length)]
        patches += [
            (name, cfg, self.section_length) for name, cfg in self.sections.items()
        ]
        x = 0.0
        for _name, source, length in patches:
            section = deepcopy(source)
            section.size = (length, self.size[1] - 2 * self.shoulder_width)
            output = section.function(difficulty, spec, rng)
            offset = np.array([x, self.shoulder_width, 0.0])
            for geometry in output.geometries:
                geometry.geom.pos = np.asarray(geometry.geom.pos) + offset
            geometries.extend(output.geometries)
            x += length
        for y in (0.0, self.size[1] - self.shoulder_width):
            shoulder = BoxFlatTerrainCfg(size=(self.size[0], self.shoulder_width))
            output = shoulder.function(difficulty, spec, rng)
            for geometry in output.geometries:
                geometry.geom.pos = np.asarray(geometry.geom.pos) + (0.0, y, 0.0)
            geometries.extend(output.geometries)
        # No full-width floor under the course: it would fill inverted terrain.
        return TerrainOutput(
            origin=np.array([self.start_length / 2, self.size[1] / 2, 0.0]),
            geometries=geometries,
        )


def configure_play_terrain(cfg, mode: PlayTerrain = "auto", *, keyboard=False):
    """Configure a freshly loaded play env; task/training factories stay intact."""
    if keyboard:
        cfg.viewer.enable_shadows = False
        cfg.viewer.enable_reflections = False
    terrain = cfg.scene.terrain
    generator = None if terrain is None else terrain.terrain_generator
    use_strip = mode == "strip" or (
        mode == "auto" and keyboard and generator is not None
    )
    if not use_strip:
        return
    if generator is None:
        raise ValueError("--play-terrain strip requires a task with generated terrain.")
    sections = deepcopy(generator.sub_terrains)
    size = (6.0 + 8.0 * len(sections), 10.0)
    strip = StripTerrainCfg(size=size, sections=sections)
    terrain.terrain_generator = TerrainGeneratorCfg(
        seed=42,
        size=size,
        num_rows=1,
        num_cols=1,
        sub_terrains={"strip": strip},
        difficulty_range=(0.55, 0.55),
        border_width=1.0,
        add_lights=False,
    )
    terrain.max_init_terrain_level = 0
    # Warp imports the model at its zero pose before resetting to the start.
    # That pose intersects the middle section (96 contacts in the Rough strip).
    if cfg.sim.nconmax is not None:
        cfg.sim.nconmax = max(cfg.sim.nconmax, 128)
    cfg.events.pop("randomize_terrain", None)
    cfg.curriculum = {}
    pose = cfg.events["reset_base"].params["pose_range"]
    pose.update(x=(0.0, 0.0), y=(0.0, 0.0), yaw=(0.0, 0.0))
    # The source DreamWaQ episode lasts 20 s, shorter than traversing the strip.
    cfg.episode_length_s = 3600.0
