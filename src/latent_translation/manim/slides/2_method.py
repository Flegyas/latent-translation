import itertools
from typing import *
from lightning import seed_everything
from manim.opengl import Color
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from manim import *
from manim_editor import PresentationSectionType
import torch
from scipy.stats import ortho_group


dtype = torch.float32


POINTS = 100
ANCHORS_IDX = [0, 5]


def build_space(random: bool = False, N: int = 2):
    x = torch.linspace(-0.5, 0.5, 5, dtype=dtype)
    y = torch.linspace(-0.5, 0.5, 5, dtype=dtype)

    xx, yy = torch.meshgrid(x, y, indexing="ij")

    x_grid = torch.stack([xx, yy], dim=-1).reshape(-1, N)
    if random:
        x_grid *= torch.randn_like(x_grid)

    return x_grid


def svd_translation(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # """Compute the translation vector that aligns A to B using SVD."""
    assert A.size(1) == B.size(1)
    u, s, vt = torch.svd((B.T @ A).T)
    R = u @ vt.T
    return R, s


def transform_latents(
    x: torch.Tensor,
    norm_mode: Optional[str],
    seed: int,
    isometry_first: bool = True,
    translation: Optional[torch.Tensor] = None,
    noise: bool = False,
    N: int = 2,
) -> torch.Tensor:
    x = x.clone()
    if translation is not None:
        x += translation.unsqueeze(0)

    norm = torch.tensor([1], dtype=dtype)

    if norm_mode == "independent":
        seed_everything(seed=seed)
        norm = torch.abs((torch.randn(x.size(0), dtype=dtype) + 0.001) * 100)
    elif norm_mode == "consistent":
        grid_side: int = int(x.size(0) ** (1 / 2))
        norm = x.reshape(grid_side, grid_side, N).sum(dim=-1)
        norm = (norm**2).flatten()
        norm = (norm - norm.min()) / (norm.max() - norm.min()) + 1
    elif norm_mode == "smooth":
        grid_side: int = int(x.size(0) ** (1 / 2))
        norm = x.reshape(grid_side, grid_side, N).sum(dim=-1)
        norm = (norm**3).flatten()
        norm = (norm - norm.min()) / (norm.max() - norm.min()) + 1
    elif norm_mode == "fixed":
        x = x * 10

    if noise:
        x = x + torch.abs(torch.randn_like(x) * 0.05)

    if isometry_first:
        out = iso_transform(x, seed=seed) * norm.unsqueeze(-1)
    else:
        out = iso_transform(x * norm.unsqueeze(-1), seed=seed)

    return out


def iso_transform(x, seed: int = 42, dtype: torch.dtype = torch.float32, return_transform: bool = False):
    opt_isometry: np.ndarray = ortho_group.rvs(x.shape[-1], random_state=seed)
    opt_isometry: torch.Tensor = torch.as_tensor(opt_isometry, dtype=dtype)
    out = x @ opt_isometry

    if return_transform:
        return out, opt_isometry

    return out


def transform_latents(
    x: torch.Tensor,
    norm_mode: Optional[str],
    seed: int,
    isometry_first: bool = True,
    scale: float = 1,
    translation: Optional[torch.Tensor] = None,
    noise: bool = False,
    N: int = 2,
) -> torch.Tensor:
    x = x.clone() * scale
    if translation is not None:
        x += translation.unsqueeze(0)

    norm = torch.tensor([1], dtype=dtype)

    if norm_mode == "independent":
        seed_everything(seed=seed)
        norm = torch.abs((torch.randn(x.size(0), dtype=dtype) + 0.001) * 100)
    elif norm_mode == "consistent":
        grid_side: int = int(x.size(0) ** (1 / 2))
        norm = x.reshape(grid_side, grid_side, N).sum(dim=-1)
        norm = (norm**2).flatten()
        norm = (norm - norm.min()) / (norm.max() - norm.min()) + 1
    elif norm_mode == "smooth":
        grid_side: int = int(x.size(0) ** (1 / 2))
        norm = x.reshape(grid_side, grid_side, N).sum(dim=-1)
        norm = (norm**3).flatten()
        norm = (norm - norm.min()) / (norm.max() - norm.min()) + 1
    elif norm_mode == "fixed":
        x = x * 10

    if noise:
        x = x + torch.abs(torch.randn_like(x) * 0.05)

    if isometry_first:
        out = iso_transform(x, seed=seed) * norm.unsqueeze(-1)
    else:
        out = iso_transform(x * norm.unsqueeze(-1), seed=seed)

    return out


def random_transform(x: torch.Tensor, seed: int) -> torch.Tensor:
    seed_everything(seed=seed)
    random_matrix = torch.randn((x.size(1), x.size(1)), dtype=dtype)
    return x @ random_matrix


class ParallelOpt(Scene):
    def build_optimization(self, axes, run_opt: pd.DataFrame, sample_ids: Sequence[int], max_lim: int):
        left_dots = []
        left_anims = []
        for sample_id in sample_ids[:max_lim]:
            point_df = run_opt[run_opt["sample_id"] == sample_id]

            points = np.asarray([(x, y, 0) for x, y in zip(point_df.dim_0, point_df.dim_1)])
            # points = ((points - points.min(0)) / (points.max(0) - points.min())) * 2 - 1
            points = axes.coords_to_point(points)
            zero_point = points[0]
            point_target = int(point_df[point_df["step"] == 0].target)
            dot = Dot(zero_point, stroke_width=1, fill_opacity=0.75, color=TARGET_COLORS[point_target], radius=0.05)
            dot.target = point_target
            dot.sample_id = point_df.sample_id
            left_dots.append(dot)

            path = VMobject()
            path.set_points_smoothly(points)
            left_anims.append(MoveAlongPath(dot, path, rate_func=rate_functions.smooth))
        return left_dots, left_anims

    def build_pipeline(self, axis, source_space, target_space, colors):
        centered_source_space = source_space - source_space.mean(dim=0)
        centered_target_space = target_space - target_space.mean(dim=0)

        scaled_source_space = centered_source_space / source_space.std(dim=0)
        scaled_target_space = centered_target_space / target_space.std(dim=0)

        rotated_source_space = scaled_source_space @ svd_translation(scaled_source_space, scaled_target_space)[0]
        descaled_source_space = rotated_source_space * target_space.std(dim=0)
        decentered_source_space = descaled_source_space + target_space.mean(dim=0)

        dots = []
        anims = []
        previous_state = torch.cat([source_space, torch.zeros((source_space.size(0), 1), dtype=dtype)], dim=1)
        for next_state in (centered_source_space, scaled_source_space, rotated_source_space, decentered_source_space):
            state_dots = []
            state_anims = []
            # pad next_state adding 0 to last dimension
            next_state = torch.cat([next_state, torch.zeros((next_state.size(0), 1), dtype=dtype)], dim=1)
            previous_points = axis.coords_to_point(previous_state)
            next_points = axis.coords_to_point(next_state)
            for previous_point, next_point, color in zip(previous_points, next_points, colors):
                previous_dot = Dot(previous_point, stroke_width=1, fill_opacity=0.75, color=color, radius=0.05)
                previous_dot.generate_target()
                previous_dot.target.move_to(next_point)
                state_anims.append(MoveToTarget(previous_dot, rate_func=rate_functions.smooth))
                state_dots.append(previous_dot)

            anims.append(
                AnimationGroup(
                    *state_anims,
                    lag_ratio=0,
                )
            )
            dots.append(state_dots)

            previous_state = next_state

        return dots, anims

    def construct(self):
        self.next_section("Method", type=PresentationSectionType.NORMAL, skip_animations=False)

        x = build_space(random=False)
        y = transform_latents(
            x,
            translation=torch.tensor([1, 1.2]),
            norm_mode=None,
            seed=24,
            noise=True,
            scale=2,
        )

        colors = (np.arctan2(x[:, 1] - 0.5, x[:, 0] - 0.5) + np.pi).flatten()
        colors = plt.get_cmap("Spectral_r")(colors)
        colors = [rgba_to_color(color) for color in colors]

        # reconstructed_target_space = decentered_source_space.clone()

        NumberLine()
        axis_range = (-3, 3)
        left_axis = (
            Axes(
                x_range=axis_range,
                y_range=axis_range,
                # axis_config={"unit_size": 10},
                x_length=6,
                y_length=6,
                tips=True,
            ).to_edge(LEFT)
            # .set_color(GRAY)
        )
        right_axis = (
            Axes(
                x_range=axis_range,
                y_range=axis_range,
                # axis_config={"unit_size": 10},
                x_length=6,
                y_length=6,
                tips=True,
            ).to_edge(RIGHT)
            # .set_color(GRAY)
        )

        left_dots, pipeline_anims = self.build_pipeline(axis=left_axis, source_space=x, target_space=y, colors=colors)
        right_dots = [Dot(right_axis.coords_to_point(x, y, 0), radius=0.05, color=c) for (x, y), c in zip(y, colors)]

        self.play(
            Create(left_axis),
            Create(right_axis),
        )
        # self.play(*(Create(x) for x in left_dots[0]), run_time=0.1)
        self.play(*(Create(x) for x in right_dots), run_time=0.1)

        self.wait()
        self.play(*pipeline_anims, run_time=10)

        self.wait()
        self.next_section("Reset", type=PresentationSectionType.SKIP, skip_animations=False)

        self.play(
            AnimationGroup(*(Uncreate(x) for x in left_dots[0]), lag_ratio=0.2),
            AnimationGroup(*(Uncreate(x) for x in right_dots), lag_ratio=0.2),
            Uncreate(left_axis),
            Uncreate(right_axis),
            # Uncreate(midline),
            run_time=2,
        )
