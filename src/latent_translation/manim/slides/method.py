from typing import *

import numpy as np
import torch
from lightning import seed_everything
from manim import *
from manim_editor import PresentationSectionType
from matplotlib import pyplot as plt
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


class Method(Scene):
    def build_pipeline(self, axis, source_space, target_space, colors):
        centered_source_space = source_space - source_space.mean(dim=0)
        centered_target_space = target_space - target_space.mean(dim=0)

        scaled_source_space = centered_source_space / source_space.std(dim=0)
        scaled_target_space = centered_target_space / target_space.std(dim=0)

        rotated_source_space = scaled_source_space @ svd_translation(scaled_source_space, scaled_target_space)[0]
        descaled_source_space = rotated_source_space * target_space.std(dim=0)
        decentered_source_space = descaled_source_space + target_space.mean(dim=0)

        previous_step = torch.cat([source_space, torch.zeros((source_space.size(0), 1), dtype=dtype)], dim=1)
        previous_step = [
            Dot(axis.coords_to_point(*x), stroke_width=1, fill_opacity=0.75, color=c, radius=0.05)
            for x, c in zip(previous_step, colors)
        ]
        dots = []
        anims = []
        label = None
        for next_step, next_step_name in (
            (centered_source_space, "Centering"),
            (scaled_source_space, "Scaling"),
            (rotated_source_space, r"$\mathcal{T}$ application"),
            (descaled_source_space, "Descaling"),
            (decentered_source_space, "Decentering"),
        ):
            step_dots = []
            step_anims = []
            next_step = torch.cat([next_step, torch.zeros((next_step.size(0), 1), dtype=dtype)], dim=1)
            next_points = axis.coords_to_point(next_step)

            for previous_dot, next_point in zip(previous_step, next_points):
                previous_dot.generate_target()
                previous_dot.target.move_to(next_point)
                step_anims.append(MoveToTarget(previous_dot, rate_func=rate_functions.smooth))
                step_dots.append(previous_dot)

            label = Tex(next_step_name).next_to(axis, DOWN, buff=LARGE_BUFF)
            anims.append(
                AnimationGroup(
                    Unwrite(label) if label is not None else None,
                    AnimationGroup(
                        FadeIn(label),
                        AnimationGroup(*step_anims, lag_ratio=0),
                        lag_ratio=1,
                    ),
                )
            )

            dots.append(step_dots)

        return dots, anims

    def construct(self):
        self.next_section("Method", type=PresentationSectionType.NORMAL, skip_animations=False)

        x = build_space(random=False)
        colors = (np.arctan2(x[:, 1] - 0.5, x[:, 0] - 0.5) + np.pi).flatten()

        x = x + torch.tensor([0.5, 0.5])
        y = transform_latents(
            x,
            translation=torch.tensor([1, 1.2]),
            norm_mode=None,
            seed=24,
            noise=True,
            scale=2,
        )

        colors = plt.get_cmap("Spectral_r")(colors)
        colors = [rgba_to_color(color) for color in colors]

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
        left_axis_label = Tex(r"Source space").next_to(left_axis, UP, buff=LARGE_BUFF)
        left_axis_block = VGroup(left_axis_label, left_axis)
        left_axis_block.scale(0.7)

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
        right_axis_label = Tex(r"Target space").next_to(right_axis, UP, buff=LARGE_BUFF)
        right_axis_block = VGroup(right_axis_label, right_axis)
        right_axis_block.scale(0.7)

        left_dots, pipeline_anims = self.build_pipeline(axis=left_axis, source_space=x, target_space=y, colors=colors)
        right_dots = [Dot(right_axis.coords_to_point(x, y, 0), radius=0.05, color=c) for (x, y), c in zip(y, colors)]

        self.play(
            Create(left_axis_block),
            Create(right_axis_block),
        )
        self.play(*(Create(x) for x in left_dots[0]), run_time=0.1)
        self.play(*(Create(x) for x in right_dots), run_time=0.1)

        self.wait()
        for pipeline_step in pipeline_anims:
            self.play(pipeline_step, run_time=1)

        self.wait()

        self.play(
            Uncreate(left_axis_block),
            Uncreate(right_axis_block),
        )

        self.wait()

        approx_symbol = Tex(r"$\approx$")
        to_compare = VGroup(VGroup(*left_dots[0]), approx_symbol, VGroup(*right_dots))
        self.play(
            AnimationGroup(
                to_compare.animate.arrange(RIGHT, buff=LARGE_BUFF * 2).center(), Write(approx_symbol), lag_ratio=0.5
            ),
            run_time=2,
        )

        self.next_section("Reset", type=PresentationSectionType.SKIP, skip_animations=False)

        self.play(
            AnimationGroup(*(Uncreate(x) for x in to_compare), lag_ratio=0.2),
            run_time=2,
        )
