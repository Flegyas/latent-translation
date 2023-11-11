from typing import *

from manim import *
from manim_editor import *


def section_slide(scene, section_title: str, math: bool = False):
    scene.next_section(section_title, type=PresentationSectionType.NORMAL)

    text_obj = Tex if not math else MathTex
    slide_title = text_obj(section_title, font_size=72)
    scene.play(
        AnimationGroup(
            DrawBorderThenFill(slide_title, run_time=0.75),
            Circumscribe(
                slide_title,
                color=BLUE,
                buff=MED_LARGE_BUFF,
            ),
            lag_ratio=0.5,
        )
    )

    scene.wait()
    scene.next_section("Reset", type=PresentationSectionType.SKIP)
    scene.play(
        FadeOut(slide_title, shift=UP),
        run_time=0.25,
    )


def vgroup_hightlight(
    x: VGroup,
    scale_active: float = 1.0,
    active_idxs=None,
    previously_active_idxs=None,
) -> Mobject:
    anims = []

    if active_idxs is not None:
        if not isinstance(active_idxs, Sequence):
            active_idxs = [active_idxs]

        for active_idx in active_idxs:
            x.submobjects[active_idx].target = x.submobjects[active_idx].saved_state.copy()
            x.submobjects[active_idx].target.scale(scale_active)
            x.submobjects[active_idx].target.set_opacity(1)
            anims.append(MoveToTarget(x.submobjects[active_idx]))

    if previously_active_idxs is not None:
        if not isinstance(previously_active_idxs, Sequence):
            previously_active_idxs = [previously_active_idxs]

        for previously_active_idx in previously_active_idxs:
            anims.append(
                x.submobjects[previously_active_idx].animate.restore(),
            )

    return anims
