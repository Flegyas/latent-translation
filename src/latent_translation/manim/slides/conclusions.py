from manim import *
from manim_editor import *
from powermanim.layouts.arrangedbullets import Bullet
from powermanim.templates.bulletlist import BulletList

from latent_translation.manim.utils import section_slide

FONT_SIZE = 38


class Conclusions(Scene):
    def construct(self):
        section_slide(self, "Conclusions")
        self.next_section("Latent Translation!", type=PresentationSectionType.NORMAL)

        conclusions = (
            Bullet(
                r"We achieve \textbf{zero-shot latent communication} in a variety of settings:",
                font_size=FONT_SIZE,
                level=0,
                group=0,
                symbol=None,
            ),
            Bullet(r"Both on generation and classification tasks", font_size=FONT_SIZE, level=1, group=1),
            Bullet(
                r"\textbf{Cross-architecture}: more than 10 pre-trained models!", font_size=FONT_SIZE, level=1, group=2
            ),
            Bullet(
                r"\textbf{Cross-modality}: stitching between vision and language latent spaces",
                font_size=FONT_SIZE,
                level=1,
                group=3,
            ),
            Bullet(r"And evaluate it on more than 10 datasets ", font_size=FONT_SIZE, level=1, group=4),
            Bullet(
                r"We observe that the transformation $\mathcal{T}$ can be constrained to be \textbf{orthogonal} without performance loss",
                font_size=FONT_SIZE,
                level=0,
                group=5,
                symbol=None,
                adjustment=DOWN / 2,
            ),
        )

        bulletlist = BulletList(
            *conclusions,
            line_spacing=MED_LARGE_BUFF * 0.8,
            left_buff=MED_LARGE_BUFF * 0.9,
            global_shift=DOWN * 0.4,
            scale_active=1.025,
            inactive_opacity=0.35,
        ).center()

        self.play(
            AnimationGroup(
                FadeIn(bulletlist),
                lag_ratio=0.5,
            ),
            run_time=1.25,
        )

        for group in range(max(x.group for x in conclusions) + 1):
            self.wait(0.1)
            self.next_section("Next", type=PresentationSectionType.NORMAL)
            to_play = [bulletlist.only_next()]
            if group == 5:
                to_play.append(Circumscribe(conclusions[group], color=BLUE, buff=SMALL_BUFF))
            self.play(*to_play)

        self.next_section("Reset", type=PresentationSectionType.SKIP)
        self.play(
            AnimationGroup(
                AnimationGroup(bulletlist.clear(), run_time=0.5),
                Uncreate(bulletlist),
                lag_ratio=0.7,
            ),
            run_time=1.5,
        )
        self.wait(0.1)
