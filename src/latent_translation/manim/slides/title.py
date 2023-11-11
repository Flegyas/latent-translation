from manim import *
from manim_editor import PresentationSectionType

from nn_core.common import PROJECT_ROOT


class Title(Scene):
    def construct(self):
        paper_title = Tex("Latent Space Translation", font_size=72).to_edge(LEFT).shift(UP * 2.5)
        paper_title_sub = Tex("via Semantic Alignment", font_size=52)
        paper_title_sub.next_to(paper_title, DOWN)
        paper_title_sub.align_to(paper_title, LEFT)

        top_authors = VGroup(
            *[
                Tex(x, font_size=40)
                for x in (
                    r"\textbf{Valentino Maiorca}",
                    r"\textbf{Luca Moschella}",
                )
            ]
        )

        other_authors = VGroup(
            *[
                Tex(x, font_size=40)
                for x in ("Antonio Norelli", "Marco Fumero", "Francesco Locatello", "Emanuele Rodolà")
            ]
        )

        top_authors.arrange(RIGHT, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER * 1.5).shift(DOWN * 0.25)
        other_authors.arrange(RIGHT, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER * 1.5).next_to(top_authors, DOWN, buff=0.5)

        sapienzalogo = (
            ImageMobject(PROJECT_ROOT / "data" / "assets" / "logos" / "sapienza_logo.png")
            .scale(0.1)
            .to_corner(DL)
            .set_opacity(0.75)
        )
        gladialogo = (
            ImageMobject(PROJECT_ROOT / "data" / "assets" / "logos" / "logo_gladia_white.png")
            .rescale_to_fit(length=sapienzalogo.height, dim=1)
            .to_corner(DR)
        )
        iclr_logo = (
            ImageMobject(PROJECT_ROOT / "data" / "assets" / "logos" / "neurips_logo.png")
            .rescale_to_fit(length=sapienzalogo.height, dim=1)
            .to_corner(DR)
        )
        gladialogo.next_to(sapienzalogo, RIGHT, buff=LARGE_BUFF)
        iclr_logo.next_to(gladialogo, RIGHT, buff=LARGE_BUFF)
        Group(gladialogo, sapienzalogo, iclr_logo).move_to(ORIGIN).to_edge(DOWN)

        authors_group = Group(top_authors, other_authors, Group(sapienzalogo, iclr_logo, gladialogo))

        self.play(
            Succession(
                FadeIn(paper_title, shift=DOWN, run_time=0.5),
                Create(paper_title_sub, shift=DOWN, run_time=2),
            ),
            AnimationGroup(
                *(FadeIn(x, shift=UP, run_time=1.5) for x in authors_group),
                run_time=2,
                lag_ratio=0.2,
            ),
        )

        self.wait()
        self.next_section("Uncreate Title", type=PresentationSectionType.SKIP)

        self.play(
            Uncreate(paper_title),
            Uncreate(paper_title_sub),
            AnimationGroup(
                *(FadeOut(x, shift=DOWN, run_time=1.5) for x in authors_group[::-1]),
                run_time=2,
                lag_ratio=0.2,
            ),
        )
