from manim import *
from manim_editor import *

from nn_core.common import PROJECT_ROOT

FONT_SIZE = 38

from powermanim.components.powermanim import PowerManim


class Thanks(Scene):
    def construct(self):
        thankyou = Tex("Thank you!", font_size=84)
        # poster = Tex(r"\emph{come at our poster \textbf{\#73} at 11:30 a.m.}", font_size=32)
        # poster.next_to(thankyou, DOWN, buff=1)
        # VGroup(thankyou, poster).move_to(ORIGIN)

        # Sponsor
        erc = (
            ImageMobject(PROJECT_ROOT / "data" / "assets" / "logos" / "erc.jpg")
            # .set_opacity(0.75)
            .scale(0.25).to_corner(UR)
        )

        # Sponsor
        # qrcode = (
        #     ImageMobject(PROJECT_ROOT / "data" / "assets" / "logos" / "qrcode.png")
        #     # .set_opacity(0.75)
        #     .scale(0.5 * 0.6).to_corner(UL)
        # )

        # Powered by...
        # powered = Tex(r"\emph{Powered by...}", font_size=28).to_edge(LEFT).shift(DOWN)

        banner = ManimBanner().scale(0.1).to_edge(LEFT, buff=LARGE_BUFF * 1.15)

        manimeditor = ImageMobject(PROJECT_ROOT / "data" / "assets" / "logos" / "manimeditor.png")
        manimeditor.scale_to_fit_height(banner.get_height()).align_to(banner, DOWN).align_to(
            banner.get_critical_point(RIGHT), LEFT
        )

        powermanim_banner = PowerManim(font_color=WHITE, logo_black=WHITE, gap=0.5).build_banner()
        powermanim_banner.scale_to_fit_height(banner.get_height() * 1.6)
        powermanim_banner.next_to(banner, DOWN).align_to(banner, LEFT)

        Group(banner, manimeditor, powermanim_banner).to_corner(DL).shift(RIGHT * 0.5)

        # Slides by...
        cc = Tex(r"\emph{...slides by \textbf{Valentino Maiorca} \& \textbf{Luca Moschella}}", font_size=28).to_corner(
            DR
        )

        self.play(
            AnimationGroup(
                AnimationGroup(
                    DrawBorderThenFill(thankyou, run_time=0.75),
                    Circumscribe(
                        thankyou,
                        color=BLUE,
                        buff=MED_LARGE_BUFF,
                    ),
                    lag_ratio=0.5,
                ),
                # AnimationGroup(
                #     Create(poster),
                #     ShowPassingFlash(Underline(poster, color=YELLOW)),
                #     lag_ratio=0.5,
                # ),
                lag_ratio=0.5,
            )
        )

        self.play(
            AnimationGroup(
                # FadeIn(qrcode),
                AnimationGroup(
                    FadeIn(erc),
                    AnimationGroup(
                        banner.create(),
                        FadeIn(manimeditor),
                        FadeIn(powermanim_banner),
                        lag_ratio=0.1,
                    ),
                ),
                Create(cc),
                lag_ratio=0.5,
            ),
            run_time=2,
        )

        # Expand banner
        manimeditor.add_updater(lambda m: m.align_to(banner.get_critical_point(RIGHT), LEFT))
        powermanim_banner.add_updater(lambda m: m.align_to(banner, LEFT))
        self.play(banner.expand(), run_time=1)
        self.wait(2)
