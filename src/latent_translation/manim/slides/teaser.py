from manim import *
from manim_editor import PresentationSectionType
from manim_mobject_svg import *

from latent_translation.manim.utils import section_slide

_ABS_COLOR = TEAL_D
_REL_COLOR = GOLD_D

font_size: int = 20
scale: float = 0.7

custom_tex_template = TexTemplate()
custom_tex_template.add_to_preamble(r"\usepackage{bm}")


class Teaser(Scene):
    def _build_space(self, enc_type: str, label: str):
        if enc_type == "relative":
            color = _REL_COLOR
        elif enc_type == "absolute":
            color = _ABS_COLOR
        else:
            raise NotImplementedError

        item = Square(color=color, fill_opacity=0.7).scale(scale)
        label = MathTex(label).scale_to_fit_width(item.width / 2 - 0.1)
        label.move_to(item.get_center())
        return VGroup(item, label)

    def _build_decoder(self, enc_type: str, label: str):
        if enc_type == "relative":
            color = _REL_COLOR
        elif enc_type == "absolute":
            color = _ABS_COLOR
        else:
            raise NotImplementedError

        item = (
            Polygon(RIGHT + 2 * UP, RIGHT + 2 * DOWN, DOWN + LEFT, UP + LEFT, fill_opacity=0.7, color=color)
            # .set_opacity(START_MODULE_OPACITY)
            .scale(scale)
        ).rotate(-PI / 2)
        label = MathTex(label).scale_to_fit_width(item.height - 0.1)
        label.move_to(item.get_center())
        return VGroup(item, label)

    def _build_projection_arrow(
        self,
        start,
        end,
        color2,
        label: str,
        label_side,
        color1=None,
        label_rotation: float = PI * 2,
        diagonal: bool = False,
        label_shift: float = [0, 0, 0],
    ):
        obj = DashedLine(start=start, end=end, color=GREY_B, fill_opacity=0.7, dashed_ratio=0.35)
        obj.add_tip(tip_shape=ArrowTriangleTip)

        if color1 is not None:
            obj.add_tip(tip_shape=ArrowTriangleTip)
            obj.get_tips()[1].set_color(color1)

        obj.get_tips()[0].set_color(color2)

        label = MathTex(label)
        label.next_to(obj.get_start() if diagonal else obj, label_side).rotate(label_rotation).shift(label_shift)

        return VGroup(obj, label)

    def _build_encoding_arrow(
        self,
        start,
        end,
        color2,
        label: str,
        label_side,
        color1=None,
        label_rotation: float = PI * 2,
        diagonal: bool = False,
        shift: float = [0, 0, 0],
    ):
        obj = Arrow(start=start, end=end, color=GREY_B, fill_opacity=0.7, stroke_width=15)
        obj.get_tips()[0].set_color(color2)

        if color1 is not None:
            obj.add_tip(tip_shape=ArrowTriangleTip)
            obj.get_tips()[1].set_color(color1)

        label = MathTex(label)
        label.next_to(obj.get_start() if diagonal else obj, label_side).rotate(label_rotation).shift(shift)

        return VGroup(obj, label)

    def _build_legend(self):
        legend = (
            VGroup(
                VGroup(
                    Square(color=_ABS_COLOR, fill_opacity=0.7).scale(0.1),
                    Tex(r"Absolute").scale(0.7),
                ).arrange(RIGHT, buff=SMALL_BUFF),
                VGroup(
                    Square(color=_REL_COLOR, fill_opacity=0.7).scale(0.1),
                    Tex(r"Relative").scale(0.7),
                ).arrange(RIGHT, buff=SMALL_BUFF),
            )
            .arrange_in_grid(rows=2, cols=1, cell_alignment=LEFT, buff=MED_SMALL_BUFF)
            .to_corner(DL)
            .shift(RIGHT / 2)
        )

        return legend

    def construct(self):
        section_slide(self, "Latent Translation")

        abs1 = self._build_space(enc_type="absolute", label=r"\mathbf{X}")  # .to_edge(LEFT)
        abs2 = self._build_space(enc_type="absolute", label=r"\mathbf{Y}")  # .next_to(abs1, UP)
        rel = self._build_space(enc_type="relative", label=r"\mathbf{Z}")  # .to_edge(RIGHT)

        rel_dec = self._build_decoder(enc_type="relative", label=r"dec_\mathbf{Z}")  # .to_edge(RIGHT)
        dec1 = self._build_decoder(enc_type="absolute", label=r"dec_\mathbf{X}")  # .next_to(rel1, UP)
        dec2 = self._build_decoder(enc_type="absolute", label=r"dec_\mathbf{Y}")  # .next_to(rel1, UP)

        block1 = VGroup(abs1, dec1).arrange(DOWN, buff=LARGE_BUFF).to_edge(UP, buff=LARGE_BUFF)
        block2 = VGroup(abs2, dec2).arrange(DOWN, buff=LARGE_BUFF).to_edge(UP, buff=LARGE_BUFF)
        block_rel = VGroup(rel, rel_dec).arrange(DOWN, buff=LARGE_BUFF)
        blocks: VGroup = VGroup(
            block1,
            block_rel,
            block2,
        ).arrange(RIGHT, buff=LARGE_BUFF * 2)
        block_rel.shift(DOWN)

        rel_proj1 = self._build_projection_arrow(
            start=abs1.get_right() + [0, -0.2, 0],
            end=rel.get_left(),
            color2=_REL_COLOR,
            label="\mathbf{X} \cdot \mathbf{A^{T}_{\mathbf{X}}}",
            # label="rel(\mathbf{X}, \mathbf{A_{\mathbf{X}}})",
            label_side=UP,
            label_rotation=-PI / 14,
            label_shift=[0, -1.3, 0],
        )

        rel_proj2 = self._build_projection_arrow(
            start=abs2.get_left() + [0, -0.2, 0],
            end=rel.get_right(),
            color2=_REL_COLOR,
            label="\mathbf{Y} \cdot \mathbf{A^{T}_{\mathbf{Y}}}",
            # label="rel(\mathbf{Y}, \mathbf{A_{\mathbf{Y}}})",
            label_side=UP,
            label_rotation=PI / 14,
            label_shift=[0, -1.3, 0],
        )

        decoding_abs1 = self._build_encoding_arrow(
            start=abs1.get_bottom(),
            end=dec1.get_top(),
            color2=_ABS_COLOR,
            label="",
            label_side=UP,
        )

        decoding_abs2 = self._build_encoding_arrow(
            start=abs2.get_bottom(),
            end=dec2.get_top(),
            color2=_ABS_COLOR,
            label="",
            label_side=UP,
        )

        decoding_rel = self._build_encoding_arrow(
            start=rel.get_bottom(),
            end=rel_dec.get_top(),
            color2=_REL_COLOR,
            label="",
            label_side=UP,
        )

        r = self._build_projection_arrow(
            start=abs1.get_right() + [0, 0.3, 0],
            end=abs2.get_left() + [0, 0.3, 0],
            color2=_ABS_COLOR,
            label="\mathcal{T}(\mathbf{X})}",
            label_side=UP,
            label_shift=[0, -0.3, 0],
            # label_rotation=PI / 2,
        )
        inverse_r = self._build_projection_arrow(
            start=abs2.get_top() + [0.3, 0, 0],
            end=abs1.get_bottom() + [0.3, 0, 0],
            color2=_ABS_COLOR,
            label="\mathcal{T}^{-1}(\mathbf{Y})}",
            label_side=LEFT,
            label_shift=[1.5, 0, 0],
            label_rotation=-PI / 2,
        )

        self.play(
            AnimationGroup(
                FadeIn(abs1),
                FadeIn(abs2),
                lag_ratio=0,
            )
        )
        self.next_section("Absolute decoding", type=PresentationSectionType.NORMAL)

        self.play(
            AnimationGroup(
                FadeIn(decoding_abs1),
                FadeIn(dec1),
                FadeIn(decoding_abs2),
                FadeIn(dec2),
                lag_ratio=0,
            )
        )

        self.next_section("Relative projection", type=PresentationSectionType.NORMAL)

        legend = self._build_legend()
        self.play(
            AnimationGroup(
                AnimationGroup(
                    FadeIn(rel_proj1),
                    FadeIn(rel_proj2),
                    # FadeIn(rel_dec),
                    lag_ratio=0,
                ),
                FadeIn(rel),
                FadeIn(legend),
                lag_ratio=0,
            )
        )

        self.play(AnimationGroup(FadeIn(decoding_rel), FadeIn(rel_dec), lag_ratio=0))

        self.next_section("Remove relative", type=PresentationSectionType.NORMAL)

        rel_block = VGroup(rel, decoding_rel, rel_dec, rel_proj1, rel_proj2)
        cross = Cross(rel_block, stroke_width=10)
        self.play(Create(cross))

        self.play(
            rel_block.animate.set_opacity(0.1),
            cross.animate.set_opacity(0.1),
        )

        self.next_section("Direct translation", type=PresentationSectionType.NORMAL)
        self.play(
            AnimationGroup(
                FadeIn(r),
                lag_ratio=0,
            )
        )
