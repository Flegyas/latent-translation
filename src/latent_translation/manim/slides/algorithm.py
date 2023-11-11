from manim import *
from manim_editor import PresentationSectionType
from powermanim.layouts.arrangedbullets import Bullet, MathBullet
from powermanim.templates.bulletlist import BulletList
from torchvision.datasets import CIFAR100
from torchvision.transforms import transforms

from nn_core.common import PROJECT_ROOT

from latent_translation.manim.utils import section_slide

DISABLED_OPACITY = 0.4
FONT_SIZE = 28
SCALE_ACTIVE = 1.25
DATASET1 = CIFAR100(root=PROJECT_ROOT / "data", train=True, download=True)
# same as Dataset1 but with grayscale conversion
DATASET2 = CIFAR100(root=PROJECT_ROOT / "data", train=True, download=True, transform=transforms.Grayscale())

ANCHORS_COLOR = RED
ANCHORS_POINT_COLORS = [RED_A, RED_C, RED_E]
SAMPLE_COLOR = GREEN
SIM_COLOR = BLUE


class Algorithm(Scene):
    def construct(self):
        section_slide(self, "Latent Translation")

        slide_title = Tex("Algorithm").to_edge(UP)

        self.next_section("Background algo", type=PresentationSectionType.NORMAL, skip_animations=False)
        self.play(Create(slide_title))

        algo = (
            Bullet(
                r"Select a subset $\mathbb{A_\mathbb{X}}$ of the training set $\mathbb{X}$",
                font_size=FONT_SIZE,
                group=0,
            ),
            Bullet(
                r"Select a \textbf{corresponding} $\mathbb{A_\mathbb{Y}}$ of the training set $\mathbb{Y}$",
                font_size=FONT_SIZE,
                group=1,
            ),
            Bullet(
                r"Apply the respective encoding functions $E_x$ and $E_y$",
                font_size=FONT_SIZE,
                group=2,
            ),
            Bullet(
                r"The \textbf{latent translation} operator $\mathcal{T}$ ",
                " is obtained via solving:",
                font_size=FONT_SIZE,
                group=3,
            ),
            MathBullet(
                r"\mathcal{T}(\mathbf{x}) = \mathbf{R} \mathbb{A_\mathbb{X}} + \mathbf{b}",
                font_size=FONT_SIZE + 5,
                symbol=None,
                level=1,
                group=3,
                adjustment=UP * MED_LARGE_BUFF * 1.25 * 0.25,
            ),
        )
        # algo[0][1].set_color(ANCHORS_COLOR)
        # algo[0][3].set_color(ANCHORS_COLOR)
        # algo[1][1].set_color(SAMPLE_COLOR)
        # algo[-2][1].set_color(SAMPLE_COLOR)
        # algo[3][1].set_color(SIM_COLOR)
        # for color, entity_idxs in (
        #     (SIM_COLOR, [3, 9, 15]),
        #     (SAMPLE_COLOR, [1, 5, 11, 17]),
        # ):
        #     for entity_idx in entity_idxs:
        #         algo[-1][entity_idx].set_color(color)
        # for color, entity_idx in zip(ANCHORS_POINT_COLORS, [7, 13, 19]):
        #     algo[-1][entity_idx].set_color(color)

        bulletlist = BulletList(
            *algo,
            line_spacing=MED_LARGE_BUFF * 1.3,
            indent_buff=MED_LARGE_BUFF * 1.25,
            left_buff=MED_LARGE_BUFF,
            scale_active=1.25,
            global_shift=DOWN * 0.275,
        )
        self.play(FadeIn(bulletlist), run_time=0.5)

        self.wait(0.5)
        self.next_section("Anchors", type=PresentationSectionType.NORMAL, skip_animations=False)
        anchor_images1 = (
            Group(*[ImageMobject(DATASET1[sample_idx][0], image_mode="RGB") for sample_idx in [0, 1, 3]])
            .scale(3.5)
            .arrange(RIGHT, buff=MED_LARGE_BUFF * 2)
            .to_edge(RIGHT)
            .shift(UP)
        )
        anchor_images2 = (
            Group(*[ImageMobject(DATASET2[sample_idx][0], image_mode="RGB") for sample_idx in [0, 1, 3]])
            .scale(3.5)
            .arrange(RIGHT, buff=MED_LARGE_BUFF * 2)
            .to_edge(RIGHT)
            .next_to(anchor_images1, DOWN, buff=MED_LARGE_BUFF * 2)
        )

        corresponding_anchors = []
        for anchor_image1, anchor_image2 in zip(anchor_images1.submobjects, anchor_images2.submobjects):
            corresponding_anchors.append(
                Group(
                    Line(
                        anchor_image1.get_bottom() + DOWN * 0.25,
                        anchor_image2.get_top() + UP * 0.25,
                        color=ANCHORS_COLOR,
                        stroke_width=5,
                    ),
                    anchor_image2,
                )
            )

        semantic_alignment = SurroundingRectangle(
            Group(anchor_images1, *corresponding_anchors), color=ANCHORS_COLOR, stroke_width=5
        )
        semantic_alignment_label = Tex(r"\textbf{Semantic Alignment!}", font_size=38).next_to(semantic_alignment, UP)
        semantic_alignment = Group(semantic_alignment, semantic_alignment_label)

        # anchor1
        self.play(
            AnimationGroup(
                bulletlist.only_next(),
                run_time=1,
            ),
            FadeIn(anchor_images1),
        )
        self.wait(0.5)

        # anchor2
        self.play(
            AnimationGroup(
                bulletlist.only_next(),
                run_time=1,
            ),
            AnimationGroup(*[FadeIn(correspondence) for correspondence in corresponding_anchors], lag_ratio=0.7),
        )
        self.play(FadeIn(semantic_alignment))
        self.wait(0.5)

        self.next_section("Encoding", type=PresentationSectionType.NORMAL, skip_animations=False)
        anims = []
        dots = []
        for image, color in zip(anchor_images1.submobjects, ANCHORS_POINT_COLORS):
            anims.append(
                AnimationGroup(
                    FadeOut(image),
                    FadeIn(d := Dot(point=image.get_center(), radius=0.15, color=color, fill_opacity=1, z_index=1)),
                    lag_ratio=0.5,
                )
            )
            dots.append(d)

        for image, color in zip(anchor_images2.submobjects, ANCHORS_POINT_COLORS):
            anims.append(
                AnimationGroup(
                    FadeOut(image),
                    # create a triangle centered in image.get_center()
                    FadeIn(
                        d := Polygon(
                            image.get_center() + UP * 0.25,
                            image.get_center() + LEFT * 0.15,
                            image.get_center() + RIGHT * 0.15,
                            color=color,
                            fill_opacity=1,
                            z_index=1,
                        )
                    ),
                    lag_ratio=0.5,
                )
            )
            dots.append(d)

        self.play(
            AnimationGroup(
                AnimationGroup(
                    bulletlist.only_next(),
                    run_time=1,
                ),
                AnimationGroup(
                    *anims,
                    lag_ratio=0.5,
                    run_time=2.5,
                ),
                lag_ratio=0.5,
            ),
        )

        arrow = Arrow(0.2 * RIGHT, 0.2 * LEFT).next_to(bulletlist[-1], RIGHT, buff=MED_LARGE_BUFF)
        arrow_label = Tex(r"\textbf{Mostly orthogonal!}", font_size=38).next_to(arrow, RIGHT)

        self.play(bulletlist.only_next(), GrowArrow(arrow), FadeIn(arrow_label))

        # # self.wait(0.5)
        # # self.next_section("Sim", type=PresentationSectionType.NORMAL, skip_animations=False)
        # # self.play(
        # #     AnimationGroup(
        # #         bulletlist.only_next(),
        # #         run_time=1,
        # #     )
        # # )

        # # self.wait(0.5)
        # # self.next_section("RelRep", type=PresentationSectionType.NORMAL, skip_animations=False)
        # # lines_anim = []
        # # lines = []
        # # for anchor in anchors_dots:
        # #     lines_anim.append(
        # #         Create(l := Line(anchor.get_center(), sample_dot.get_center(), color=SIM_COLOR).set_opacity(0.5))
        # #     )
        # #     lines.append(l)
        # # self.play(
        # #     AnimationGroup(
        # #         bulletlist.only_next(),
        # #         run_time=1,
        # #     ),
        # #     AnimationGroup(*lines_anim, lag_ratio=0.5, run_time=3),
        # # )

        # self.wait(0.5)
        # # self.next_section("Code", type=PresentationSectionType.NORMAL, skip_animations=False)
        # #         rendered_code = (
        # #             Code(
        # #                 code="""import torch
        # # import torch.nn.functional as F

        # # def latent_translation(x, y):
        # # \tx = F.normalize(x, p=2, dim=-1)
        # # \tanchors = F.normalize(anchors, p=2, dim=-1)
        # # \treturn torch.einsum("nd, ad -> na", x, anchors)""",
        # #                 tab_width=3,
        # #                 language="Python",
        # #                 style="monokai",
        # #                 insert_line_no=False,
        # #                 font_size=18,
        # #             )
        # #             .to_edge(LEFT, buff=LARGE_BUFF)
        # #             .align_to(
        # #                 VGroup(*dots),
        # #                 direction=UP,
        # #             )
        # #         )

        # # cosine_label = Tex("$sim$", " $=$ ", "cosine similarity", font_size=38).next_to(rendered_code, UP)
        # # cosine_label[0].set_color(SIM_COLOR)

        # # self.play(
        # #     AnimationGroup(
        # #         FadeOut(bulletlist, shift=LEFT),
        # #         AnimationGroup(
        # #             FadeIn(cosine_label, shift=RIGHT),
        # #             # FadeIn(rendered_code, shift=RIGHT),
        # #             lag_ratio=0.5,
        # #         ),
        # #         lag_ratio=0.75,
        # #         run_time=1.5,
        # #     )
        # # )

        # # differentiable = Tex("(differentiable!)").next_to(rendered_code, DOWN, buff=LARGE_BUFF).set_opacity(0.5)
        # # self.play(FadeIn(differentiable))

        # self.wait()
        # # self.next_section("Properties", type=PresentationSectionType.NORMAL)

        # # title = Tex(f"Properties").to_edge(UP)

        # # properties = (
        # #     MathBullet(
        # #         r"\mathbf{r}_{",
        # #         r"\mathbf{x}^{(i)}",
        # #         r"} \text{ is a universal representation computed \emph{indipendently} for each space }",
        # #         font_size=FONT_SIZE,
        # #     ),
        # #     MathBullet(
        # #         r"\text{The size of } \mathbf{r}_{",
        # #         r"\mathbf{x}^{(i)}",
        # #         r"} \text{ depends on the number of }",
        # #         r"\text{anchors }",
        # #         r"|",
        # #         r"\mathbb{A}",
        # #         r"|",
        # #         font_size=FONT_SIZE,
        # #     ),
        # #     Bullet(
        # #         r"The ",
        # #         r"anchors",
        # #         r" and ",
        # #         r"similarity",
        # #         r" function choices determine the representation \emph{properties}",
        # #         font_size=FONT_SIZE,
        # #         force_inline=True,
        # #     ),
        # # )
        # # properties[0][1].set_color(SAMPLE_COLOR)

        # # properties[1][1].set_color(SAMPLE_COLOR)
        # # properties[1][3].set_color(ANCHORS_COLOR)
        # # properties[1][5].set_color(ANCHORS_COLOR)

        # # properties[2][1].set_color(ANCHORS_COLOR)
        # # properties[2][3].set_color(SIM_COLOR)

        # # properties_list = BulletList(
        # #     *properties,
        # #     left_buff=MED_LARGE_BUFF * 1.5,
        # #     scale_active=1.15,
        # # )

        # # self.play(
        # #     AnimationGroup(
        # #         AnimationGroup(
        # #             *(
        # #                 Uncreate(x)
        # #                 for x in [
        # #                     differentiable,
        # #                     cosine_label,
        # #                     rendered_code,
        # #                     *lines,
        # #                     *anchors_dots,
        # #                     sample_dot,
        # #                     anchors_brace,
        # #                     sample_brace,
        # #                 ]
        # #             )
        # #         ),
        # #         ReplacementTransform(slide_title, title),
        # #         FadeIn(properties_list),
        # #         lag_ratio=0.5,
        # #     )
        # # )

        # # self.wait()
        # # self.next_section("First property", type=PresentationSectionType.NORMAL)
        # # self.play(properties_list.also_next())

        # # self.wait()
        # # self.next_section("Second property", type=PresentationSectionType.NORMAL)
        # # self.play(properties_list.also_next())

        # # self.wait()
        # # self.next_section("Third property", type=PresentationSectionType.NORMAL)
        # # self.play(properties_list.also_next())

        # # self.wait()
        # # self.next_section("Consequence", type=PresentationSectionType.NORMAL)

        # # cosine = Tex("$\mathbf{sim}$", " $=$ ", "cosine similarity", font_size=34).shift(DOWN)
        # # cosine[0].set_color(SIM_COLOR)

        # # arrow = Arrow(0.5 * UP, 0.5 * DOWN).next_to(cosine, DOWN, buff=MED_LARGE_BUFF)
        # # invariance = Tex(
        # #     r"invariant to \textbf{angle-preserving transformations} of the latent space", font_size=34
        # # ).next_to(arrow, DOWN, buff=MED_LARGE_BUFF)

        # # self.play(
        # #     AnimationGroup(
        # #         properties_list.animate.shift(UP * 1.2),
        # #         AnimationGroup(
        # #             AnimationGroup(FadeIn(cosine)),
        # #             GrowArrow(arrow),
        # #             Create(invariance),
        # #             lag_ratio=0.75,
        # #         ),
        # #         lag_ratio=0.15,
        # #     ),
        # #     run_time=3,
        # # )
        # # self.play(
        # #     AnimationGroup(
        # #         ShowPassingFlash(Underline(cosine, color=YELLOW)),
        # #         ShowPassingFlash(Underline(invariance, color=YELLOW)),
        # #         lag_ratio=0.5,
        # #         run_time=2,
        # #     )
        # # )

        # # self.wait()
        # # self.next_section("Reset", type=PresentationSectionType.SKIP)
        # # self.play(*(Uncreate(x) for x in (title, properties_list, cosine, arrow, invariance)))

        # # self.wait(0.1)
