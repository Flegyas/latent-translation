from manim import *
from manim_editor import PresentationSectionType
from powermanim.layouts.arrangedbullets import Bullet, MathBullet
from powermanim.templates.bulletlist import BulletList
from torchvision.datasets import CIFAR100
from torchvision.transforms import transforms

from nn_core.common import PROJECT_ROOT

DISABLED_OPACITY = 0.4
FONT_SIZE = 28
SCALE_ACTIVE = 1.25
DATASET1 = CIFAR100(root=PROJECT_ROOT / "data", train=True, download=True)
# same as Dataset1 but with grayscale conversion
DATASET2 = CIFAR100(root=PROJECT_ROOT / "data", train=True, download=True, transform=transforms.Grayscale())

ANCHORS_COLOR = RED
ANCHORS_POINT_COLORS = [RED_D, GRAY_C, YELLOW_D]
SAMPLE_COLOR = GREEN
SIM_COLOR = BLUE


class Algorithm(Scene):
    def construct(self):
        slide_title = Tex("Algorithm").to_edge(UP)

        self.next_section("Background algo", type=PresentationSectionType.NORMAL, skip_animations=False)
        self.play(Create(slide_title))

        algo = (
            Bullet(
                r"Given a subset $\mathbb{A_\mathbb{X}}$ of the training set $\mathbb{X}$",
                font_size=FONT_SIZE,
                group=0,
            ),
            Bullet(
                r"a subset $\mathbb{A_\mathbb{Y}}$ of the training set $\mathbb{Y}$",
                font_size=FONT_SIZE,
                group=1,
            ),
            Bullet(
                r"and a \textbf{correspondence} between them",
                font_size=FONT_SIZE,
                group=2,
            ),
            Bullet(
                r"Apply the respective encoding functions $E_x$ and $E_y$",
                font_size=FONT_SIZE,
                group=3,
            ),
            Bullet(
                r"Normalize the encodings",
                font_size=FONT_SIZE,
                group=4,
            ),
            Bullet(
                r"The \textbf{latent translation} operator $\mathcal{T}$ ",
                " is obtained via solving:",
                font_size=FONT_SIZE,
                group=5,
            ),
            MathBullet(
                r"\mathcal{T}(\mathbf{x}) = \mathbf{R} \mathbb{A_\mathbb{X}} + \mathbf{b}",
                font_size=FONT_SIZE + 5,
                symbol=None,
                level=1,
                group=5,
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

        samples = [42, 7, 33]
        anchor_images1 = (
            Group(*[ImageMobject(DATASET1[sample_idx][0], image_mode="RGB") for sample_idx in samples])
            .scale(3.5)
            .arrange(RIGHT, buff=MED_LARGE_BUFF * 2)
            .to_edge(RIGHT)
            .shift(UP)
        )
        anchor_images2 = (
            Group(*[ImageMobject(DATASET2[sample_idx][0], image_mode="RGB") for sample_idx in samples])
            .scale(3.5)
            .arrange(RIGHT, buff=MED_LARGE_BUFF * 2)
            .to_edge(RIGHT)
            .next_to(anchor_images1, DOWN, buff=MED_LARGE_BUFF * 2)
        )

        correspondence = []
        for anchor_image1, anchor_image2 in zip(anchor_images1.submobjects, anchor_images2.submobjects):
            correspondence.append(
                Line(
                    anchor_image1.get_bottom() + DOWN * 0.25,
                    anchor_image2.get_top() + UP * 0.25,
                    color=TEAL_D,
                    stroke_width=5,
                ),
            )

        semantic_alignment = SurroundingRectangle(
            Group(anchor_images1, anchor_images2, *correspondence), color=TEAL_D, stroke_width=5
        )
        semantic_alignment_label = Tex(r"\textbf{Semantic Alignment!}", font_size=38).next_to(semantic_alignment, UP)
        semantic_alignment = Group(semantic_alignment, semantic_alignment_label)

        self.next_section("X anchors", type=PresentationSectionType.NORMAL)
        # anchor1
        self.play(
            AnimationGroup(
                bulletlist.only_next(),
                run_time=1,
            ),
            FadeIn(anchor_images1),
        )

        self.next_section("Y anchors", type=PresentationSectionType.NORMAL)
        # anchor2
        self.play(
            AnimationGroup(
                bulletlist.only_next(),
                run_time=1,
            ),
            FadeIn(anchor_images2),
        )

        self.next_section("Correspondence", type=PresentationSectionType.NORMAL)
        # correspondence
        self.play(
            AnimationGroup(
                bulletlist.only_next(),
                AnimationGroup(*[FadeIn(c) for c in correspondence], lag_ratio=0.5),
                run_time=1.5,
            )
        )
        self.play(FadeIn(semantic_alignment))

        anims = []
        polygons = []
        regularized_anims = []

        np.random.seed(42)
        min_displacement = 0.05
        displacement_factor: float = 0.3
        for image, color in zip(anchor_images1.submobjects, ANCHORS_POINT_COLORS):
            polygon = Polygon(
                image.get_center() + UL * np.random.uniform(min_displacement, image.width * displacement_factor),
                image.get_center() + UR * np.random.uniform(min_displacement, image.width * displacement_factor),
                image.get_center() + DR * np.random.uniform(min_displacement, image.width * displacement_factor),
                image.get_center() + DL * np.random.uniform(min_displacement, image.width * displacement_factor),
                color=color,
                fill_opacity=1,
                z_index=1,
            )
            anims.append(
                AnimationGroup(
                    FadeOut(image),
                    FadeIn(polygon),
                    lag_ratio=0.5,
                )
            )
            polygons.append(polygon)

            polygon.target = Polygon(
                image.get_center() + UL * image.width * displacement_factor,
                image.get_center() + UR * image.width * displacement_factor,
                image.get_center() + DR * image.width * displacement_factor,
                image.get_center() + DL * image.width * displacement_factor,
                color=color,
                fill_opacity=1,
                z_index=1,
            )
            regularized_anims.append(MoveToTarget(polygon))

        np.random.seed(42)
        for image, color in zip(anchor_images2.submobjects, ANCHORS_POINT_COLORS):
            polygon = Polygon(
                image.get_center() + UP * np.random.uniform(min_displacement, image.width * displacement_factor),
                image.get_center() + LEFT * np.random.uniform(min_displacement, image.width * displacement_factor),
                image.get_center() + RIGHT * np.random.uniform(min_displacement, image.width * displacement_factor),
                color=color,
                fill_opacity=1,
                z_index=1,
            )
            anims.append(
                AnimationGroup(
                    FadeOut(image),
                    FadeIn(polygon),
                    lag_ratio=0.5,
                )
            )
            polygons.append(polygon)

            polygon.target = Polygon(
                image.get_center() + UP * 2 * image.width * displacement_factor,
                image.get_center() + LEFT * image.width * displacement_factor,
                image.get_center() + RIGHT * image.width * displacement_factor,
                color=color,
                fill_opacity=1,
                z_index=1,
            )
            regularized_anims.append(MoveToTarget(polygon))

        self.next_section("Encoding", type=PresentationSectionType.NORMAL)
        # encoding
        self.play(
            AnimationGroup(
                AnimationGroup(
                    bulletlist.only_next(),
                    run_time=1,
                ),
                AnimationGroup(
                    AnimationGroup(
                        *anims[: len(anims) // 2],
                        lag_ratio=0.5,
                        run_time=1,
                    ),
                    AnimationGroup(
                        *anims[len(anims) // 2 :],
                        lag_ratio=0.5,
                        run_time=1,
                    ),
                ),
                lag_ratio=0.5,
            ),
        )

        self.next_section("Normalization", type=PresentationSectionType.NORMAL)
        # regularization
        self.play(
            AnimationGroup(
                bulletlist.only_next(),
                AnimationGroup(
                    AnimationGroup(
                        *regularized_anims[: len(regularized_anims) // 2],
                        lag_ratio=0.5,
                        run_time=1.5,
                    ),
                    AnimationGroup(
                        *regularized_anims[len(regularized_anims) // 2 :],
                        lag_ratio=0.5,
                        run_time=1.5,
                    ),
                ),
                lag_ratio=0.5,
            ),
        )

        self.next_section("Formula", type=PresentationSectionType.NORMAL)
        self.play(bulletlist.only_next())

        arrow = Arrow(0.2 * RIGHT, 0.2 * LEFT).next_to(bulletlist.rows[-1][-1], RIGHT, buff=MED_LARGE_BUFF)
        arrow_label = Tex(r"\textbf{Mostly orthogonal!}", font_size=38).next_to(arrow, RIGHT)

        self.next_section("Mostly orthogonal", type=PresentationSectionType.NORMAL)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    GrowArrow(arrow),
                    FadeIn(arrow_label),
                ),
                ShowPassingFlash(Underline(arrow_label, color=YELLOW)),
                lag_ratio=0.5,
            )
        )
