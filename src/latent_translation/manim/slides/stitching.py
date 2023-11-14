from lightning import seed_everything
from manim import *
from manim.mobject.geometry.line import Arrow
from manim.mobject.geometry.polygram import Polygon, Rectangle
from manim_editor import PresentationSectionType
from powermanim.components.chartbars import ChartBars
from scipy.special import softmax

from nn_core.common import PROJECT_ROOT

START_MODULE_OPACITY: float = 0.5

OBJ_OPACITY = 0.5

N_BARS = 5

seed_everything(0)


def build_stitching_ae(model_id: str, encoder_color: str, decoder_color: str, label_pos):
    scale: float = 0.6

    rr_color = GREEN_D
    rr_size = 1.25
    encoder = (
        Polygon(
            2 * UP + LEFT,
            RIGHT + UP * rr_size,
            RIGHT + DOWN * rr_size,
            2 * DOWN + LEFT,
            fill_opacity=0.6,
            color=encoder_color,
        )
        # .set_opacity(START_MODULE_OPACITY)
        .scale(scale)
    )
    rr_block = Rectangle(height=rr_size * 2, width=1, fill_opacity=0.05).scale(scale).next_to(encoder, RIGHT)

    rr_label = Tex("RR", z_index=2).scale(scale).move_to(rr_block.get_critical_point(ORIGIN))
    rr_block = VGroup(rr_block, rr_label)

    encoder = VGroup(encoder, rr_block)

    decoder = (
        Polygon(RIGHT + 2 * UP, RIGHT + 2 * DOWN, DOWN + LEFT, UP + LEFT, fill_opacity=0.6, color=decoder_color)
        # .set_opacity(START_MODULE_OPACITY)
        .scale(scale)
    )

    latent = Rectangle(height=2 * 0.6, width=0.5, fill_opacity=0.6, color=PURPLE)

    encoded_arrow = Arrow(
        LEFT / 2,
        RIGHT / 2,
    )
    decoding_arrow = Arrow(
        LEFT / 2,
        RIGHT / 2,
    )

    encoded_arrow.next_to(latent, LEFT)
    decoding_arrow.next_to(latent, RIGHT)

    encoder.next_to(encoded_arrow, LEFT)
    decoder.next_to(decoding_arrow, RIGHT)

    encoder_label = Tex(f"Encoder {model_id}", z_index=2).next_to(encoder, label_pos)
    # .scale_to_fit_width(encoder.height * 2 / 3)
    # encoder_label.move_to(encoder.get_center()).rotate(PI / 2)

    decoder_label = Tex(f"Decoder {model_id}", z_index=2).next_to(decoder, label_pos)
    # .scale_to_fit_width(decoder.height * 2 / 3)
    # decoder_label.move_to(decoder.get_center()).rotate(PI / 2)

    return VDict(
        dict(
            encoder=encoder,
            decoder=decoder,
            encoder_label=encoder_label,
            encoded_arrow=encoded_arrow,
            decoding_arrow=decoding_arrow,
            decoder_label=decoder_label,
        )
    )


def fadein_and_move(mob: VMobject, alpha: float, t=0.5):
    mob.restore()
    if alpha <= t:
        opacity = interpolate(0, OBJ_OPACITY, alpha * 2)
        # fill_opacity = interpolate(0, OBJ_OPACITY, alpha * 2)
        mob.set_opacity(opacity)
        # mob.set_style(fill_opacity=fill_opacity)

    else:
        shift_x = interpolate(mob.get_center()[0], mob.target.get_center()[0], (alpha - 0.5) * 2)
        shift_y = interpolate(mob.get_center()[1], mob.target.get_center()[1], (alpha - 0.5) * 2)

        mob.move_to([shift_x, shift_y, 0])


def autoencode_anim(
    encoder, decoder, image_in, image_out, encoded_arrow: VMobject, decoding_arrow: VMobject, latent_color=PURPLE
):
    # Encoding
    encoding_arrow = Arrow(
        LEFT / 2,
        RIGHT / 2,
    )
    decoded_arrow = Arrow(
        LEFT / 2,
        RIGHT / 2,
    )
    image_in.rescale_to_fit(length=encoder.height, dim=1)
    image_out.rescale_to_fit(length=decoder.height, dim=1)

    encoding_arrow.next_to(encoder.get_left(), LEFT)
    decoded_arrow.next_to(decoder.get_right(), RIGHT)
    image_in.next_to(encoding_arrow.get_left(), LEFT)
    image_out.next_to(decoded_arrow.get_right(), RIGHT)

    latent = Rectangle(height=2 * 0.6, width=0.5, fill_opacity=0.6, color=latent_color)
    latent.next_to(encoded_arrow, RIGHT)
    latent.save_state()

    latent.generate_target()
    latent.target.next_to(decoding_arrow.get_left(), LEFT)

    forward_anims = [
        AnimationGroup(
            *[
                FadeIn(image_in, shift=RIGHT),
                GrowArrow(encoding_arrow, shift=RIGHT),
            ],
            # rate_func=rate_functions.smooth,
            lag_ratio=0.5,
            run_time=1,
        ),
        AnimationGroup(
            *[
                Flash(encoder, flash_radius=encoder.height / 2, color=encoder.get_color()),
                GrowArrow(encoded_arrow, shift=RIGHT),
                UpdateFromAlphaFunc(mobject=latent, update_function=fadein_and_move),
            ],
            # rate_func=rate_functions.smooth,
            lag_ratio=0.5,
            run_time=1,
        ),
        AnimationGroup(
            *[
                GrowArrow(decoding_arrow, shift=RIGHT),
                Flash(decoder, flash_radius=decoder.height / 2, color=decoder.get_color()),
                GrowArrow(decoded_arrow, shift=RIGHT),
                FadeIn(image_out, shift=RIGHT),
            ],
            # rate_func=rate_functions.smooth,
            lag_ratio=0.5,
            run_time=1,
        ),
    ]

    end_anim = AnimationGroup(
        *[
            FadeOut(image_out),
            FadeOut(decoded_arrow),
            FadeOut(decoding_arrow),
            FadeOut(latent),
            FadeOut(encoded_arrow),
            FadeOut(encoding_arrow),
            FadeOut(image_in),
        ],
        rate_func=rate_functions.smooth,
        lag_ratio=0,
        run_time=1,
    )

    return forward_anims, end_anim, latent


def autoencode_anim_big_stiching(
    encoder,
    decoder,
    image_in,
    image_out,
    encoded_arrow: VMobject,
    decoding_arrow: VMobject,
    start_color,
    end_color,
):
    # Encoding
    encoding_arrow = Arrow(
        LEFT / 2,
        RIGHT / 2,
    )
    decoded_arrow = Arrow(
        LEFT / 2,
        RIGHT / 2,
    )
    image_in.rescale_to_fit(length=encoder.height, dim=1)
    image_out.rescale_to_fit(length=decoder.height, dim=1)

    encoding_arrow.next_to(encoder.get_left(), LEFT)
    decoded_arrow.next_to(decoder.get_right(), RIGHT)
    image_in.next_to(encoding_arrow.get_left(), LEFT)
    image_out.next_to(decoded_arrow.get_right(), RIGHT)

    latent = Rectangle(height=2 * 0.6, width=0.5, fill_opacity=0.6, color=start_color)
    latent.move_to(
        (
            ((encoded_arrow.get_center() + decoding_arrow.get_center()) / 2)[0],
            encoded_arrow.get_center()[1],
            0,
        )
    )
    latent.save_state()

    latent.generate_target()
    latent.target.next_to(decoding_arrow.get_left(), LEFT)

    first_half_start = [
        FadeIn(image_in, shift=RIGHT),
        GrowArrow(encoding_arrow, shift=RIGHT),
        Flash(encoder, flash_radius=encoder.height / 2, color=encoder.get_color()),
        GrowArrow(encoded_arrow, shift=RIGHT),
        FadeIn(latent),
    ]

    latent.generate_target()
    latent.target.match_y(decoding_arrow)
    latent.target.set_color(end_color)
    move_latent_start = [Transform(latent, latent.target)]

    second_half_start = [
        GrowArrow(decoding_arrow, shift=RIGHT),
        Flash(decoder, flash_radius=decoder.height / 2, color=decoder.get_color()),
        GrowArrow(decoded_arrow, shift=RIGHT),
        FadeIn(image_out, shift=RIGHT),
    ]

    end_anim = AnimationGroup(
        *[
            FadeOut(image_out, shift=LEFT),
            FadeOut(decoded_arrow),
            FadeOut(decoding_arrow),
            FadeOut(latent),
            FadeOut(encoded_arrow),
            FadeOut(encoding_arrow),
            FadeOut(image_in, shift=UP),
        ],
        rate_func=rate_functions.smooth,
        lag_ratio=0.75,
        run_time=1.5,
    )

    return first_half_start, move_latent_start, second_half_start, end_anim


model1_color = RED
model2_color = BLUE
rel_color = PURPLE


class Stitching(Scene):
    def construct(self):
        self.next_section("Translation start", type=PresentationSectionType.NORMAL)
        slide_title = Tex("Relative Training").to_edge(UP)

        ae1 = build_stitching_ae(
            model_id="1", encoder_color=model1_color, decoder_color=model1_color, label_pos=UP
        ).scale(0.85)
        ae2 = build_stitching_ae(
            model_id="2", encoder_color=model2_color, decoder_color=model2_color, label_pos=DOWN
        ).scale(0.85)
        VGroup(ae1, ae2).arrange(buff=1, direction=DOWN)
        VGroup(ae1, ae2).shift(DOWN * 0.5)

        self.play(
            AnimationGroup(Create(slide_title), run_time=0.5),
            AnimationGroup(ShowPassingFlash(Underline(slide_title, color=YELLOW)), run_time=2.5),
            *{Create(obj) for obj_name, obj in ae1.submob_dict.items() if "arrow" not in obj_name},
            *{Create(obj) for obj_name, obj in ae2.submob_dict.items() if "arrow" not in obj_name},
        )

        # Standard AE mechanism, two different models
        image_indices1 = [0, 1, 2][:1]
        image_indices2 = [0, 4, 5][:1]
        for i, (sample1, sample2) in enumerate(zip(image_indices1, image_indices2)):
            image1_in = ImageMobject(PROJECT_ROOT / "data" / "assets" / "pebble")
            dist1 = softmax(np.random.randn(N_BARS) * 1.15)

            image1_out = ChartBars(
                Axes(x_range=[0, 6], y_range=[0, 1.5], x_length=0.5, y_length=2),
                dist1,
                xs=list(range(N_BARS)),
                fill_color=RED,
                stroke_width=1,
            )

            image2_in = Tex("Pebble").scale_to_fit_height(image1_in.get_width())
            image2_out = ChartBars(
                Axes(x_range=[0, 6], y_range=[0, 1.5], x_length=0.5, y_length=2),
                dist1,
                xs=list(range(N_BARS)),
                fill_color=RED,
                stroke_width=1,
            )

            forward_anims1, image1_end_anim, latent1 = autoencode_anim(
                encoder=ae1["encoder"],
                decoder=ae1["decoder"],
                encoded_arrow=ae1["encoded_arrow"],
                decoding_arrow=ae1["decoding_arrow"],
                image_in=image1_in,
                image_out=image1_out,
            )
            forward_anims2, image2_end_anim, latent2 = autoencode_anim(
                encoder=ae2["encoder"],
                decoder=ae2["decoder"],
                encoded_arrow=ae2["encoded_arrow"],
                decoding_arrow=ae2["decoding_arrow"],
                image_in=image2_in,
                image_out=image2_out,
            )

            latent_equal = Tex("=", z_index=2).scale(2).rotate(PI / 2)
            decoder_equal = Tex("=", z_index=2).scale(2).rotate(PI / 2)

            latent_equal.add_updater(lambda m: m.move_to((latent1.get_center() + latent2.get_center()) / 2))
            decoder_equal.add_updater(
                lambda m: m.move_to((ae1["decoder"].get_center() + ae2["decoder"].get_center()) / 2)
            )

            self.wait(0.1)
            self.next_section("Forward anim", type=PresentationSectionType.NORMAL)
            self.play(forward_anims1[0], forward_anims2[0])

            self.wait(0.1)
            self.next_section("Forward anim", type=PresentationSectionType.NORMAL)
            self.play(forward_anims1[1], forward_anims2[1])
            self.play(
                Create(latent_equal),
            )

            self.wait(0.1)
            self.next_section("Forward anim", type=PresentationSectionType.NORMAL)
            self.play(forward_anims1[2], forward_anims2[2])
            self.play(Create(decoder_equal))

            self.wait(0.1)
            self.next_section("Forward anim", type=PresentationSectionType.NORMAL)

            for i in range(2):
                self.play(
                    AnimationGroup(
                        AnimationGroup(
                            MoveAlongPath(
                                ae1["decoder"],
                                ArcBetweenPoints(ae1["decoder"].get_center(), ae2["decoder"].get_center()),
                            ),
                            MoveAlongPath(
                                ae2["decoder"],
                                ArcBetweenPoints(ae2["decoder"].get_center(), ae1["decoder"].get_center()),
                            ),
                        ),
                    )
                )

        latent_disegual = (
            MathTex("\\neq", z_index=2).scale(2).rotate(PI / 2).move_to(latent_equal.get_critical_point(ORIGIN))
        )
        decoder_disegual = (
            MathTex("\\neq", z_index=2).scale(2).rotate(PI / 2).move_to(decoder_equal.get_critical_point(ORIGIN))
        )
        cross1 = Cross(ae1["encoder"][1]).scale(1.15)
        cross2 = Cross(ae2["encoder"][1]).scale(1.15)

        self.wait(0.1)
        self.next_section("Translation", type=PresentationSectionType.NORMAL)
        slide_title2 = Tex("Absolute Training").to_edge(UP)
        self.play(
            Transform(slide_title, slide_title2),
            Create(cross1),
            Create(cross2),
            AnimationGroup(ShowPassingFlash(Underline(slide_title2, color=YELLOW)), run_time=2.5),
        )

        self.wait(0.1)
        self.next_section("Translation", type=PresentationSectionType.NORMAL)
        self.play(
            Transform(latent_equal, latent_disegual),
        )

        self.wait(0.1)
        self.next_section("Translation", type=PresentationSectionType.NORMAL)
        self.play(
            Transform(decoder_equal, decoder_disegual),
        )

        self.wait(0.1)
        self.next_section("First stitching", type=PresentationSectionType.NORMAL)
        self.play(
            image1_end_anim,
            image2_end_anim,
            FadeOut(latent_equal),
            FadeOut(decoder_equal),
        )

        # Standard AE mechanism, two different models
        image_indices1 = [2, 2][:1]
        image_indices2 = [2, 5][:1]
        for i, (sample1, sample2) in enumerate(zip(image_indices1, image_indices2)):
            image1_in = ImageMobject(PROJECT_ROOT / "data" / "assets" / "ananas.png")

            dist1 = softmax(np.random.randn(N_BARS) * 1.15)
            image1_out = ChartBars(
                Axes(x_range=[0, 6], y_range=[0, 1.5], x_length=0.5, y_length=2),
                dist1,
                xs=list(range(N_BARS)),
                fill_color=RED,
                stroke_width=1,
            )

            image2_in = Tex("Ananas").scale_to_fit_width(image1_in.get_width())
            image2_out = ChartBars(
                Axes(x_range=[0, 6], y_range=[0, 1.5], x_length=0.5, y_length=2),
                dist1,
                xs=list(range(N_BARS)),
                fill_color=RED,
                stroke_width=1,
            )

            forward_anims1, image1_end_anim, latent1 = autoencode_anim(
                encoder=ae1["encoder"],
                decoder=ae1["decoder"],
                encoded_arrow=ae1["encoded_arrow"],
                decoding_arrow=ae1["decoding_arrow"],
                image_in=image1_in,
                image_out=image1_out,
                latent_color=ORANGE,
            )
            forward_anims2, image2_end_anim, latent2 = autoencode_anim(
                encoder=ae2["encoder"],
                decoder=ae2["decoder"],
                encoded_arrow=ae2["encoded_arrow"],
                decoding_arrow=ae2["decoding_arrow"],
                image_in=image2_in,
                image_out=image2_out,
                latent_color=GREEN,
            )

            self.wait(0.1)
            self.next_section("Forward anim", type=PresentationSectionType.NORMAL)
            self.play(forward_anims1[0], forward_anims2[0])

            self.wait(0.1)
            self.next_section("Forward anim", type=PresentationSectionType.NORMAL)
            self.play(forward_anims1[1], forward_anims2[1])

            self.wait(0.1)
            self.next_section("Forward anim", type=PresentationSectionType.NORMAL)
            self.play(forward_anims1[2], forward_anims2[2])

            self.wait(0.1)
            self.next_section("Forward anim", type=PresentationSectionType.NORMAL)
            self.play(
                image1_end_anim,
                image2_end_anim,
            )

        # self.wait(0.1)
        self.next_section("Translation", type=PresentationSectionType.NORMAL)
        slide_title3 = Tex("Latent Translation").to_edge(UP)

        # Prepare for stitching
        t_symbol = Tex(r"$\mathcal{T}$", z_index=2)
        t_symbol.set_z_index(2)
        tblock = SurroundingRectangle(
            t_symbol,
            buff=0.1,
            color=WHITE,
            fill_color=BLACK,
            fill_opacity=1,
            z_index=1,
        )
        tblock.stretch(2, dim=0)
        tblock_a = VGroup(tblock, t_symbol)
        tblock_a.move_to(
            VGroup(
                ae1["encoded_arrow"],
                ae2["encoded_arrow"],
                ae1["decoding_arrow"],
                ae2["decoding_arrow"],
            ).get_center()
        )

        self.play(
            ae1["decoder"].animate.set_opacity(0.1),
            ae2["encoder"].animate.set_opacity(0.1),
            cross2.animate.set_opacity(0.1),
            Transform(slide_title, slide_title3),
            AnimationGroup(
                ShowPassingFlash(Underline(slide_title, color=YELLOW)),
                Create(tblock),
                Create(t_symbol),
                run_time=2.5,
            ),
        )

        # Stitching animation
        stitching_indices = [8]
        for sample in stitching_indices:
            image_in = ImageMobject(PROJECT_ROOT / "data" / "assets" / "snowflake")

            dist1 = softmax(np.random.randn(N_BARS) * 1.15)

            image_out = ChartBars(
                Axes(x_range=[0, 6], y_range=[0, 1.5], x_length=0.5, y_length=2),
                dist1,
                xs=list(range(N_BARS)),
                fill_color=RED,
                stroke_width=1,
            )

            first_half_start, move_latent_start, second_half_start, end_anim = autoencode_anim_big_stiching(
                encoder=ae1["encoder"],
                decoder=ae2["decoder"],
                encoded_arrow=ae1["encoded_arrow"],
                decoding_arrow=ae2["decoding_arrow"],
                image_in=image_in,
                image_out=image_out,
                start_color=ORANGE,
                end_color=GREEN,
            )

            self.wait(0.1)
            self.next_section("Embed", type=PresentationSectionType.NORMAL)
            self.play(
                AnimationGroup(
                    *first_half_start,
                    lag_ratio=0.5,
                    run_time=1,
                )
            )

            self.wait(0.1)
            self.next_section("Stitch", type=PresentationSectionType.NORMAL)
            self.play(*move_latent_start)

            self.wait(0.1)
            self.next_section("Decode", type=PresentationSectionType.NORMAL)
            self.play(
                AnimationGroup(
                    *second_half_start,
                    lag_ratio=0.5,
                    run_time=1,
                )
            )

        self.wait()
        self.next_section("3 stitched models", type=PresentationSectionType.NORMAL)

        self.play(
            end_anim,
            Uncreate(cross1),
            Uncreate(tblock),
            Uncreate(t_symbol),
            Uncreate(cross2),
            FadeOut(slide_title),
            *(Uncreate(ae1[x]) for x in ("encoder", "decoder", "encoder_label", "decoder_label")),
            *(Uncreate(ae2[x]) for x in ("encoder", "decoder", "encoder_label", "decoder_label")),
        )
