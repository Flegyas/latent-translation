import itertools
from typing import Tuple

import pandas as pd
from manim import *
from manim_editor import *
from pygments.styles.rainbow_dash import RED_DARK

from nn_core.common import PROJECT_ROOT

from latent_translation.manim.utils import value_to_color

domain2encoders = {
    "vision": [
        "vit_base_patch16_224",
        "rexnet_100",
        "vit_base_patch16_384",
        "vit_small_patch16_224",
        "vit_base_resnet50_384",
        "cspdarknet53",
        # "openai/clip-vit-base-patch32",
    ],
    "text": [
        "bert-base-cased",
        "bert-base-uncased",
        "google/electra-base-discriminator",
        "roberta-base",
        # "albert-base-v2",
        "xlm-roberta-base",
        # "openai/clip-vit-base-patch32",
    ],
}
encoder2name = {
    "google/electra-base-discriminator": "Electra",
    "roberta-base": "RoBERTa",
    "xlm-roberta-base": "XLM-R",
    "vit_base_patch16_224": "ViT-base-224",
    "vit_base_patch16_384": "ViT-base-384",
    "rexnet_100": "RexNet",
    "vit_small_patch16_224": "ViT-small",
    "vit_base_resnet50_384": "ViT-ResNet50",
    "cspdarknet53": "DarkNet",
    "openai/clip-vit-base-patch32-vision": "CLIP-vision",
    "openai/clip-vit-base-patch32-text": "CLIP-text",
    "bert-base-cased": "BERT-cased",
    "bert-base-uncased": "BERT-uncased",
    "albert-base-v2": "ALBERT",
}


def get_space_name(encoder_name: str, domain: str):
    if encoder_name.startswith("openai/clip-vit-base-patch32"):
        return encoder2name[f"openai/clip-vit-base-patch32-{domain}"]

    return encoder2name[encoder_name]


class Heatmap(VGroup):
    @classmethod
    def build_heatmap(
        cls, data: np.ndarray, x_values, x_colors, y_values, y_colors, x_label: str, y_label: str, z_label: str, cmap
    ):
        max_val = data.max()
        min_val = data.min()

        cells = []
        first_col = []
        last_row = []
        for i_row, j_col in itertools.product(range(data.shape[0]), range(data.shape[1])):
            value = data[i_row, j_col]
            color = value_to_color(value, min_val, max_val, cmap=cmap)
            cell = Square(side_length=1, stroke_width=0).set_fill(color, opacity=0.8)
            label = MathTex(f"{value:.2f}").scale_to_fit_width(cell.width * 0.5).move_to(cell.get_center())
            cell = VGroup(cell, label)
            if j_col == 0:
                first_col.append(cell)
            if i_row == data.shape[0] - 1:
                last_row.append(cell)
            cells.append(cell)

        heatmap = VGroup(*cells).arrange_in_grid(data.shape[0], data.shape[1], buff=0)
        heatmap_border = SurroundingRectangle(heatmap, buff=0, color=WHITE, stroke_width=2)

        x_values = VGroup(
            *[
                Tex(tick, color=color)
                .scale_to_fit_height(cell.width * 0.35)
                .rotate(PI / 4)
                .next_to(cell.get_bottom(), DL)
                for tick, color, cell in zip(x_values, x_colors, last_row)
            ]
        )
        x_label = Tex(x_label).next_to(x_values, DOWN, buff=LARGE_BUFF)

        y_values = VGroup(
            *[
                Tex(tick, color=color).scale_to_fit_height(cell.height * 0.35).next_to(cell.get_left(), LEFT)
                for tick, color, cell in zip(y_values, y_colors, first_col)
            ]
        )

        y_label = Tex(y_label).rotate(PI / 2).next_to(y_values, LEFT, buff=LARGE_BUFF)

        colorbar = VGroup(
            *[
                Square(side_length=1, stroke_width=0).set_fill(
                    value_to_color(value, min_val, max_val, cmap=cmap), opacity=0.8
                )
                for value in reversed(np.linspace(min_val, max_val, 10))
            ]
        ).arrange(DOWN, buff=0)

        colorbar_labels = VGroup(
            *[
                MathTex(f"{value:.2f}").move_to(colorbar.get_center())
                for value in reversed(np.linspace(min_val, max_val, 10))
            ]
        )
        for colorbar_item, colorbar_label in zip(colorbar, colorbar_labels):
            colorbar_label.scale_to_fit_height(colorbar_item.height * 0.35).next_to(
                colorbar_item, RIGHT, buff=MED_SMALL_BUFF
            )

        colorbar_name = Tex(
            z_label,
        ).next_to(colorbar, UP, buff=MED_LARGE_BUFF)

        colorbar = VGroup(SurroundingRectangle(colorbar, buff=0, color=WHITE, stroke_width=2), colorbar)
        colorbar = VGroup(colorbar, colorbar_labels, colorbar_name)

        colorbar.scale_to_fit_height(heatmap.height * 0.75).next_to(heatmap, RIGHT, buff=LARGE_BUFF)

        return VGroup(heatmap, colorbar, VGroup(*x_values), VGroup(*y_values), heatmap_border, x_label, y_label)

    def __init__(
        self,
        data: np.ndarray,
        x_values,
        x_colors,
        y_values,
        y_colors,
        x_label: str,
        y_label: str,
        z_label: str,
        cmap="Blues",
    ):
        self.n_rows, self.n_cols = data.shape
        (
            self.matrix,
            self.colorbar,
            self.x_ticks,
            self.y_ticks,
            self.heatmap_border,
            self.x_label,
            self.y_label,
        ) = Heatmap.build_heatmap(
            data,
            cmap=cmap,
            x_values=x_values,
            x_colors=x_colors,
            y_values=y_values,
            y_colors=y_colors,
            x_label=x_label,
            y_label=y_label,
            z_label=z_label,
        )

        super().__init__(
            self.matrix, self.colorbar, self.x_ticks, self.y_ticks, self.heatmap_border, self.x_label, self.y_label
        )

    def __getitem__(self, row_col: Tuple[int, int]):
        row, col = row_col
        return self.matrix[row * self.n_cols + col]

    def highlight(self, start_row: int, end_row: int, start_col: int, end_col: int, color=RED) -> VMobject:
        assert start_row < end_row
        assert start_col < end_col
        assert start_row >= 0 and end_row < self.n_rows
        assert start_col >= 0 and end_col < self.n_cols

        matching_cells = []
        for i_row, j_col in itertools.product(range(start_row, end_row + 1), range(start_col, end_col + 1)):
            matching_cells.append(self[i_row, j_col])

        matching_cells = VGroup(*matching_cells)
        border = SurroundingRectangle(matching_cells, buff=0, color=color, stroke_width=2)

        return border


class Crossmodal(Scene):
    def construct(self):
        df = pd.read_csv(PROJECT_ROOT / "data" / "multimodal_stitching.tsv", sep="\t")
        # map encoder names to human-readable names
        df["encoding_space"] = df["encoding_space"].apply(lambda x: encoder2name[x])
        df["decoding_space"] = df["decoding_space"].apply(lambda x: encoder2name[x])

        x_values, x_colors = list(
            zip(
                *(
                    list(map(lambda x: (get_space_name(x, domain="vision"), TEAL_E), domain2encoders["vision"]))
                    + list(map(lambda x: (get_space_name(x, domain="text"), YELLOW_E), domain2encoders["text"]))
                )
            )
        )
        y_values, y_colors = list(x_values), list(x_colors)

        self.play(Write(Tex("Cross-modal Stitching").scale(1).to_edge(UP)), duration=0.5)

        data: np.ndarray = self.to_matrix(df, x_values, y_values)
        heatmap = (
            Heatmap(
                data=data,
                x_values=x_values,
                x_colors=x_colors,
                y_values=y_values,
                y_colors=y_colors,
                x_label=r"\textbf{Encoder}",
                y_label=r"\textbf{Decoder}",
                z_label=r"\textbf{Score}",
                cmap="Blues",
            )
            .center()
            .scale(0.35)
            .to_edge(DOWN)
        )
        self.play(FadeIn(heatmap))

        # language encodings are better
        highlight = heatmap.highlight(start_row=6, end_row=10, start_col=6, end_col=10, color=RED_DARK)
        self.play(Create(highlight))
        self.wait()
        self.play(Uncreate(highlight))

        # better encoder -> better performances than training modality
        highlight = heatmap.highlight(start_row=0, end_row=5, start_col=9, end_col=10, color=RED_DARK)
        self.play(Create(highlight))
        self.wait()
        self.play(Uncreate(highlight))

    def to_matrix(self, df, x_values, y_values):
        z_key: str = "score"
        z = np.zeros((len(x_values), len(y_values)))
        for (i_x, x), (j_y, y) in itertools.product(enumerate(x_values), enumerate(y_values)):
            z[i_x, j_y] = df[(df["encoding_space"] == x) & (df["decoding_space"] == y)][z_key].mean()

        return z.T
