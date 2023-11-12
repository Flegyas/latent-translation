import itertools
from typing import Tuple

import pandas as pd
from manim import *
from manim_editor import *

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
        "albert-base-v2",
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


x_values = list(map(lambda x: get_space_name(x, domain="vision"), domain2encoders["vision"])) + list(
    map(lambda x: get_space_name(x, domain="text"), domain2encoders["text"])
)
y_values = x_values.copy()


class Heatmap(VGroup):
    @classmethod
    def build_heatmap(cls, data: np.ndarray, cmap):
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

        x_ticks = VGroup(
            *[
                Tex(tick).scale_to_fit_height(cell.width * 0.35).rotate(70).next_to(cell.get_bottom(), DOWN + LEFT / 2)
                for tick, cell in zip(x_values, last_row)
            ]
        )
        y_ticks = VGroup(
            *[
                Tex(tick).scale_to_fit_height(cell.height * 0.35).next_to(cell.get_left(), LEFT)
                for tick, cell in zip(y_values, first_col)
            ]
        )

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

        colorbar = VGroup(SurroundingRectangle(colorbar, buff=0, color=WHITE, stroke_width=2), colorbar)
        colorbar = VGroup(colorbar, colorbar_labels)

        colorbar.scale_to_fit_height(heatmap.height * 0.75).next_to(heatmap, RIGHT, buff=LARGE_BUFF)

        return VGroup(heatmap, colorbar, VGroup(*x_ticks), VGroup(*y_ticks), heatmap_border)

    def __init__(self, data: np.ndarray, cmap="Blues"):
        self.n_rows, self.n_cols = data.shape
        self.matrix, self.colorbar, self.x_ticks, self.y_ticks, self.heatmap_border = Heatmap.build_heatmap(
            data, cmap=cmap
        )

        super().__init__(self.matrix, self.colorbar, self.x_ticks, self.y_ticks)

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

        data: np.ndarray = self.to_matrix(df)

        heatmap = Heatmap(data=data, cmap="Blues").center().scale(0.5)
        self.add(heatmap)

        highlight = heatmap.highlight(start_row=0, end_row=5, start_col=0, end_col=5, color=RED)
        # self.play(Create(highlight))
        # self.wait()
        # self.play(Uncreate(highlight))

    def to_matrix(self, df):
        z_key: str = "score"
        z = np.zeros((len(y_values), len(x_values)))
        for (i_x, x), (j_y, y) in itertools.product(enumerate(x_values), enumerate(y_values)):
            z[i_x, j_y] = df[(df["encoding_space"] == x) & (df["decoding_space"] == y)][z_key].mean()

        return z
