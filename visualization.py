import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def ensure_output_dir(output_dir: str = "outputs") -> None:
    """
    Garante que a pasta de saída exista.
    """
    os.makedirs(output_dir, exist_ok=True)


def draw_box(ax, x, y, w, h, text, fontsize=11):
    """
    Desenha uma caixa arredondada com texto centralizado.
    """
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        fill=False,
        linewidth=1.8
    )
    ax.add_patch(box)

    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True
    )


def draw_arrow(ax, x1, y1, x2, y2):
    """
    Desenha uma seta entre dois pontos.
    """
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", linewidth=1.8)
    )


def plot_encoder_pipeline(
    output_dir: str = "outputs",
    filename: str = "encoder_pipeline.png",
    show: bool = True
) -> None:
    """
    Plota o fluxo geral do Transformer Encoder.
    """
    ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(18, 4.8))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 5)
    ax.axis("off")

    y = 2.0
    w = 2.05
    h = 1.0

    boxes = [
        (0.6, y, "Frase de entrada"),
        (3.0, y, "Tokenização\n5 tokens"),
        (5.4, y, "Vocabulário + IDs\nvocab_size = 5"),
        (7.8, y, "Embeddings\nshape = (5, 64)"),
        (10.2, y, "Tensor X\nshape = (1, 5, 64)"),
        (12.6, y, "Encoder Layer"),
        (15.0, y, "Repetir 6x"),
    ]

    for x, y_box, text in boxes:
        draw_box(ax, x, y_box, w, h, text, fontsize=11)

    draw_box(
        ax,
        13.4,
        0.55,
        3.8,
        1.0,
        "Saída final do Encoder\nshape = (1, 5, 64)",
        fontsize=12
    )

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + w
        y1 = boxes[i][1] + h / 2
        x2 = boxes[i + 1][0]
        y2 = boxes[i + 1][1] + h / 2
        draw_arrow(ax, x1, y1, x2, y2)

    draw_arrow(ax, 16.0, 2.0, 15.3, 1.55)

    ax.text(
        1.625,
        1.55,
        '"os pinguins não tem joelhos"',
        ha="center",
        va="center",
        fontsize=10
    )

    ax.text(
        9,
        4.35,
        "Fluxo Geral do Transformer Encoder",
        ha="center",
        va="center",
        fontsize=16
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, filename),
        dpi=220,
        bbox_inches="tight"
    )

    if show:
        plt.show()

    plt.close()


def plot_encoder_layer_detail(
    output_dir: str = "outputs",
    filename: str = "encoder_layer_detail.png",
    show: bool = True
) -> None:
    """
    Plota o detalhamento de uma única camada do encoder.
    """
    ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(18, 5.2))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    y = 2.1
    w = 2.6
    h = 1.1

    steps = [
        (0.7, y, "Entrada X\n(1, 5, 64)"),
        (3.9, y, "Self-Attention\nQ, K, V\nScaled Dot-Product"),
        (7.1, y, "Add & Norm\nX + Attention"),
        (10.3, y, "Feed-Forward\n64 → 128 → 64"),
        (13.5, y, "Add & Norm\nX_norm1 + FFN"),
    ]

    for x, y_box, text in steps:
        draw_box(ax, x, y_box, w, h, text, fontsize=11)

    draw_box(
        ax,
        13.5,
        0.55,
        w,
        h,
        "Saída da camada\n(1, 5, 64)",
        fontsize=11
    )

    for i in range(len(steps) - 1):
        x1 = steps[i][0] + w
        y1 = steps[i][1] + h / 2
        x2 = steps[i + 1][0]
        y2 = steps[i + 1][1] + h / 2
        draw_arrow(ax, x1, y1, x2, y2)

    draw_arrow(ax, 14.8, 2.1, 14.8, 1.65)

    ax.text(
        9,
        4.8,
        "Detalhamento de uma Encoder Layer",
        ha="center",
        va="center",
        fontsize=16
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, filename),
        dpi=220,
        bbox_inches="tight"
    )

    if show:
        plt.show()

    plt.close()
