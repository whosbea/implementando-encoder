import numpy as np

from attention import initialize_attention_weights, self_attention
from feed_forward import initialize_ffn_weights, feed_forward
from math_utils import layer_norm


def initialize_encoder_layer_weights(d_model: int, d_k: int, d_ff: int) -> dict:
    """
    Inicializa todos os pesos necessários para uma camada do encoder.

    Retorna um dicionário com:
    - pesos da self-attention
    - pesos da feed-forward network
    """
    w_q, w_k, w_v = initialize_attention_weights(d_model, d_k)
    w1, b1, w2, b2 = initialize_ffn_weights(d_model, d_ff)

    return {
        "w_q": w_q,
        "w_k": w_k,
        "w_v": w_v,
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
    }


def encoder_layer(x: np.ndarray, layer_weights: dict) -> tuple[np.ndarray, dict]:
    """
    Executa uma camada completa do encoder.

    Fluxo:
    1. Self-Attention
    2. Residual + LayerNorm
    3. Feed-Forward Network
    4. Residual + LayerNorm

    Entrada:
    - x: (batch_size, sequence_length, d_model)

    Saída:
    - x_out: (batch_size, sequence_length, d_model)
    - debug_info: informações intermediárias
    """
    # Self-attention
    attention_output, attention_debug = self_attention(
        x,
        layer_weights["w_q"],
        layer_weights["w_k"],
        layer_weights["w_v"],
    )

    # Primeiro Add & Norm
    x_res1 = x + attention_output
    x_norm1 = layer_norm(x_res1)

    # Feed-forward
    ffn_output, ffn_debug = feed_forward(
        x_norm1,
        layer_weights["w1"],
        layer_weights["b1"],
        layer_weights["w2"],
        layer_weights["b2"],
    )

    # Segundo Add & Norm
    x_res2 = x_norm1 + ffn_output
    x_out = layer_norm(x_res2)

    debug_info = {
        "attention_output": attention_output,
        "x_res1": x_res1,
        "x_norm1": x_norm1,
        "ffn_output": ffn_output,
        "x_res2": x_res2,
        "x_out": x_out,
        "attention_debug": attention_debug,
        "ffn_debug": ffn_debug,
    }

    return x_out, debug_info


def initialize_encoder_stack(n_layers: int, d_model: int, d_k: int, d_ff: int) -> list[dict]:
    """
    Inicializa os pesos de todas as camadas do encoder.
    """
    layers = []

    for _ in range(n_layers):
        layer_weights = initialize_encoder_layer_weights(d_model, d_k, d_ff)
        layers.append(layer_weights)

    return layers


def encoder_stack(x: np.ndarray, layers: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """
    Executa a pilha de camadas do encoder.

    Entrada:
    - x: (batch_size, sequence_length, d_model)

    Saída:
    - x: saída final após todas as camadas
    - all_debug_info: lista com debug de cada camada
    """
    all_debug_info = []

    for layer_index, layer_weights in enumerate(layers, start=1):
        x, debug_info = encoder_layer(x, layer_weights)
        debug_info["layer_index"] = layer_index
        all_debug_info.append(debug_info)

    return x, all_debug_info
