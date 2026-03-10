import numpy as np
from math_utils import softmax


def initialize_attention_weights(d_model: int, d_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inicializa as matrizes de pesos W_Q, W_K e W_V.

    Shapes:
    - W_Q: (d_model, d_k)
    - W_K: (d_model, d_k)
    - W_V: (d_model, d_k)
    """
    w_q = np.random.randn(d_model, d_k)
    w_k = np.random.randn(d_model, d_k)
    w_v = np.random.randn(d_model, d_k)
    return w_q, w_k, w_v


def compute_qkv(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula Q, K e V a partir da entrada X.

    Entrada:
    - x: (batch_size, sequence_length, d_model)

    Saída:
    - Q: (batch_size, sequence_length, d_k)
    - K: (batch_size, sequence_length, d_k)
    - V: (batch_size, sequence_length, d_k)
    """
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v
    return q, k, v


def scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula a atenção escalada.

    Retorna:
    - scores
    - attention_weights
    - output
    """
    d_k = q.shape[-1]

    # K transposto nos dois últimos eixos
    k_transposed = np.transpose(k, (0, 2, 1))

    # Scores: (batch_size, sequence_length, sequence_length)
    scores = q @ k_transposed

    # Escalonamento
    scaled_scores = scores / np.sqrt(d_k)

    # Softmax ao longo do último eixo
    attention_weights = softmax(scaled_scores, axis=-1)

    # Saída final
    output = attention_weights @ v

    return scaled_scores, attention_weights, output


def self_attention(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray
) -> tuple[np.ndarray, dict]:
    """
    Executa todo o bloco de self-attention.

    Retorna:
    - output final da atenção
    - dicionário com intermediários para debug
    """
    q, k, v = compute_qkv(x, w_q, w_k, w_v)
    scaled_scores, attention_weights, output = scaled_dot_product_attention(q, k, v)

    debug_info = {
        "q": q,
        "k": k,
        "v": v,
        "scaled_scores": scaled_scores,
        "attention_weights": attention_weights
    }

    return output, debug_info
