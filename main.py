import numpy as np
import pandas as pd
from attention import initialize_attention_weights, self_attention
from math_utils import relu, softmax, layer_norm
from feed_forward import initialize_ffn_weights, feed_forward


def tokenize(sentence: str) -> list[str]:
    """
    Divide a frase em tokens usando espaço.
    """
    return sentence.lower().split()


def build_vocab(tokens: list[str]) -> tuple[dict[str, int], dict[int, str], pd.DataFrame]:
    """
    Cria o vocabulário a partir da lista de tokens.

    Retorna:
    - token_to_id: dicionário palavra -> id
    - id_to_token: dicionário id -> palavra
    - vocab_df: DataFrame com o vocabulário
    """
    unique_tokens = sorted(set(tokens))

    token_to_id = {token: idx for idx, token in enumerate(unique_tokens)}
    id_to_token = {idx: token for token, idx in token_to_id.items()}

    vocab_df = pd.DataFrame({
        "token": list(token_to_id.keys()),
        "id": list(token_to_id.values())
    })

    return token_to_id, id_to_token, vocab_df


def encode_tokens(tokens: list[str], token_to_id: dict[str, int]) -> list[int]:
    """
    Converte lista de tokens em lista de ids.
    """
    return [token_to_id[token] for token in tokens]


def create_embedding_matrix(vocab_size: int, d_model: int) -> np.ndarray:
    """
    Cria a matriz de embeddings aleatória.

    Shape:
    (vocab_size, d_model)
    """
    return np.random.randn(vocab_size, d_model)


def tokens_to_embeddings(token_ids: list[int], embedding_matrix: np.ndarray) -> np.ndarray:
    """
    Busca os embeddings correspondentes aos token_ids.

    Entrada:
    - token_ids: lista de ids

    Saída:
    - embeddings com shape (sequence_length, d_model)
    """
    return embedding_matrix[token_ids]


def create_input_tensor(embeddings: np.ndarray) -> np.ndarray:
    """
    Adiciona a dimensão de batch.

    Entrada:
    - embeddings: (sequence_length, d_model)

    Saída:
    - X: (batch_size=1, sequence_length, d_model)
    """
    return np.expand_dims(embeddings, axis=0)


def main():
    sentence = "os pinguins não tem joelhos"
    d_model = 64

    print("=== ETAPA 1: PREPARAÇÃO DOS DADOS ===")

    # 1. Tokenização
    tokens = tokenize(sentence)
    print("Tokens:", tokens)

    # 2. Vocabulário
    token_to_id, id_to_token, vocab_df = build_vocab(tokens)
    print("\nVocabulário:")
    print(vocab_df)

    # 3. Converter tokens em ids
    token_ids = encode_tokens(tokens, token_to_id)
    print("\nToken IDs:", token_ids)

    # 4. Criar matriz de embeddings
    vocab_size = len(token_to_id)
    embedding_matrix = create_embedding_matrix(vocab_size, d_model)
    print("\nShape da matriz de embeddings:", embedding_matrix.shape)

    # 5. Buscar embeddings da frase
    embeddings = tokens_to_embeddings(token_ids, embedding_matrix)
    print("Shape dos embeddings da frase:", embeddings.shape)

    # 6. Criar tensor de entrada X
    X = create_input_tensor(embeddings)
    print("Shape final do tensor X:", X.shape)


    print("\n=== ETAPA 2: SELF-ATTENTION ===")

    d_k = d_model
    w_q, w_k, w_v = initialize_attention_weights(d_model, d_k)

    attention_output, debug_info = self_attention(X, w_q, w_k, w_v)

    # Residual connection após atenção
    x_res1 = X + attention_output

    # Layer normalization
    x_norm1 = layer_norm(x_res1)

    print("\n=== ETAPA 3: ADD & NORM APÓS SELF-ATTENTION ===")
    print("Shape de X original:", X.shape)
    print("Shape da saída da atenção:", attention_output.shape)
    print("Shape após residual:", x_res1.shape)
    print("Shape após layer norm:", x_norm1.shape)

    print("\nMédia por token após layer norm:")
    print(np.mean(x_norm1, axis=-1))

    print("\nVariância por token após layer norm:")
    print(np.var(x_norm1, axis=-1))

    print("Shape de Q:", debug_info["q"].shape)
    print("Shape de K:", debug_info["k"].shape)
    print("Shape de V:", debug_info["v"].shape)
    print("Shape dos scaled_scores:", debug_info["scaled_scores"].shape)
    print("Shape dos attention_weights:", debug_info["attention_weights"].shape)
    print("Shape da saída da atenção:", attention_output.shape)

    print("\nSoma das linhas dos attention_weights:")
    print(np.sum(debug_info["attention_weights"], axis=-1))

    print("\n=== ETAPA 4: FEED-FORWARD NETWORK ===")

    d_ff = 128
    w1, b1, w2, b2 = initialize_ffn_weights(d_model, d_ff)

    x_ffn, ffn_debug = feed_forward(x_norm1, w1, b1, w2, b2)

    print("Shape da entrada da FFN:", x_norm1.shape)
    print("Shape após primeira linear:", ffn_debug["hidden_linear"].shape)
    print("Shape após ReLU:", ffn_debug["hidden_relu"].shape)
    print("Shape da saída da FFN:", x_ffn.shape)


if __name__ == "__main__":
    main()
