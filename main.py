import numpy as np
import pandas as pd


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