import numpy as np

# Reprodutibilidade
SEED = 42
np.random.seed(SEED)

# Hiperparâmetros do modelo
D_MODEL = 64        # dimensão dos embeddings
D_FF = 128          # dimensão interna do feed-forward
N_LAYERS = 6        # número de camadas do encoder
EPSILON = 1e-6      # usado na layer normalization

# Frase de teste inicial
SENTENCE = "os pinguins não tem joelhos"
