import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, LayerNormalization, Dense
from tensorflow.keras.models import Model

VOCAB_SIZE = 20000
EMBED_DIM  = 256
MAX_LEN    = 50
NUM_HEADS  = 8
KEY_DIM    = 32


def create_causal_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # 0 = allowed, 1 = blocked

print(create_causal_mask(4))

# [[0, 1, 1, 1],
#  [0, 0, 1, 1],
#  [0, 0, 0, 1],
#  [0, 0, 0, 0]]

enc_in  = Input(shape=(MAX_LEN,))
dec_in  = Input(shape=(MAX_LEN,))

enc_emb = Embedding(VOCAB_SIZE, EMBED_DIM)(enc_in)
dec_emb = Embedding(VOCAB_SIZE, EMBED_DIM)(dec_in)

enc_out = MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM)(
              query=enc_emb, key=enc_emb, value=enc_emb)
enc_out = LayerNormalization()(enc_out + enc_emb)

mask    = create_causal_mask(MAX_LEN)
dec_out = MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM)(
              query=dec_emb, key=dec_emb, value=dec_emb,
              attention_mask=mask)                        
dec_out = LayerNormalization()(dec_out + dec_emb)
cross   = MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM)(
              query=dec_out, key=enc_out, value=enc_out)
cross   = LayerNormalization()(cross + dec_out)

out     = Dense(VOCAB_SIZE, activation='softmax')(cross)

model   = Model([enc_in, dec_in], out)
