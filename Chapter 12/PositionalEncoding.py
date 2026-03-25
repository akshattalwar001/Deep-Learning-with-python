import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, LayerNormalization, Dense
from tensorflow.keras.models import Model

VOCAB_SIZE = 20000
EMBED_DIM  = 256
MAX_LEN    = 50
NUM_HEADS  = 8
KEY_DIM    = 32


def positional_encoding(max_len, embed_dim):
    positions = np.arange(max_len)[:, np.newaxis]        # (max_len, 1)
    dims      = np.arange(embed_dim)[np.newaxis, :]      # (1, embed_dim)

    angles    = positions / np.power(10000, (2 * (dims // 2)) / embed_dim)

    angles[:, 0::2] = np.sin(angles[:, 0::2])            # even dims == sin
    angles[:, 1::2] = np.cos(angles[:, 1::2])            # odd  dims == cos

    return tf.cast(angles[np.newaxis, :, :], tf.float32) # (1, max_len, embed_dim)


pos_encoding = positional_encoding(MAX_LEN, EMBED_DIM)   # shape (1, 50, 256)


enc_in  = Input(shape=(MAX_LEN,))
dec_in  = Input(shape=(MAX_LEN,))

enc_emb = Embedding(VOCAB_SIZE, EMBED_DIM)(enc_in)
dec_emb = Embedding(VOCAB_SIZE, EMBED_DIM)(dec_in)

enc_emb = enc_emb + pos_encoding
dec_emb = dec_emb + pos_encoding

enc_out = MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM)(
              query=enc_emb, key=enc_emb, value=enc_emb)
enc_out = LayerNormalization()(enc_out + enc_emb)

dec_out = MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM)(
              query=dec_emb, key=dec_emb, value=dec_emb)
dec_out = LayerNormalization()(dec_out + dec_emb)

cross   = MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM)(
              query=dec_out, key=enc_out, value=enc_out)
cross   = LayerNormalization()(cross + dec_out)

out     = Dense(VOCAB_SIZE, activation='softmax')(cross)

model   = Model([enc_in, dec_in], out)
model.summary()
