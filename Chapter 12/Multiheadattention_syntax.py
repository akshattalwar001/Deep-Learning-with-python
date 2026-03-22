import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, MultiHeadAttention, LayerNormalization
from tensorflow.keras.models import Model

VOCAB_SIZE = 20000
EMBED_DIM  = 256
MAX_LEN    = 50
NUM_HEADS  = 8
KEY_DIM    = 32  # EMBED_DIM // NUM_HEADS

# ── 1. LSTM ───────────────────────────────────────────────────────
enc_in     = Input(shape=(MAX_LEN,))
dec_in     = Input(shape=(MAX_LEN,))

enc_emb    = Embedding(VOCAB_SIZE, EMBED_DIM)(enc_in)
dec_emb    = Embedding(VOCAB_SIZE, EMBED_DIM)(dec_in)

_, h, c    = LSTM(512, return_state=True)(enc_emb)
dec_out    = LSTM(512, return_sequences=True)(dec_emb, initial_state=[h, c])
out        = Dense(VOCAB_SIZE, activation='softmax')(dec_out)

lstm_model = Model([enc_in, dec_in], out)


# ── 2. SELF ATTENTION ─────────────────────────────────────────────
enc_in     = Input(shape=(MAX_LEN,))
dec_in     = Input(shape=(MAX_LEN,))

enc_emb    = Embedding(VOCAB_SIZE, EMBED_DIM)(enc_in)
dec_emb    = Embedding(VOCAB_SIZE, EMBED_DIM)(dec_in)

enc_out    = tf.keras.layers.Attention()([enc_emb, enc_emb])  # self = same input twice
dec_out    = tf.keras.layers.Attention()([dec_emb, enc_out])  # cross = decoder looks at encoder
out        = Dense(VOCAB_SIZE, activation='softmax')(dec_out)

self_attn_model = Model([enc_in, dec_in], out)


# ── 3. MULTI HEAD ATTENTION ───────────────────────────────────────
enc_in     = Input(shape=(MAX_LEN,))
dec_in     = Input(shape=(MAX_LEN,))

enc_emb    = Embedding(VOCAB_SIZE, EMBED_DIM)(enc_in)
dec_emb    = Embedding(VOCAB_SIZE, EMBED_DIM)(dec_in)

enc_out    = MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM)(
                 query=enc_emb, key=enc_emb, value=enc_emb)
enc_out    = LayerNormalization()(enc_out + enc_emb)           # residual + norm

dec_out    = MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM)(
                 query=dec_emb, key=dec_emb, value=dec_emb)
dec_out    = LayerNormalization()(dec_out + dec_emb)           # residual + norm

cross      = MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM)(
                 query=dec_out, key=enc_out, value=enc_out)    # decoder attends to encoder
cross      = LayerNormalization()(cross + dec_out)             # residual + norm

out        = Dense(VOCAB_SIZE, activation='softmax')(cross)

mha_model  = Model([enc_in, dec_in], out)
