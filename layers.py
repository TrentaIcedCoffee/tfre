import tensorflow as tf
import numpy as np


class PositionEmbedding(tf.keras.layers.Layer):

  def __init__(self, *, vocab_size, max_tokens, d_model):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
    self.pos_encoding = self.generate_position_encoding(length=max_tokens,
                                                        d_model=d_model)

  def generate_position_encoding(self, length, d_model):
    # https://arxiv.org/abs/1706.03762, Section 3.5.
    positions = np.arange(length)[..., np.newaxis]
    d_model_half = d_model // 2
    d_scales = (1 / 10000**(np.arange(d_model_half) / d_model_half))[np.newaxis,
                                                                     ...]

    position_encoding = np.empty(shape=(length, d_model), dtype=np.float32)
    position_encoding[:, 0::2] = np.sin(positions * d_scales)
    position_encoding[:, 1::2] = np.cos(positions * d_scales)

    return tf.cast(position_encoding, dtype=tf.float32)

  def call(self, x):
    length = tf.shape(x)[1]
    # TODO: Position embedding and token embedding should be in the same distrubution.
    x = self.embedding(x) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x


class MHA(tf.keras.layers.Layer):

  def __init__(self, *, num_heads, key_dim, dropout):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads,
                                                  key_dim=key_dim,
                                                  dropout=dropout)
    self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-7)


class GlobalSelfAttention(MHA):

  def call(self, x):
    x = x + self.mha(query=x, value=x, key=x)
    x = self.layer_norm(x)
    return x


class CrossAttention(MHA):

  def call(self, x, context):
    x = x + self.mha(query=x, key=context, value=context)
    x = self.layer_norm(x)
    return x


class CausalSelfAttention(MHA):

  def call(self, x):
    x = x + self.mha(query=x, value=x, key=x, use_causal_mask=True)
    x = self.layer_norm(x)
    return x


class FeedForward(tf.keras.layers.Layer):

  def __init__(self, *, d_model, dff, dropout):
    super().__init__()
    self.seq = tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model),
        tf.keras.layers.Dropout(dropout)
    ])
    self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-7)

  def call(self, x):
    x = x + self.seq(x)
    x = self.layer_norm(x)
    return x


class DecoderLayer(tf.keras.layers.Layer):

  def __init__(self, *, d_model, num_heads, dff, dropout):
    super().__init__()

    self.causal_self_attention = CausalSelfAttention(num_heads=num_heads,
                                                     key_dim=d_model,
                                                     dropout=dropout)

    self.cross_attention = CrossAttention(num_heads=num_heads,
                                          key_dim=d_model,
                                          dropout=dropout)

    self.ffn = FeedForward(d_model=d_model, dff=dff, dropout=dropout)

  def call(self, x, context):
    x = self.causal_self_attention(x=x)
    x = self.cross_attention(x=x, context=context)
    x = self.ffn(x)
    return x


class Decoder(tf.keras.layers.Layer):

  def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
               max_tokens, dropout):
    super().__init__()

    self.pos_embedding = PositionEmbedding(vocab_size=vocab_size,
                                           max_tokens=max_tokens,
                                           d_model=d_model)
    self.dropout = tf.keras.layers.Dropout(dropout)
    self.layers = [
        DecoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     dff=dff,
                     dropout=dropout) for _ in range(num_layers)
    ]

  def call(self, x, context):
    x = self.pos_embedding(x)
    x = self.dropout(x)
    for layer in self.layers:
      x = layer(x, context)

    return x


def make_decoder_only(*, num_layers, d_model, num_heads, dff, vocab_size,
                      max_tokens, dropout):
  return tf.keras.Sequential([
      PositionEmbedding(vocab_size=vocab_size,
                        max_tokens=max_tokens,
                        d_model=d_model),
      tf.keras.layers.Dropout(dropout),
      *[
          tf.keras.Sequential([
              CausalSelfAttention(
                  num_heads=num_heads, key_dim=d_model, dropout=dropout),
              FeedForward(d_model=d_model, dff=dff, dropout=dropout),
          ]) for _ in range(num_layers)
      ],
      tf.keras.layers.Dense(vocab_size),
  ])


class Transformer(tf.keras.Model):

  def __init__(self, *, num_layers, d_model, num_heads, dff, encoder_vocab_size,
               decoder_vocab_size, max_tokens, dropout):
    super().__init__()

    self.encoder = tf.keras.Sequential([
        PositionEmbedding(vocab_size=encoder_vocab_size,
                          max_tokens=max_tokens,
                          d_model=d_model),
        *[
            tf.keras.Sequential([
                GlobalSelfAttention(
                    num_heads=num_heads, key_dim=d_model, dropout=dropout),
                FeedForward(d_model=d_model, dff=dff, dropout=dropout),
            ]) for _ in range(num_layers)
        ],
        tf.keras.layers.Dropout(dropout),
    ])

    self.decoder = Decoder(num_layers=num_layers,
                           d_model=d_model,
                           num_heads=num_heads,
                           dff=dff,
                           vocab_size=decoder_vocab_size,
                           max_tokens=max_tokens,
                           dropout=dropout)

    self.final_linear_layer = tf.keras.layers.Dense(decoder_vocab_size)

  def call(self, x):
    context, x = x
    return self.final_linear_layer(self.decoder(x, self.encoder(context)))
