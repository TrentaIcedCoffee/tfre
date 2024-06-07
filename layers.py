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

  def __init__(self,
               *,
               num_heads,
               d_model,
               use_causal_mask=False,
               dropout=None):
    super(MHA, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    self.use_causal_mask = use_causal_mask
    self.query = tf.keras.layers.Dense(num_heads * d_model, use_bias=False)
    self.key = tf.keras.layers.Dense(num_heads * d_model, use_bias=False)
    self.value = tf.keras.layers.Dense(num_heads * d_model, use_bias=False)
    self.dropout = tf.keras.layers.Dropout(dropout) if dropout else None
    self.dense = tf.keras.layers.Dense(d_model, use_bias=False)

  def make_mask(self, length):
    ''' Causal mask.
    E.g.
      [[[[ True, False, False],
         [ True,  True, False],
         [ True,  True,  True]]]]
    '''
    i = tf.range(length)[:, tf.newaxis]
    j = tf.range(length)
    return tf.reshape(tf.cast(i >= j, dtype=tf.bool), (1, 1, length, length))

  def call(self, x, training=None):
    batch_size, length = tf.shape(x)[0], tf.shape(x)[1]
    q, k, v = self.query(x), self.key(x), self.value(x)
    q = tf.reshape(q, (batch_size, -1, self.num_heads, self.d_model))
    q = tf.transpose(q, perm=[0, 2, 1, 3])
    k = tf.reshape(k, (batch_size, -1, self.num_heads, self.d_model))
    k = tf.transpose(k, perm=[0, 2, 1, 3])
    v = tf.reshape(v, (batch_size, -1, self.num_heads, self.d_model))
    v = tf.transpose(v, perm=[0, 2, 1, 3])

    attn_score = q @ tf.transpose(k, perm=[0, 1, 3, 2]) / tf.math.sqrt(
        tf.cast(self.d_model,
                tf.float32))  # (batch_size, num_heads, seq_len, seq_len)
    attn_score += -1e9 * tf.cast(~self.make_mask(length), attn_score.dtype)
    attn_weight = tf.nn.softmax(attn_score, axis=-1)
    if self.dropout:
      attn_weight = self.dropout(attn_weight, training=training)
    attn_out = attn_weight @ v  # (batch_size, num_heads, seq_len, d_model)
    attn_out = tf.transpose(attn_out, perm=[0, 2, 1, 3])
    attn_out = tf.reshape(attn_out,
                          (batch_size, -1, self.num_heads * self.d_model))
    return self.dense(attn_out)


class ResidualNormedMHA(tf.keras.layers.Layer):

  def __init__(self,
               *,
               num_heads,
               d_model,
               use_causal_mask=False,
               dropout=None):
    super().__init__()
    self.mha = MHA(num_heads=num_heads,
                   d_model=d_model,
                   use_causal_mask=use_causal_mask,
                   dropout=dropout)
    self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-7)

  def call(self, x):
    x = x + self.mha(x)
    x = self.layer_norm(x)
    return x


class MultiHeadAttention(tf.keras.layers.Layer):

  def __init__(self, *, num_heads, key_dim, dropout):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads,
                                                  key_dim=key_dim,
                                                  dropout=dropout)
    self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-7)


class GlobalSelfAttention(MultiHeadAttention):

  def call(self, x):
    x = x + self.mha(query=x, value=x, key=x)
    x = self.layer_norm(x)
    return x


class CrossAttention(MultiHeadAttention):

  def call(self, x, context):
    x = x + self.mha(query=x, key=context, value=context)
    x = self.layer_norm(x)
    return x


class CausalSelfAttention(MultiHeadAttention):

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
              ResidualNormedMHA(num_heads=num_heads,
                                d_model=d_model,
                                use_causal_mask=True,
                                dropout=dropout),
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


class RNN(tf.keras.layers.Layer):

  def __init__(self, *, length, d_model):
    super().__init__()
    self.d_model = d_model
    self.length = length
    self.w_x = tf.keras.layers.Dense(d_model, use_bias=False)
    self.w_h = tf.keras.layers.Dense(d_model, use_bias=False)

  def call(self, x):
    output = [tf.random.normal((x.shape[0], self.d_model))]
    for i in range(self.length):
      output.append(tf.nn.tanh(self.w_x(x[:, i, :]) + self.w_h(output[-1])))
    return tf.stack(output[1:], axis=1)
