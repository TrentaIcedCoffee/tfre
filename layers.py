import tensorflow as tf
import numpy as np
import dataclasses


class PositionEmbedding(tf.keras.layers.Layer):

  def __init__(self, *, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
    self.pos_encoding = self.generate_position_encoding(length=1_000_000,
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


class MMHA(tf.keras.layers.Layer):

  def __init__(self,
               *,
               segment_size,
               num_heads,
               d_model,
               use_causal_mask=False,
               dropout=None):
    super(MMHA, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    self.use_causal_mask = use_causal_mask
    self.query = tf.keras.layers.Dense(num_heads * d_model, use_bias=False)
    self.key = tf.keras.layers.Dense(num_heads * d_model, use_bias=False)
    self.value = tf.keras.layers.Dense(num_heads * d_model, use_bias=False)
    self.dropout = tf.keras.layers.Dropout(dropout) if dropout else None
    self.dense = tf.keras.layers.Dense(d_model, use_bias=False)

    # Infini-Transformer.
    self.segment_size = segment_size
    self.beta = self.add_weight(
        'beta',
        shape=(num_heads),
        initializer='zeros',
        trainable=True,
    )

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

  def elu_1(self, x):
    return tf.nn.elu(x) + 1

  def call(self, x, training=None):
    batch_size, length = tf.shape(x)[0], tf.shape(x)[1]

    def step(x_start, x_segment, memory, z, out):
      q, k, v = self.query(x_segment), self.key(x_segment), self.value(
          x_segment)

      q = tf.reshape(q, (batch_size, -1, self.num_heads, self.d_model))
      q = tf.transpose(q, perm=[0, 2, 1, 3])
      k = tf.reshape(k, (batch_size, -1, self.num_heads, self.d_model))
      k = tf.transpose(k, perm=[0, 2, 1, 3])
      v = tf.reshape(v, (batch_size, -1, self.num_heads, self.d_model))
      v = tf.transpose(v, perm=[0, 2, 1, 3])
      # qkv are in (batch_size, num_heads, segment_size, d_model).

      attn_score = q @ tf.transpose(k, perm=[0, 1, 3, 2]) / tf.math.sqrt(
          tf.cast(self.d_model, tf.float32)
      )  # (batch_size, num_heads, segment_size, segment_size)
      attn_score += -1e9 * tf.cast(~self.make_mask(tf.shape(x_segment)[1]),
                                   attn_score.dtype)
      attn_weight = tf.nn.softmax(attn_score, axis=-1)
      if self.dropout:
        attn_weight = self.dropout(attn_weight, training=training)
      attn_out = attn_weight @ v  # (batch_size, num_heads, segment_size, d_model)
      attn_out = tf.transpose(
          attn_out, perm=[0, 2, 1,
                          3])  # (batch_size, segment_size, num_heads, d_model)

      # Memory retrieval.
      q = tf.transpose(
          q, [0, 2, 1, 3])  # (batch_size, segment_size, num_heads, d_model)
      q = tf.reshape(q, (batch_size, -1, self.num_heads * self.d_model))
      memory_retrived = self.elu_1(q) @ memory / (self.elu_1(q) @ z)
      memory_retrived = tf.reshape(
          memory_retrived,
          (batch_size, -1, self.num_heads,
           self.d_model))  # (batch_size, segment_size, num_heads, d_model)

      step_out = tf.nn.sigmoid(self.beta)[..., tf.newaxis] * memory_retrived + (
          1 - tf.nn.sigmoid(self.beta))[..., tf.newaxis] * attn_out

      # Memory update.
      k = tf.transpose(k, [0, 2, 1, 3])
      k = tf.reshape(k, (batch_size, -1, self.num_heads * self.d_model))
      v = tf.transpose(v, [0, 2, 1, 3])
      v = tf.reshape(v, (batch_size, -1, self.num_heads * self.d_model))
      # kv are in (batch_size, segment_size, num_heads * d_model).
      memory += tf.reduce_sum(
          tf.transpose(self.elu_1(k), [0, 2, 1]) @ (v - self.elu_1(k) @ memory /
                                                    (self.elu_1(k) @ z)),
          axis=0)
      z += tf.reduce_sum(tf.transpose(self.elu_1(k), [0, 2, 1]),
                         axis=-1,
                         keepdims=True)

      return memory, z, tf.concat(
          [
              out[:, :x_start, ...],
              step_out,
              out[:, x_start + self.segment_size:, ...],
          ],
          axis=1,
      ),

    out = tf.zeros(shape=(batch_size, length, self.num_heads, self.d_model))
    memory = tf.zeros(
        (self.num_heads * self.d_model, self.num_heads * self.d_model))
    z = tf.ones((batch_size, self.num_heads * self.d_model, 1)) / self.d_model
    for i in range(0, length, self.segment_size):
      x_segment = x[:, i:i + self.segment_size, :]
      memory, z, out = step(i, x_segment, memory, z, out)

    return self.dense(
        tf.reshape(out, (batch_size, length, self.num_heads * self.d_model)))


class ResidualNormedMHA(tf.keras.layers.Layer):

  def __init__(self,
               *,
               segment_size,
               num_heads,
               d_model,
               use_causal_mask=False,
               dropout=None):
    super().__init__()
    if segment_size:
      self.mha = MMHA(segment_size=segment_size,
                      num_heads=num_heads,
                      d_model=d_model,
                      use_causal_mask=use_causal_mask,
                      dropout=dropout)
    else:
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


class MoE(tf.keras.layers.Layer):

  def __init__(self, *, experts: list[tf.keras.layers.Layer], d_output, k):
    super().__init__()
    self.router = tf.keras.layers.Dense(len(experts), use_bias=False)
    self.experts = experts
    self.k = k
    # TODO: d_output should be inferred from the experts.
    self.d_output = d_output

  def call(self, x):
    router_logits = self.router(x)
    weights, selected_experts = tf.math.top_k(router_logits, k=self.k)
    weights = tf.nn.softmax(weights, axis=-1)

    results = tf.zeros(shape=(tf.shape(x)[0], tf.shape(x)[1], self.d_output))
    for i, expert in enumerate(self.experts):
      pos = tf.where(selected_experts == i)
      results = tf.tensor_scatter_nd_add(
          results,
          # T, 2
          pos[:, :2],
          # (T, 1) * (T, d_model) -> T, d_model
          tf.gather_nd(weights, pos)[..., tf.newaxis] *
          expert(tf.gather_nd(x, pos[:, :2])),
      )

    return results


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
               dropout):
    super().__init__()

    self.pos_embedding = PositionEmbedding(vocab_size=vocab_size,
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


@dataclasses.dataclass
class MoEFeature:
  num_experts: int
  k: int


@dataclasses.dataclass
class InfiniFeature:
  segment_size: int


@dataclasses.dataclass
class Feature:
  moe: MoEFeature | None = None
  infini: InfiniFeature | None = None


@dataclasses.dataclass
class HParam:
  num_layers: int
  d_model: int
  num_heads: int
  dff: int
  dropout: float
  feature: Feature


def make_decoder_only(*, vocab_size: int, hparam: HParam):
  decoder = tf.keras.Sequential([
      PositionEmbedding(vocab_size=vocab_size, d_model=hparam.d_model),
      tf.keras.layers.Dropout(hparam.dropout),
  ])
  for _ in range(hparam.num_layers):
    decoder.add(
        ResidualNormedMHA(segment_size=hparam.feature.infini.segment_size
                          if hparam.feature.infini else None,
                          num_heads=hparam.num_heads,
                          d_model=hparam.d_model,
                          use_causal_mask=True,
                          dropout=hparam.dropout))
    decoder.add(
        MoE(
            experts=[
                FeedForward(d_model=hparam.d_model,
                            dff=hparam.dff,
                            dropout=hparam.dropout)
                for _ in range(hparam.feature.moe.num_experts)
            ],
            k=hparam.feature.moe.k,
            d_output=hparam.d_model,
        ) if hparam.feature.moe else FeedForward(
            d_model=hparam.d_model, dff=hparam.dff, dropout=hparam.dropout))

  decoder.add(tf.keras.layers.Dense(vocab_size, use_bias=False))

  return decoder


class Transformer(tf.keras.Model):

  def __init__(self, *, num_layers, d_model, num_heads, dff, encoder_vocab_size,
               decoder_vocab_size, dropout):
    super().__init__()

    self.encoder = tf.keras.Sequential([
        PositionEmbedding(vocab_size=encoder_vocab_size, d_model=d_model),
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


class LSTM(tf.keras.layers.Layer):

  def __init__(self, *, length, d_model):
    super().__init__()
    self.length = length
    self.d_model = d_model
    self.kernel = tf.keras.layers.Dense(d_model * 3, use_bias=False)
    self.recurrent_kernel = tf.keras.layers.Dense(d_model * 3, use_bias=False)

  def call(self, x):
    '''A slight variation of LSTM. 
      - Drops the input gate.
      - Highlights that prediction is based on the long-term memory, selected by output gate.
    '''
    long_term_memory = tf.random.normal((x.shape[0], self.d_model))
    output_sequence = [tf.random.normal((x.shape[0], self.d_model))]
    for i in range(self.length):
      forget, output, cell = tf.split(
          self.kernel(x[:, i, :]) + self.recurrent_kernel(output_sequence[-1]),
          3,
          axis=-1,
      )

      long_term_memory = long_term_memory * tf.sigmoid(forget) + (
          1 - tf.sigmoid(forget)) * tf.tanh(cell)
      output_sequence.append(tf.sigmoid(output) * tf.tanh(long_term_memory))

    return tf.stack(output_sequence[1:], axis=1)


class GRU(tf.keras.layers.Layer):

  def __init__(self, *, length, d_model):
    super().__init__()
    self.length = length
    self.d_model = d_model
    self.reset_gate = tf.keras.layers.Dense(d_model, use_bias=False)
    self.update_gate = tf.keras.layers.Dense(d_model, use_bias=False)
    self.candidate = tf.keras.layers.Dense(d_model, use_bias=False)

  def call(self, x):
    '''A slight variation of GRU that concats the input to the hidden state for simplicity.'''
    hidden_states = [tf.zeros((x.shape[0], self.d_model))]
    for i in range(self.length):
      x_i = tf.concat([x[:, i, :], hidden_states[-1]], axis=-1)
      reset = tf.sigmoid(self.reset_gate(x_i))
      update = tf.sigmoid(self.update_gate(x_i))
      candidate = tf.tanh(
          self.candidate(
              tf.concat([x[:, i, :], reset * hidden_states[-1]], axis=-1)))
      hidden = update * hidden_states[-1] + (1 - update) * candidate
      hidden_states.append(hidden)
    return tf.stack(hidden_states[1:], axis=1)


class MRU(tf.keras.layers.Layer):

  class RU(tf.keras.layers.Layer):

    def __init__(self, *, length, d_model):
      super().__init__()
      self.length = length
      self.d_model = d_model
      self.forget_gate = tf.keras.layers.Dense(d_model, use_bias=False)
      self.candidate_gate = tf.keras.layers.Dense(d_model, use_bias=False)

    def call(self, x):
      hidden_states = [tf.zeros((x.shape[0], self.d_model))]
      for i in range(self.length):
        x_i = tf.concat([x[:, i, :], hidden_states[-1]], axis=-1)
        forget = tf.sigmoid(self.forget_gate(x_i))
        new_info = tf.tanh(self.candidate_gate(x_i))
        hidden_states.append(forget * hidden_states[-1] +
                             (1 - forget) * new_info)
      return tf.stack(hidden_states[1:], axis=1)

  def __init__(self, *, length, d_model, num_heads):
    super().__init__()
    self.rus = [
        self.RU(length=length, d_model=d_model) for _ in range(num_heads)
    ]
    self.linear = tf.keras.layers.Dense(d_model)

  def call(self, x):
    x = tf.concat([ru(x) for ru in self.rus], axis=-1)
    return self.linear(x)
