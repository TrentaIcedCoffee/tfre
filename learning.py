import tensorflow as tf


class CanonicalSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  '''https://arxiv.org/pdf/1706.03762, Section 5.3. Peeks after warmup then decays.'''

  def __init__(self, d_model: int, warmup=4000):
    super().__init__()
    self.d_model = d_model
    self.warmup = warmup

  def __call__(self, step):
    d_model = tf.cast(self.d_model, tf.float32)
    step = tf.cast(step, tf.float32)
    return tf.math.rsqrt(d_model) * tf.math.minimum(tf.math.rsqrt(step),
                                                    step * self.warmup**-1.5)

  def get_config(self):
    return {'d_model': self.d_model, 'warmup': self.warmup}


def masked_loss(label, pred):
  loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                           reduction='none')
  loss = loss_obj(label, pred)

  mask = label != 0
  loss *= tf.cast(mask, loss.dtype)
  return tf.reduce_sum(loss) / tf.reduce_sum(tf.cast(mask, tf.float32))


# TODO: Unit test.
def masked_loss_ignoring_context(separator_token):

  def masked_loss(label, pred):
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                             reduction='none')
    loss = loss_obj(label, pred)

    mask = tf.math.cumsum(
        tf.math.cumsum(
            tf.cast(label == separator_token, dtype=tf.int32),
            axis=-1,
        ),
        axis=-1,
    ) > 1
    mask &= label != 0
    loss *= tf.cast(mask, loss.dtype)

    return tf.reduce_sum(loss) / tf.reduce_sum(tf.cast(mask, tf.float32))

  return masked_loss


def masked_accuracy(label, pred):
  pred = tf.argmax(pred, axis=-1)
  label = tf.cast(label, pred.dtype)

  mask = label != 0
  match = (pred == label) & mask

  return tf.reduce_sum(tf.cast(match, tf.float32)) / tf.reduce_sum(
      tf.cast(mask, tf.float32))


def masked_accuracy_ignoring_context(separator_token):

  def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=-1)
    label = tf.cast(label, pred.dtype)

    mask = tf.math.cumsum(
        tf.math.cumsum(
            tf.cast(label == separator_token, dtype=tf.int32),
            axis=-1,
        ),
        axis=-1,
    ) > 1
    mask &= label != 0
    match = (pred == label) & mask

    return tf.reduce_sum(tf.cast(match, tf.float32)) / tf.reduce_sum(
        tf.cast(mask, tf.float32))

  return masked_accuracy
