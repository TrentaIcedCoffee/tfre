from typing import Iterable, Callable
import tensorflow as tf


def build_simple_tokenizers(examples: Iterable[str],
                            separator=' ') -> tuple[Callable, Callable, int]:
  """Builds tokenizers for a given set of examples.
    Returns:
      tokenize: string tensor -> int tensor.
      detokenize: int tensor -> string tensor.
      vocab_size: The size of the vocabulary.
  """
  vocab = set()
  for example in examples:
    if isinstance(example, tf.Tensor):
      example = example.numpy()
    vocab.update(example.decode('utf-8').split(separator))
  vocab = ['[PAD]', '[UNK]', '[START]', '[END]'] + list(sorted(vocab))

  idx_to_word = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
          keys=range(len(vocab)),
          values=vocab,
      ),
      default_value='[UNK]',
  )
  word_to_idx = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
          keys=vocab,
          values=range(len(vocab)),
      ),
      default_value=1,
  )

  def tokenize_batch(inputs):
    return tf.map_fn(
        lambda s: word_to_idx.lookup(
            tf.concat(
                [
                    ['[START]'],
                    tf.strings.split(s, sep=separator),
                    ['[END]'],
                ],
                axis=0,
            )),
        inputs,
        fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=tf.int32),
    )

  def detokenize_batch(inputs):
    return tf.map_fn(
        lambda idxs: tf.strings.join(
            idx_to_word.lookup(tf.boolean_mask(idxs, idxs >= 4)),
            separator=separator,
        ),
        inputs,
        fn_output_signature=tf.TensorSpec(shape=[], dtype=tf.string),
    )

  return tokenize_batch, detokenize_batch, len(vocab)
