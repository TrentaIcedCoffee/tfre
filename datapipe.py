from typing import Iterable, Callable
from tiktoken_ext import openai_public
import tensorflow as tf
import tiktoken
import threading

_tokenizer_lock = threading.RLock()
TOKENIZER = None


def get_tokenizer(vocab_size: int) -> tiktoken.Encoding:
  """Returns a smaller version of cl100k_base tokenizer of `vocab_size`."""
  global TOKENIZER
  if TOKENIZER:
    return TOKENIZER

  with _tokenizer_lock:
    if TOKENIZER:
      return TOKENIZER

    config = openai_public.cl100k_base()
    mergeable_ranks = {
        token: rank
        for token, rank in config['mergeable_ranks'].items()
        if rank < vocab_size
    }
    config = {
        **config,
        'name': 'cl4k_base',
        'mergeable_ranks': mergeable_ranks,
        'special_tokens': {
            '<|endoftext|>':
                len(mergeable_ranks) +
                1,  # I don't know why rank `len(mergeable_ranks)` is not used but following the same pattern.
            '<|endofprompt|>': len(mergeable_ranks) + 2,
        },
    }
    TOKENIZER = tiktoken.Encoding(**config)
    return TOKENIZER


def build_char_tokenizer(
    chars: Iterable[str]) -> tuple[Callable, Callable, int]:
  vocab = sorted(list(set(chars)))
  vocab = ['[PAD]', '[UNK]', '[START]', '[END]'] + vocab

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
            tf.concat([
                ['[START]'],
                s,
                ['[END]'],
            ], axis=0)),
        inputs,
        fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=tf.int32),
    )

  def detokenize_batch(inputs):
    return tf.map_fn(
        lambda idxs: idx_to_word.lookup(tf.boolean_mask(idxs, idxs >= 4)),
        inputs,
        fn_output_signature=tf.TensorSpec(shape=[None], dtype=tf.string),
    )

  return tokenize_batch, detokenize_batch, len(vocab)


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
