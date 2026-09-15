# Seq2Seq using PyTorch

This repo contains notebooks that help to understand and implement sequence-to-sequence (seq2seq) models using [PyTorch](https://github.com/pytorch/pytorch). Specifically, we'll train models to translate from German to English.

## Getting Started

Install the required dependencies with: `pip install -r requirements.txt --upgrade`.

We'll also make use of [spaCy](https://spacy.io/) to tokenize our data which requires installing both the English and German models with:

```bash
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

## Notebooks

- 1 - [Sequence to Sequence Learning with Neural Networks](https://github.com/Jopaul07/seq2seq/blob/main/build_seq2seq_part1_encdec.ipynb)

  This first notebook covers the workflow of a seq2seq project with PyTorch. We'll cover the basics of seq2seq networks using encoder-decoder models, how to implement these models in PyTorch, and how to use the datasets/spacy/evaluate libraries to do all of the heavy lifting. The model itself will be based off an implementation of [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215), which uses multi-layer LSTMs.

- 2 - [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://github.com/Jopaul07/seq2seq/blob/main/build_seq2seq_part2_gru.ipynb)

  Now we have the basic workflow covered, this notebook will focus on improving our results. Building on our knowledge of PyTorch, we'll implement a second model, which helps with the information compression problem faced by encoder-decoder models. This model will be based off an implementation of [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078), which uses GRUs.

- 3 - [Neural Machine Translation by Jointly Learning to Align and Translate](https://github.com/Jopaul07/seq2seq/blob/main/build_seq2seq_part3_attn.ipynb)

  Next, we learn about attention by implementing [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473). This further allievates the information compression problem by allowing the decoder to "look back" at the input sentence by creating context vectors that are weighted sums of the encoder hidden states. The weights for this weighted sum are calculated via an attention mechanism, where the decoder learns to pay attention to the most relevant words in the input sentence.
