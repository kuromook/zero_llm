import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input='data/input.txt',
    model_prefix='tokenizer',
    vocab_size=500,
    model_type='bpe'
)

