import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file='tokenizer.model')
ids = sp.encode("こんにちは、世界！")
print(ids)
print(sp.decode(ids))

