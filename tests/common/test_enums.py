from multilingual_embedding.common.enums import (
    SpecialToken,
    TokenizerModel,
)


def test_tokenizer_models() -> None:
    assert TokenizerModel.BPE.value == "bpe"
    assert TokenizerModel.UNIGRAM.value == "unigram"


def test_special_tokens() -> None:
    assert SpecialToken.UNK.value == "<unk>"
    assert SpecialToken.PAD.value == "<pad>"
