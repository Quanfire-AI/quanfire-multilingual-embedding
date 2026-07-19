from multilingual_embedding.common.types import (
    CorpusText,
    DocumentText,
    ParagraphText,
    SentenceText,
)


def test_sentence_text() -> None:
    sentence: SentenceText = "Hello"
    assert isinstance(sentence, str)


def test_paragraph_text() -> None:
    paragraph: ParagraphText = [
        "Sentence 1",
        "Sentence 2",
    ]
    assert len(paragraph) == 2


def test_document_text() -> None:
    document: DocumentText = [
        [
            "Paragraph 1 Sentence 1",
            "Paragraph 1 Sentence 2",
        ],
        [
            "Paragraph 2 Sentence 1",
        ],
    ]
    assert len(document) == 2


def test_corpus_text() -> None:
    corpus: CorpusText = [
        [
            [
                "Document 1 Paragraph 1 Sentence 1",
                "Document 1 Paragraph 1 Sentence 2",
            ]
        ],
        [
            [
                "Document 2 Paragraph 1 Sentence 1",
            ]
        ],
    ]
    assert len(corpus) == 2
