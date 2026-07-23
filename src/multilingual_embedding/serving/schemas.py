"""
Request and response shapes for the embeddings endpoint.

The schema is the de facto industry standard one, so a client migrates by
changing a base URL. It is reproduced here rather than imported because
depending on a vendor's SDK to describe a wire format this project does
not control would be a dependency taken for a data class.

One field is an addition rather than a copy: ``input_type``. The standard
schema has no place to say whether a string is a query or a passage, and
for the models this project serves that distinction is not decoration —
see :mod:`multilingual_embedding.serving.app` for why the server refuses
to guess.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Bounds the work one request can ask for. A request is embedded in a
# single batch, so without a ceiling one caller can hold the event loop
# for an unbounded time and every other caller waits behind it.
MAX_BATCH = 512

MAX_INPUT_CHARACTERS = 100_000


class EmbeddingsRequest(BaseModel):
    """One embeddings call: a string or a batch of them."""

    input: str | list[str]

    model: str | None = None

    input_type: Literal["query", "passage"] | None = Field(
        default=None,
        description=(
            "Which side of an asymmetric model this text belongs to. "
            "Required when the served model records query and passage "
            "prefixes and the server has no configured default."
        ),
    )

    encoding_format: Literal["float"] = "float"

    @field_validator("input")
    @classmethod
    def _reject_empty_or_oversized(cls, value: str | list[str]) -> str | list[str]:
        texts = [value] if isinstance(value, str) else value

        if not texts:
            raise ValueError("input must contain at least one string")

        if len(texts) > MAX_BATCH:
            raise ValueError(f"input contains {len(texts)} items; the maximum is {MAX_BATCH}")

        # An all-whitespace string is almost always a caller bug — an
        # empty field passed through a template — and it embeds to
        # something meaningless rather than failing, so it is refused
        # here instead of quietly occupying a row in the response.
        for index, text in enumerate(texts):
            if not text.strip():
                raise ValueError(f"input[{index}] is empty or whitespace only")

            if len(text) > MAX_INPUT_CHARACTERS:
                raise ValueError(
                    f"input[{index}] is {len(text)} characters; "
                    f"the maximum is {MAX_INPUT_CHARACTERS}"
                )

        return value

    def texts(self) -> list[str]:
        """The input as a list, whichever form it arrived in."""

        return [self.input] if isinstance(self.input, str) else list(self.input)


class EmbeddingObject(BaseModel):
    """One vector, carrying the index of the input that produced it."""

    object: Literal["embedding"] = "embedding"

    index: int

    embedding: list[float]


class Usage(BaseModel):
    """Token accounting.

    Both fields are the same number because an embeddings call has no
    completion half. It is reported anyway: clients written against the
    standard schema read it, and omitting it makes them fail on a key
    error rather than on anything meaningful.
    """

    prompt_tokens: int

    total_tokens: int


class EmbeddingsResponse(BaseModel):
    """The batch of vectors, in the order the inputs arrived."""

    object: Literal["list"] = "list"

    data: list[EmbeddingObject]

    model: str

    usage: Usage

    prefix_applied: str | None = Field(
        default=None,
        description=(
            "The prefix this server prepended before encoding, echoed so a "
            "caller can tell which side of the model answered. Null when "
            "the served model records no prefixes."
        ),
    )


class ModelCard(BaseModel):
    """What is being served, in enough detail to reproduce a vector."""

    id: str

    object: Literal["model"] = "model"

    owned_by: str = "quanfire"

    dimension: int

    max_length: int

    normalized: bool

    query_prefix: str | None = None

    passage_prefix: str | None = None

    framework_version: str


class ModelsResponse(BaseModel):
    object: Literal["list"] = "list"

    data: list[ModelCard]


class ErrorBody(BaseModel):
    message: str

    type: str

    param: str | None = None


class ErrorResponse(BaseModel):
    """The standard error envelope, so existing clients parse failures too."""

    error: ErrorBody
