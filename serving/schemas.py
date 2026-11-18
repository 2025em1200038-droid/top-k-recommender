"""Request and response contracts for the online API."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.utils import RETRIEVAL_K


class RecommendationQuery(BaseModel):
    """The fields every request needs, whether it arrives over HTTP or from a batch file."""

    # Unknown fields and wrongly typed values (for example "5" or 5.5) are rejected.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    user_id: int = Field(gt=0, strict=True, description="MovieLens user id")
    genre: str | None = Field(default=None, min_length=1, max_length=50)


class PredictRequest(RecommendationQuery):
    """Body of POST /predict."""

    k: int = Field(default=10, ge=1, le=RETRIEVAL_K, strict=True, description="Number of movies")


class BatchRecord(RecommendationQuery):
    """One row of a batch input file. Cells are text, so the id is parsed from digits."""

    record_id: str

    @field_validator("user_id", mode="before")
    @classmethod
    def parse_user_id(cls, value):
        if isinstance(value, str):
            value = value.strip()
            if not (value.isascii() and value.isdigit()):
                raise ValueError("must be a positive whole number")
            return int(value)
        return value

    @field_validator("genre", mode="before")
    @classmethod
    def blank_genre_means_no_filter(cls, value):
        return None if isinstance(value, str) and not value.strip() else value


class BatchOptions(BaseModel):
    """Command-line options of the batch script that need validating."""

    model_config = ConfigDict(extra="forbid")

    k: int = Field(default=10, ge=1, le=RETRIEVAL_K)


class Recommendation(BaseModel):
    rank: int
    movie_id: int
    title: str
    score: float


class PredictResponse(BaseModel):
    user_id: int
    genre: str | None
    k: int
    recommendations: list[Recommendation]
    degraded: bool = Field(description="True when the reranker failed and retriever order was used")
    latency_ms: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    n_users: int
    genres: list[str]


class FieldError(BaseModel):
    field: str
    message: str
    value: object | None = Field(default=None, description="The rejected value, truncated")


class ErrorResponse(BaseModel):
    error: str
    message: str | None = None
    details: list[FieldError] = []
