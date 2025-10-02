from typing import Any  # noqa: N999

from langchain_community.embeddings import JinaEmbeddings
from lfx.base.models.model import LCModelComponent
from lfx.field_typing import Embeddings
from lfx.io import BoolInput, DropdownInput, IntInput, Output, SecretStrInput

try:
    from langchain_community.embeddings import JinaEmbeddings
except ImportError as e:
    msg = "Please install langchain-community to use the Jina model."
    raise ImportError(msg) from e

HTTP_STATUS_OK = 200

JINA_API_URL: str = "https://api.jina.ai/v1/embeddings"

JINA_MODELS = [
    "jina-code-embeddings-0.5b",
    "jina-code-embeddings-1.5b",
    # "jina-embeddings-v4", postgres not support yet
    "jina-clip-v2",
    "jina-embeddings-v3",
    "jina-clip-v1",
    "jina-embeddings-v2-base-es",
    "jina-embeddings-v2-base-code",
    "jina-embeddings-v2-base-de",
    "jina-embeddings-v2-base-zh",
    "jina-embeddings-v2-base-en",
    "jina-embedding-b-en-v1",
]


class JinaEmbeddingsNew(JinaEmbeddings):
    task: str = "classification"
    late_chunking: bool = False
    truncate: bool = False
    output_dim: int = 1024

    def __init__(self, model_name, jina_api_key, task, late_chunking, truncate, output_dim):
        super().__init__(model_name=model_name, jina_api_key=jina_api_key)
        self.task = task
        self.late_chunking = late_chunking
        self.truncate = truncate
        self.output_dim = output_dim

    def embed(self, input_: Any) -> list[list[float]]:
        # Call Jina AI Embedding API
        resp = self.session.post(  # type: ignore  # noqa: PGH003
            JINA_API_URL, json={
                "input": input_,
                "model": self.model_name,
                "task": self.task,
                "late_chunking": self.late_chunking,
                "truncate": self.truncate,
                "dimensions": self.output_dim
            }
        ).json()
        if "data" not in resp:
            raise RuntimeError(resp["detail"])

        embeddings = resp["data"]

        # Sort resulting embeddings by index
        sorted_embeddings = sorted(embeddings, key=lambda e: e["index"])  # type: ignore  # noqa: PGH003

        # Return just the embeddings
        return [result["embedding"] for result in sorted_embeddings]


class JinaEmbeddingsComponent(LCModelComponent):
    display_name: str = "Jina Embeddings"
    description: str = "Generate embeddings using Jina models."
    documentation = "https://python.langchain.com/docs/integrations/text_embedding/jina"
    icon = "LangChain"
    name = "JinaEmbeddings"

    inputs = [
        DropdownInput(
            name="model_name",
            display_name="Jina Model",
            value="",
            options=[*JINA_MODELS],
            real_time_refresh=True,
            refresh_button=True,
            combobox=True,
            required=True,
        ),
        SecretStrInput(
            name="jina_api_key",
            display_name="Jina API Key",
            info="The Jina API Key.",
            advanced=False,
            value="JINA_API_KEY",
            required=True,
        ),
        DropdownInput(
            name="task",
            display_name="Downstream Task",
            info="Jina embeddings are general-purpose and excel at popular tasks. "
            "Once a task is set, they deliver highly optimized embeddings tailored to the task."
            "retrieval.query: Embedding queries in a query-document retrieval task. "
            "retrieval.passage: Embedding documents in a query-document retrieval task. "
            "separation: Clustering documents, visualizing corpus. "
            "classification: Text classification. "
            "text-matching: Semantic text similarity, general symmetric retrieval, recommendation, find alike, deduplication."  # noqa: E501
            "none: No LoRA adapter will be used. Return generic embedding, useful for debugging or hacking.",
            options=[
                "retrieval.query",
                "retrieval.passage",
                "separation",
                "classification",
                "text-matching",
                "none",
            ],
            value="classification",
        ),
        BoolInput(
            name="late_chunking",
            display_name="Late Chunking",
            info="Apply the late chunking technique to leverage the model's long-context capabilities for generating contextual chunk embeddings.",  # noqa: E501
            value=False,
        ),
        BoolInput(
            name="truncate",
            display_name="Truncate At Maximum Length",
            info="When enabled, the model will automatically drop the tail that extends beyond the maximum context length of 8192 tokens allowed by the model instead of throwing an error.",  # noqa: E501
            value=False,
        ),
        IntInput(
            name="output_dim",
            display_name="Output Dimension",
            info="Smaller dimensions enable efficient storage and retrieval, with minimal impact thanks to Matryoshka representation.",  # noqa: E501
            value=1024,
        ),
    ]

    outputs = [
        Output(display_name="Embeddings", name="embeddings", method="build_embeddings"),
    ]

    # def update_build_config(self, build_config: dict, field_value: Any, field_name: str | None = None):
    #     try:
    #         build_model = self.build_embeddings()
    #         ids = JINA_MODELS
    #         build_config["model"]["options"] = ids
    #         build_config["model"]["value"] = ids[0]
    #     except Exception as e:
    #         msg = f"Error getting model names: {e}"
    #         raise ValueError(msg) from e
    #     return build_config

    def build_embeddings(self) -> Embeddings:
        try:
            output = JinaEmbeddingsNew(
                model_name=self.model_name,
                jina_api_key=self.jina_api_key,
                task=self.task,
                late_chunking=self.late_chunking,
                truncate=self.truncate,
                output_dim=self.output_dim
            )
        except Exception as e:
            msg = f"Could not connect to JINA API. Error: {e}"
            raise ValueError(msg) from e
        return output

