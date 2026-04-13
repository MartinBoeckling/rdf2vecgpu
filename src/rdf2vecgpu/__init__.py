from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("rdf2vecgpu")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Lazy imports — GPU libraries (cudf, cugraph, torch) are only loaded when
# GPU_RDF2Vec is actually accessed, not on `import rdf2vecgpu`.
from .config import RDF2VecConfig


def __getattr__(name):
    if name == "GPU_RDF2Vec":
        from .gpu_rdf2vec import GPU_RDF2Vec

        return GPU_RDF2Vec
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["GPU_RDF2Vec", "RDF2VecConfig"]
