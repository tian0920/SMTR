import ast
from pathlib import Path

SRC = Path("src/smtr")


def _imports_under(package: str) -> set[str]:
    root = SRC / package
    imports: set[str] = set()
    if not root.exists():
        return imports
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
    return imports


def test_core_does_not_import_toy_or_marble() -> None:
    imports = _imports_under("core")
    assert not any(name.startswith("smtr.toy") for name in imports)
    assert not any(name.startswith("smtr.marble") for name in imports)


def test_toy_does_not_import_marble() -> None:
    imports = _imports_under("toy")
    assert not any(name.startswith("smtr.marble") for name in imports)


def test_marble_does_not_import_toy() -> None:
    imports = _imports_under("marble")
    assert not any(name.startswith("smtr.toy") for name in imports)


def test_toy_and_marble_runners_are_different_classes() -> None:
    from smtr.marble.evaluation import MarbleExperimentRunner
    from smtr.toy.evaluation import ToyExperimentRunner

    assert MarbleExperimentRunner is not ToyExperimentRunner


def test_toy_and_marble_training_pipelines_are_different_classes() -> None:
    from smtr.marble.training import MarbleTrainingPipeline
    from smtr.toy.training import ToyTrainingPipeline

    assert MarbleTrainingPipeline is not ToyTrainingPipeline
