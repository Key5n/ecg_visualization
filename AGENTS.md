# Repository Guidelines

## Project Structure & Module Organization

- `src/ecg_visualization/`: source package with `datasets/` for PhysioNet loaders, `scripts/` for plotting CLI entry points, and `utils/` for helpers.
- `physionet.org/files/`: expected local mirror of raw ECG records as pulled by `dataset.sh`; keep directory names untouched so dataset classes resolve paths.
- `result/`: auto-created PDF outputs grouped by dataset id (e.g., `result/cudb/123.pdf`); clean up stale runs if regenerating.
- Supporting files: `pyproject.toml` (uv/hatch config), `uv.lock` (resolved dependencies), and `dataset.sh` for bulk downloads.

## Build, Test, and Development Commands

- Install: `uv sync` creates a Python 3.12 environment

## Coding Style & Naming Conventions

- Follow PEP 8 with 4-space indentation, descriptive snake_case function and variable names, and PascalCase for dataclasses (e.g., `ECG_Dataset`).
- Prefer explicit typing (`numpy.typing.NDArray`, return annotations) and small, pure helpers in `utils/`.
- Write main script in `src/ecg_visualization/scripts/study.py` and Write visualization script in `src/ecg_visualization/scripts/visualize.py`
  - Use visualization logic instance `StudyVisualizer` in `visualize.py`
  - In `study.py`, store any information to be used in `StudyVisualizer` to RDB by optuna or `FileSystemArtifactStore`
  - In `StudyVisualizer`, load any information to be used in visualization from RDB by optuna or `FileSystemArtifactStore`

## Testing Guidelines

- Use `unittest` for tests and place them under `tests/`.
- Keep `tests/__init__.py` so discovery works consistently.
- Preferred discovery command: `uv run python -m unittest discover -s tests`
- Don't run any tests. This project is just an individual's trial.

## Commit & Pull Request Guidelines

- Follow the existing Conventional Commit style (`feat:`, `fix:`, `chore:`) as seen in recent history.
