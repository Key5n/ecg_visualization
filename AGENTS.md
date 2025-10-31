# Repository Guidelines

## Project Structure & Module Organization

- `src/ecg_visualization/`: source package with `datasets/` for PhysioNet loaders, `scripts/` for plotting CLI entry points, and `utils/` for helpers.
- `physionet.org/files/`: expected local mirror of raw ECG records as pulled by `dataset.sh`; keep directory names untouched so dataset classes resolve paths.
- `result/`: auto-created PDF outputs grouped by dataset id (e.g., `result/cudb/123.pdf`); clean up stale runs if regenerating.
- Supporting files: `pyproject.toml` (uv/hatch config), `uv.lock` (resolved dependencies), and `dataset.sh` for bulk downloads.

## Build, Test, and Development Commands

- Install: `uv sync` creates a Python 3.12 environment with numpy, seaborn, wfdb, tqdm.
- Run main pipeline: `uv run ecg-visualization` (or `uv run python -m ecg_visualization.scripts.ecg_visualization`) to generate per-record PDFs under `result/`.
- Dry-run a single dataset: edit `src/ecg_visualization/scripts/ecg_visualization.py` to adjust `data_sources`, then rerun the command above.
- Formatting check: `uv run python -m compileall src` catches syntax issues when no tests are present.

## Coding Style & Naming Conventions

- Follow PEP 8 with 4-space indentation, descriptive snake_case function and variable names, and PascalCase for dataclasses (e.g., `ECG_Dataset`).
- Prefer explicit typing (`numpy.typing.NDArray`, return annotations) and small, pure helpers in `utils/`.
- Keep module boundaries intentional: loaders in `datasets`, orchestration in `scripts`, reusable math in `utils`.

## Testing Guidelines

- Don't run any tests. this project is just an individual's trial.

## Commit & Pull Request Guidelines

- Follow the existing Conventional Commit style (`feat:`, `fix:`, `chore:`) as seen in recent history.
- Each PR should describe the dataset subset touched, expected impacts on generated PDFs, and reference any PhysioNet issues.
- Attach before/after snippets or PDF thumbnails when visuals change, and confirm large artifacts stay out of version control.
- Link related issues and request review once `uv run ecg-visualization` (or targeted tests) completes without errors.

## Data Handling Tips

- Keep `dataset.sh` runs manual to avoid accidental re-downloads; document new datasets in both the script and loader registry.
- When sharing results, export summary metrics instead of raw patient data unless privacy clearance is confirmed.
