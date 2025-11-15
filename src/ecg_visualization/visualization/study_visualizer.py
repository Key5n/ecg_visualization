from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import numpy.typing as npt
from optuna.artifacts import FileSystemArtifactStore
from optuna.study import Study
from optuna.trial import FrozenTrial
from tqdm import tqdm

from ecg_visualization.datasets.dataset import ECG_Entity
from ecg_visualization.utils.optuna_record import VisualizationRecord
from ecg_visualization.utils.timed_sequence import TimedSequence
from ecg_visualization.visualization.export import pdf_exporter
from ecg_visualization.visualization.layouts import (
    PaginationConfig,
    create_page_layout,
    paginate_signals,
)
from ecg_visualization.visualization.limits import compute_ylim
from ecg_visualization.visualization.plotters import (
    highlight_windows,
    plot_anomaly_score,
    plot_normal_beats,
    plot_signal,
    plot_symbols,
)
from ecg_visualization.visualization.styles import EXTREME_INTERVAL_COLOR


@dataclass(slots=True)
class SequenceBundle:
    signal: TimedSequence
    scores: TimedSequence
    annotations: TimedSequence
    beats: TimedSequence


class StudyVisualizer:
    """
    Encapsulates loading an Optuna study and exporting a PDF visualization for a
    single ECG entity.
    """

    def __init__(
        self,
        *,
        entity: ECG_Entity,
        study: Study,
        artifact_store: FileSystemArtifactStore,
        pagination_config: PaginationConfig,
        visualization_root: Path,
        rr_window_beats: int,
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        self.entity = entity
        self.study = study
        self.artifact_store = artifact_store
        self.pagination_config = pagination_config
        self.visualization_root = visualization_root
        self.rr_window_beats = rr_window_beats
        self.study_name = f"{entity.dataset_id} {entity.entity_id}"
        self._log = log_fn or tqdm.write

    def visualize(self) -> Path | None:
        trial = self._select_trial(self.study)
        if trial is None:
            return None

        vis_record = self._build_visualization_record(self.study, trial)
        if vis_record is None:
            return None

        sequences = self._build_sequences(vis_record)
        ts_paged = self._paginate_signals()
        if ts_paged.size == 0:
            self._log(f"Skipping {self.study_name}: no samples available.")
            return None

        signal_ylim = compute_ylim(
            self.entity.signals,
            lower_bound=-5.0,
            upper_bound=5.0,
        )
        score_ylim = compute_ylim(sequences.scores.values)
        extreme_windows = self.entity.get_extreme_rr_windows(
            self.rr_window_beats,
            lower_percentile=5.0,
            upper_percentile=95.0,
        )
        symbol_list = self._collect_symbols(sequences.annotations)
        output_path = self._prepare_output_path()

        self._export_pdf(
            ts_paged=ts_paged,
            sequences=sequences,
            signal_ylim=signal_ylim,
            score_ylim=score_ylim,
            extreme_windows=extreme_windows,
            symbol_list=symbol_list,
            output_path=output_path,
        )
        return output_path

    def _select_trial(self, study: Study) -> FrozenTrial | None:
        if not study.trials:
            self._log(f"Skipping {self.study_name}: no trials available.")
            return None
        return study.best_trial

    def _build_visualization_record(
        self,
        study: Study,
        trial: FrozenTrial,
    ) -> VisualizationRecord | None:
        try:
            return VisualizationRecord.from_trial(
                trial,
                study_name=study.study_name,
                artifact_store=self.artifact_store,
            )
        except ValueError as exc:
            self._log(f"Skipping {self.study_name}: {exc}")
            return None

    def _build_sequences(self, vis_record: VisualizationRecord) -> SequenceBundle:
        entity = self.entity

        signal_sequence = TimedSequence.from_time_axis(
            values=entity.signals,
            time_axis=np.arange(entity.signals.size, dtype=float) / entity.sr,
        )
        annotation_sequence = TimedSequence.from_time_axis(
            values=entity.annotation.symbol,
            time_axis=np.asarray(entity.annotation.sample, dtype=float) / entity.sr,
        )
        beat_sequence = TimedSequence.from_time_axis(
            values=np.zeros_like(entity.beats),
            time_axis=entity.beats / entity.sr,
        )
        return SequenceBundle(
            signal=signal_sequence,
            scores=vis_record.score_sequence,
            annotations=annotation_sequence,
            beats=beat_sequence,
        )

    def _paginate_signals(self) -> np.ndarray:
        total_samples = int(self.entity.signals.size)
        return paginate_signals(
            total_samples,
            self.entity.sr,
            self.pagination_config,
        )

    def _collect_symbols(self, annotation_sequence: TimedSequence) -> list[str]:
        if annotation_sequence.values.size == 0:
            return []
        unique_symbols = sorted(set(annotation_sequence.values.tolist()))
        return unique_symbols

    def _prepare_output_path(self) -> Path:
        dataset_dir = self.visualization_root / self.entity.dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir / f"{self.entity.entity_id}.pdf"

    def _export_pdf(
        self,
        *,
        ts_paged: np.ndarray,
        sequences: SequenceBundle,
        signal_ylim: tuple[float, float],
        score_ylim: tuple[float, float],
        extreme_windows: Iterable[tuple[float, float]],
        symbol_list: list[str],
        output_path: Path,
    ) -> None:
        n_rows = self.pagination_config.rows_per_page
        with pdf_exporter(str(output_path)) as exporter:
            for page_idx, ts_row in enumerate(ts_paged):
                fig, axs = create_page_layout(n_rows)
                for ts, ax in zip(ts_row, np.atleast_1d(axs), strict=True):
                    self._render_row(
                        ax=ax,
                        ts=ts,
                        sequences=sequences,
                        signal_ylim=signal_ylim,
                        score_ylim=score_ylim,
                        extreme_windows=extreme_windows,
                    )

                self._decorate_page(
                    fig=fig,
                    page_idx=page_idx,
                    symbol_list=symbol_list,
                )
                exporter.add_page(fig, pad_inches=0)
                plt.close(fig)

    def _render_row(
        self,
        *,
        ax: Axes,
        ts: npt.NDArray[np.float64],
        sequences: SequenceBundle,
        signal_ylim: tuple[float, float],
        score_ylim: tuple[float, float],
        extreme_windows: Iterable[tuple[float, float]],
    ) -> None:
        window_start, window_end = float(ts[0]), float(ts[-1])

        signal_in_window = sequences.signal.slice_between(window_start, window_end)
        beats_in_window = sequences.beats.slice_between(window_start, window_end)
        scores_in_window = sequences.scores.slice_between(window_start, window_end)
        symbols_in_window = sequences.annotations.slice_between(
            window_start, window_end
        )

        signal_values = self._align_signal_to_window(ts, signal_in_window.values)

        plot_signal(
            ax,
            ts,
            signal_values,
            ylim_lower=signal_ylim[0],
            ylim_upper=signal_ylim[1],
        )

        score_ax = ax.twinx()
        plot_anomaly_score(
            score_ax,
            scores_in_window.times.tolist(),
            scores_in_window.values.tolist(),
            ylim_lower=score_ylim[0],
            ylim_upper=score_ylim[1],
            label="Anomaly Score",
        )
        plot_normal_beats(
            ax,
            beats_in_window.times.tolist(),
            ylim_lower=signal_ylim[0],
        )
        plot_symbols(
            ax,
            symbols_in_window.samples,
            ylim_lower=signal_ylim[0],
        )
        highlight_windows(
            ax,
            extreme_windows,
            window_start=window_start,
            window_end=window_end,
            ylim_upper=signal_ylim[1],
            color=EXTREME_INTERVAL_COLOR,
        )

    @staticmethod
    def _align_signal_to_window(
        ts: npt.NDArray[np.float64],
        signal_values: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        if signal_values.size == ts.size:
            return signal_values

        padded = np.full(ts.shape, np.nan, dtype=float)
        limit = min(signal_values.size, ts.size)
        padded[:limit] = signal_values[:limit]
        return padded

    def _decorate_page(
        self,
        *,
        fig: Figure,
        page_idx: int,
        symbol_list: list[str],
    ) -> None:
        if page_idx == 0:
            symbol_tokens = "".join(symbol_list)
            fig.suptitle(
                f"{self.entity.dataset_name}: {self.entity.entity_id} "
                f"{symbol_tokens} {self.rr_window_beats}"
            )
        fig.supxlabel("Time (sec)")
        fig.subplots_adjust(left=0.08, right=0.94, bottom=0.05, top=0.95)
