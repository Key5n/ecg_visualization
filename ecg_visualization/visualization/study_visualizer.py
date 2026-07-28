from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from optuna.artifacts import FileSystemArtifactStore
from optuna.study import Study
from optuna.trial import FrozenTrial

from ecg_visualization.core.analysis import get_extreme_rr_windows
from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.datasets.physionet import DATASET_REGISTRY
from ecg_visualization.utils.optuna_record import (
    Record,
    VisualizationRecord,
    get_study_identifiers,
)
from ecg_visualization.utils.timed_sequence import TimedSequence
from ecg_visualization.visualization.export import PdfExporter, pdf_exporter
from ecg_visualization.visualization.layouts import (
    PaginationConfig,
    create_page_layout,
    paginate_signals,
)
from ecg_visualization.visualization.limits import compute_ylim
from ecg_visualization.visualization.pdf_metadata import build_pdf_metadata
from ecg_visualization.visualization.plotters import (
    highlight_windows,
    plot_anomaly_score,
    plot_aux_notes,
    plot_histogram,
    plot_normal_beats,
    plot_signal,
    plot_symbols,
)
from ecg_visualization.visualization.styles import (
    EXTREME_INTERVAL_COLOR,
    TRAINING_INTERVAL_COLOR,
)
from ecg_visualization.visualization.text import sanitize_annotation_text

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SequenceBundle:
    signal: TimedSequence[np.float64]
    scores: TimedSequence[np.float64]
    annotations: TimedSequence[np.str_]
    beats: TimedSequence[np.int_]
    aux_notes: TimedSequence[np.object_]


class StudyVisualizer:
    """
    Encapsulates loading an Optuna study and exporting a PDF visualization for a
    single ECG entity.
    """

    def __init__(
        self,
        *,
        study: Study,
        artifact_store: FileSystemArtifactStore,
        pagination_config: PaginationConfig,
        visualization_root: Path,
        rr_window_beats: int,
    ) -> None:
        self.study = study
        self.artifact_store = artifact_store
        self.pagination_config = pagination_config
        self.visualization_root = visualization_root
        self.rr_window_beats = rr_window_beats
        self.study_name = study.study_name
        self.entity = self._load_entity_from_study(study)

    def visualize(self) -> Path | None:
        trial = self._select_trial(self.study)
        if trial is None:
            return None

        vis_record = self._build_visualization_record(self.study, trial)
        if vis_record is None:
            return None

        sequences = self._build_sequences(vis_record)
        training_window = self._load_training_window(vis_record)
        ts_paged = self._paginate_signals()
        if ts_paged.size == 0:
            LOGGER.warning(f"Skipping {self.study_name}: no samples available.")
            return None

        signal_ylim = compute_ylim(
            self.entity.signals,
            lower_bound=-5.0,
            upper_bound=5.0,
        )
        score_ylim = compute_ylim(sequences.scores.values)
        extreme_windows = get_extreme_rr_windows(
            self.entity,
            self.rr_window_beats,
            lower_percentile=5.0,
            upper_percentile=95.0,
        )
        symbol_list = self._collect_symbols(sequences.annotations)
        aux_note_summary = self._collect_aux_note_summary()
        output_path = self._prepare_output_path()
        pdf_metadata = build_pdf_metadata(
            entity=self.entity,
            record=vis_record.record,
        )

        self._export_pdf(
            ts_paged=ts_paged,
            sequences=sequences,
            signal_ylim=signal_ylim,
            score_ylim=score_ylim,
            extreme_windows=extreme_windows,
            symbol_list=symbol_list,
            aux_note_summary=aux_note_summary,
            output_path=output_path,
            training_window=training_window,
            pdf_metadata=pdf_metadata,
        )
        return output_path

    def _load_entity_from_study(self, study: Study) -> ECGEntity:
        dataset_id, entity_id = get_study_identifiers(study)
        dataset_cls = DATASET_REGISTRY.get(dataset_id)
        if dataset_cls is None:
            raise ValueError(
                f"Unknown dataset id '{dataset_id}' for study {study.study_name}"
            )
        return dataset_cls.get_entity(entity_id=entity_id)

    def _select_trial(self, study: Study) -> FrozenTrial | None:
        if not study.trials:
            LOGGER.warning(f"Skipping {self.study_name}: no trials available.")
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
            if str(exc).startswith("Missing artifact id for score_sequence"):
                record = Record.from_trial(trial, study_name=study.study_name)
                empty_scores = TimedSequence(
                    values=np.asarray([], dtype=float),
                    times=np.asarray([], dtype=float),
                )
                LOGGER.warning(
                    f"Missing score_sequence artifact for {self.study_name}; "
                    "plotting without anomaly scores."
                )
                return VisualizationRecord(
                    record=record,
                    score_sequence=empty_scores,
                )
            LOGGER.warning(f"Skipping {self.study_name}: {exc}")
            return None

    def _build_sequences(self, vis_record: VisualizationRecord) -> SequenceBundle:
        entity = self.entity

        signal_sequence = TimedSequence(
            values=entity.signals,
            times=np.arange(entity.signals.size, dtype=float)
            / entity.dataset.sampling_rate_hz,
        )
        annotation_sequence = TimedSequence(
            values=np.asarray(entity.annotation.symbol, dtype=str),
            times=np.asarray(entity.annotation.sample, dtype=float)
            / entity.dataset.sampling_rate_hz,
        )
        aux_note_sequence = TimedSequence(
            values=np.asarray(entity.aux_notes, dtype=object),
            times=np.asarray(entity.annotation.sample, dtype=float)
            / entity.dataset.sampling_rate_hz,
        )
        beat_sequence = TimedSequence(
            values=np.zeros_like(entity.beats),
            times=entity.beats / entity.dataset.sampling_rate_hz,
        )
        return SequenceBundle(
            signal=signal_sequence,
            scores=vis_record.score_sequence,
            annotations=annotation_sequence,
            beats=beat_sequence,
            aux_notes=aux_note_sequence,
        )

    def _load_training_window(
        self,
        vis_record: VisualizationRecord,
    ) -> tuple[float, float] | None:
        attrs = vis_record.record.user_attrs
        start_time = attrs.get("normal_window_start_time")
        end_time = attrs.get("normal_window_end_time")
        if start_time is None or end_time is None:
            return None

        start_time = float(start_time)
        end_time = float(end_time)
        if end_time <= start_time:
            return None
        return (start_time, end_time)

    def _paginate_signals(self) -> np.ndarray:
        total_samples = int(self.entity.signals.size)
        return paginate_signals(
            total_samples,
            self.entity.dataset.sampling_rate_hz,
            self.pagination_config,
        )

    @staticmethod
    def _sanitize_text(value: object) -> str:
        return sanitize_annotation_text(value)

    def _collect_symbols(
        self, annotation_sequence: TimedSequence[np.str_]
    ) -> list[str]:
        if annotation_sequence.values.size == 0:
            return []
        unique_symbols = sorted(set(annotation_sequence.values.tolist()))
        return [
            symbol
            for symbol in (
                self._sanitize_text(raw_symbol) for raw_symbol in unique_symbols
            )
            if symbol
        ]

    def _collect_aux_note_summary(self) -> str:
        notes = [
            self._sanitize_text(note)
            for note in self.entity.aux_notes
            if self._sanitize_text(note)
        ]
        if not notes:
            return "None"

        counts = Counter(notes)
        ordered_notes = sorted(counts.keys(), key=lambda n: (-counts[n], n))
        return ", ".join(f"{note} ({counts[note]})" for note in ordered_notes)

    def _prepare_output_path(self) -> Path:
        dataset_dir = self.visualization_root / self.entity.dataset.dataset_id
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
        aux_note_summary: str,
        output_path: Path,
        training_window: tuple[float, float] | None,
        pdf_metadata: dict[str, str] | None = None,
    ) -> None:
        n_rows = self.pagination_config.rows_per_page
        with pdf_exporter(str(output_path), metadata=pdf_metadata) as exporter:
            self._render_histogram_pages(exporter)
            for signal_page_idx, ts_row in enumerate(ts_paged):
                fig, axs = create_page_layout(n_rows)
                for ts, ax in zip(ts_row, np.atleast_1d(axs), strict=True):
                    self._render_row(
                        ax=ax,
                        ts=ts,
                        sequences=sequences,
                        signal_ylim=signal_ylim,
                        score_ylim=score_ylim,
                        extreme_windows=extreme_windows,
                        training_window=training_window,
                    )

                self._decorate_page(
                    fig=fig,
                    page_idx=signal_page_idx,
                    symbol_list=symbol_list,
                    aux_note_summary=aux_note_summary,
                )
                exporter.add_page(fig, pad_inches=0)
                plt.close(fig)

    def _render_histogram_pages(
        self,
        exporter: PdfExporter,
        *,
        window_sizes: tuple[int, ...] = (10, 50, 100),
        percentile_bounds: tuple[float, float] = (5.0, 95.0),
    ) -> None:
        """Add histogram pages summarizing the R-peak window durations."""
        beats = self.entity.beats
        if beats.size == 0:
            return

        beat_times = beats.astype(np.float64) / self.entity.dataset.sampling_rate_hz
        for window_size in window_sizes:
            if beats.size < window_size:
                continue

            end_times = beat_times[window_size - 1 :]
            start_times = beat_times[: beat_times.size - window_size + 1]
            durations = end_times - start_times
            if durations.size == 0:
                continue

            fig, ax = plt.subplots(figsize=(8, 4))
            plot_histogram(
                ax,
                durations,
                bins="auto",
                title=(
                    f"{self.entity.dataset.name} / "
                    f"{self.entity.entity_id} (k={window_size})"
                ),
                xlabel="Time for R-peak window (sec)",
                ylabel="Count",
                percentile_lines=percentile_bounds,
            )
            fig.tight_layout()
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
        training_window: tuple[float, float] | None,
    ) -> None:
        window_start, window_end = float(ts[0]), float(ts[-1])

        signal_in_window = sequences.signal.slice_between(window_start, window_end)
        beats_in_window = sequences.beats.slice_between(window_start, window_end)
        scores_in_window = sequences.scores.slice_between(window_start, window_end)
        symbols_in_window = sequences.annotations.slice_between(
            window_start, window_end
        )
        aux_notes_in_window = sequences.aux_notes.slice_between(
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

        if sequences.scores.values.size > 0:
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
        plot_aux_notes(
            ax,
            aux_notes_in_window.samples,
            ylim_upper=signal_ylim[1],
        )
        highlight_windows(
            ax,
            extreme_windows,
            window_start=window_start,
            window_end=window_end,
            ylim_upper=signal_ylim[1],
            color=EXTREME_INTERVAL_COLOR,
        )
        if training_window:
            highlight_windows(
                ax,
                [training_window],
                window_start=window_start,
                window_end=window_end,
                ylim_upper=signal_ylim[1],
                color=TRAINING_INTERVAL_COLOR,
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
        aux_note_summary: str,
    ) -> None:
        if page_idx == 0:
            symbol_tokens = "".join(
                symbol
                for symbol in (self._sanitize_text(token) for token in symbol_list)
                if symbol
            )
            dataset_name = self._sanitize_text(self.entity.dataset.name)
            entity_id = self._sanitize_text(self.entity.entity_id)
            fig.suptitle(
                f"{dataset_name}: {entity_id} "
                f"{symbol_tokens} {self.rr_window_beats}\n"
                f"Aux notes: {aux_note_summary}"
            )
        fig.supxlabel("Time (sec)")
        fig.subplots_adjust(left=0.08, right=0.94, bottom=0.05, top=0.95)
