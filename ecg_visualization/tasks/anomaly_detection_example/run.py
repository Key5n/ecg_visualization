from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import structlog
from matplotlib.axes import Axes
from sklearn.preprocessing import StandardScaler

from ecg_visualization.datasets.physionet import load_data_sources
from ecg_visualization.logging.config import configure_root_logging
from ecg_visualization.models.md_rs.md_rs import MDRS
from ecg_visualization.models.tsad.esn import tsad_esn
from ecg_visualization.models.tsad.md import tsad_md
from ecg_visualization.tasks.anomaly_detection_example.config import (
    AnomalyDetectionExampleConfig,
    ExampleRecordConfig,
)
from ecg_visualization.tasks.rhythm_event_sequences.config import SegmentWindow
from ecg_visualization.tasks.rhythm_event_sequences.utils import (
    ConcatenatedSequence,
    SequenceSelectionFailure,
    select_sequence_result,
)
from ecg_visualization.utils.utils import sliding_window_sequences
from ecg_visualization.visualization.styles import apply_default_style

LOGGER = structlog.get_logger(__name__)
SEGMENT_KEYS = ("TS", "PA", "AR", "RS")
SEGMENT_ATTRIBUTES = {"TS": "train", "PA": "pre_ar", "AR": "ar", "RS": "test"}
SEGMENT_COLOR_KEYS = {
    "TS": "sinus_train",
    "PA": "pre_ar",
    "AR": "ar",
    "RS": "sinus_test",
}


@dataclass(frozen=True, slots=True)
class ExampleData:
    dataset_id: str
    entity_id: str
    concat: ConcatenatedSequence
    rri_segments: dict[str, npt.NDArray[np.float64] | None]
    score_times: dict[str, npt.NDArray[np.float64] | None]
    segment_ranges: dict[str, tuple[float, float]]


def anomaly_detection_example(config: AnomalyDetectionExampleConfig) -> None:
    """Render ECG, MD, ESN, and MD_RS rows for one record per dataset."""
    configure_root_logging()
    apply_default_style()
    if config.window_size < 1:
        raise ValueError("window_size must be positive")
    if len(config.records) != 2:
        raise ValueError("records must contain exactly one LTAFDB and one SDDB record")

    examples = [_load_example(record, config) for record in config.records]
    figure = _plot_examples(examples, config)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(config.output_path, bbox_inches="tight")
    plt.close(figure)
    LOGGER.info("anomaly_detection_example_saved", output_path=str(config.output_path))


def _load_example(
    record: ExampleRecordConfig, config: AnomalyDetectionExampleConfig
) -> ExampleData:
    dataset = load_data_sources((record.dataset_id,))[0]
    entity = dataset.get_entity(entity_id=record.entity_id)
    return build_example_data(entity, record.pre_ar_duration_sec, config)


def build_example_data(
    entity, pre_ar_duration_sec: float, config: AnomalyDetectionExampleConfig
) -> ExampleData:
    """Build aligned ECG and windowed-RRI inputs for one entity."""
    selection = select_sequence_result(
        entity,
        pre_ar_duration_sec=pre_ar_duration_sec,
        sinus_extraction_config=config.sinus_extraction,
    )
    if isinstance(selection, SequenceSelectionFailure):
        raise ValueError(f"Could not select {entity}: {selection.failure_reason}")

    raw_segments: dict[str, npt.NDArray[np.float64] | None] = {}
    score_times: dict[str, npt.NDArray[np.float64] | None] = {}
    segment_ranges: dict[str, tuple[float, float]] = {}
    offset = 0.0
    for key in SEGMENT_KEYS:
        window = getattr(selection.segments_info, SEGMENT_ATTRIBUTES[key])
        if window is None:
            raw_segments[key] = None
            score_times[key] = None
            continue
        windows, local_times = _windowed_rr(entity, window, config.window_size)
        raw_segments[key] = windows
        duration = window.end_sec - window.start_sec
        score_times[key] = local_times + offset
        segment_ranges[key] = (offset, offset + duration)
        offset += duration

    train = raw_segments["TS"]
    if train is None:
        raise ValueError(f"{entity} has no training segment")
    minimum_train_windows = max(2, config.mdrs.trans_length + 1)
    if len(train) < minimum_train_windows:
        raise ValueError(
            f"{entity} training segment contains {len(train)} RR windows; "
            f"at least {minimum_train_windows} are required"
        )
    scaler = StandardScaler().fit(train)
    scaled_segments = {
        key: None if values is None else scaler.transform(values)
        for key, values in raw_segments.items()
    }
    return ExampleData(
        dataset_id=entity.dataset.dataset_id,
        entity_id=entity.entity_id,
        concat=selection.concat,
        rri_segments=scaled_segments,
        score_times=score_times,
        segment_ranges=segment_ranges,
    )


def _windowed_rr(entity, window: SegmentWindow, window_size: int):
    sampling_rate = float(entity.dataset.sampling_rate_hz)
    beat_times = np.asarray(entity.beats, dtype=np.float64) / sampling_rate
    beats = beat_times[
        (beat_times >= window.start_sec) & (beat_times <= window.end_sec)
    ]
    if beats.size < window_size + 1:
        raise ValueError(
            f"{entity} segment {window.start_sec:g}-{window.end_sec:g}s "
            f"does not contain {window_size + 1} beats"
        )
    rr_intervals = np.diff(beats)
    return (
        sliding_window_sequences(rr_intervals, window_size),
        beats[window_size:] - window.start_sec,
    )


def _mdrs_scores(example: ExampleData, config: AnomalyDetectionExampleConfig):
    model = MDRS(config.mdrs)
    train = example.rri_segments["TS"]
    assert train is not None
    model.train(train)
    scores = {key: None for key in SEGMENT_KEYS}
    model.reset_states()
    scores["TS"] = model.predict(train)
    for key in ("PA", "AR", "RS"):
        values = example.rri_segments[key]
        if values is not None:
            # Each segment originates from a non-contiguous portion of the ECG.
            model.reset_states()
            scores[key] = model.predict(values)
    threshold = (
        np.nanmax(scores["TS"][config.mdrs.trans_length :]) * config.threshold_scale
    )
    return {"scores": scores, "threshold": threshold}


def _plot_examples(examples: list[ExampleData], config: AnomalyDetectionExampleConfig):
    fig, axes = plt.subplots(4, 2, sharex="col", figsize=(15, 8), squeeze=False)
    for row in range(1, axes.shape[0]):
        axes[row, 1].sharey(axes[row, 0])
    for column, example in enumerate(examples):
        results = score_example(example, config)
        _plot_ecg(axes[0, column], example, config)
        for row, (label, result) in enumerate(results, start=1):
            _plot_scores(axes[row, column], example, result, label, config)
        axes[0, column].set_title(
            f"({chr(ord('a') + column)}) {example.dataset_id.upper()} — {example.entity_id}"
        )
        axes[-1, column].set_xlabel("Time (sec)")
    fig.tight_layout()
    return fig


def score_example(example: ExampleData, config: AnomalyDetectionExampleConfig):
    """Score one prepared entity with all three anomaly detectors."""
    return (
        (
            "MD",
            tsad_md(example.rri_segments, threshold_scale=config.threshold_scale),
        ),
        (
            "ESN",
            tsad_esn(
                example.rri_segments,
                n_reservoir=config.mdrs.N_x,
                threshold_scale=config.threshold_scale,
                seed=config.mdrs.seed,
                spectral_radius=config.mdrs.rho,
                input_scale=config.mdrs.input_scale,
                leak_rate=config.mdrs.leaking_rate,
                density=config.mdrs.density,
            ),
        ),
        ("MD_RS", _mdrs_scores(example, config)),
    )


def plot_entity_scores(
    example: ExampleData, config: AnomalyDetectionExampleConfig
) -> plt.Figure:
    """Plot one entity as an ECG row followed by the three score rows."""
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(15, 8), squeeze=False)
    column = axes[:, 0]
    _plot_ecg(column[0], example, config)
    for row, (label, result) in enumerate(score_example(example, config), start=1):
        _plot_scores(column[row], example, result, label, config)
    column[0].set_title(f"{example.dataset_id.upper()} — {example.entity_id}")
    column[-1].set_xlabel("Time (sec)")
    fig.tight_layout()
    return fig


def _plot_ecg(
    ax: Axes, example: ExampleData, config: AnomalyDetectionExampleConfig
) -> None:
    times = np.arange(example.concat.samples.size) / example.concat.sampling_rate_hz
    ax.plot(times, example.concat.samples, color="black", linewidth=0.45)
    ax.set_ylabel("Voltage\n(mV)")
    _shade_segments(ax, example, config.segment_colors)


def _plot_scores(
    ax: Axes,
    example: ExampleData,
    result,
    label: str,
    config: AnomalyDetectionExampleConfig,
) -> None:
    for key in SEGMENT_KEYS:
        times = example.score_times[key]
        scores = result["scores"].get(key)
        if times is not None and scores is not None:
            ax.plot(times, scores, color="black", linewidth=0.75)
    ax.set_ylabel(f"{label}\nScore")
    ax.set_yscale("symlog", linthresh=1e-6)
    _shade_segments(ax, example, config.segment_colors)


def _shade_segments(
    ax: Axes, example: ExampleData, segment_colors: dict[str, str]
) -> None:
    for key, (start, end) in example.segment_ranges.items():
        ax.axvspan(
            start,
            end,
            color=segment_colors[SEGMENT_COLOR_KEYS[key]],
            alpha=0.09,
            linewidth=0,
        )
    ax.set_xlim(0.0, example.concat.samples.size / example.concat.sampling_rate_hz)
