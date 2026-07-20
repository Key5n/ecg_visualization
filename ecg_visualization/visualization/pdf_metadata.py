from __future__ import annotations

import json
from typing import Any, Mapping

from ecg_visualization.core.entity import ECGEntity
from ecg_visualization.utils.optuna_record import Record


def build_pdf_metadata(
    *,
    entity: ECGEntity,
    record: Record,
) -> dict[str, str]:
    """Construct a PdfPages-compatible metadata dictionary."""

    metadata: dict[str, str] = {
        "Title": f"{entity.dataset.name} ({entity.entity_id})",
        "DatasetID": entity.dataset.dataset_id,
        "DatasetName": entity.dataset.name,
        "EntityID": entity.entity_id,
        "TrialNumber": str(record.trial_number),
        "TrialState": record.state.name,
    }
    if record.params:
        metadata["TrialParams"] = json.dumps(
            _stringify_mapping(record.params),
            sort_keys=True,
        )
    return metadata


def _stringify_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        normalized[str(key)] = value
    return normalized
