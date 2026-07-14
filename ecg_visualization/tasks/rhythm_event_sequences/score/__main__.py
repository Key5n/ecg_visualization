from ecg_visualization.tasks.rhythm_event_sequences.config import (
    load_rhythm_event_sequences_config,
)
from ecg_visualization.tasks.rhythm_event_sequences.score.score import (
    rhythm_event_sequence_scores,
)

if __name__ == "__main__":
    rhythm_event_sequence_scores(load_rhythm_event_sequences_config())
