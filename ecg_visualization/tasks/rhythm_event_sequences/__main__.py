from ecg_visualization.tasks.rhythm_event_sequences.config import (
    load_rhythm_event_sequences_config,
)
from ecg_visualization.tasks.rhythm_event_sequences.score.score import (
    rhythm_event_sequence_scores,
)
from ecg_visualization.tasks.rhythm_event_sequences.visualize.visualize import (
    rhythm_event_sequence_visualize,
)

if __name__ == "__main__":
    config = load_rhythm_event_sequences_config()
    rhythm_event_sequence_scores(config)
    rhythm_event_sequence_visualize(config)
