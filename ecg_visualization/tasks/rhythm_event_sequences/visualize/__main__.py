from ecg_visualization.tasks.rhythm_event_sequences.config import (
    load_rhythm_event_sequences_config,
)
from ecg_visualization.tasks.rhythm_event_sequences.visualize.visualize import (
    rhythm_event_sequence_visualize,
)

if __name__ == "__main__":
    rhythm_event_sequence_visualize(load_rhythm_event_sequences_config())
