from ecg_visualization.visualization.text import sanitize_annotation_text


def test_sanitize_annotation_text_removes_control_characters() -> None:
    assert sanitize_annotation_text("\x01 Aux\x00") == "Aux"


def test_sanitize_annotation_text_preserves_printable_rhythm_label() -> None:
    assert sanitize_annotation_text(" (AFIB ") == "(AFIB"
