def sanitize_annotation_text(value: object) -> str:
    """Return annotation text that is safe to render in visualizations."""
    return "".join(
        character for character in str(value) if character.isprintable()
    ).strip()
