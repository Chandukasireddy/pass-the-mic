"""Attendee character agent package for pass-the-mic."""

__all__ = ["CharacterAgent"]


def __getattr__(name):
    if name == "CharacterAgent":
        from .agent_builder import CharacterAgent
        globals()[name] = CharacterAgent
        return CharacterAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
