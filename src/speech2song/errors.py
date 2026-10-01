"""Exceptions that the CLI reports as plain messages instead of tracebacks."""


class S2SError(Exception):
    """A user-facing error: bad input, missing run, invalid preset, failed tool, ..."""


class ConfigError(S2SError):
    """Invalid config.yaml or preset file."""


class RunNotFoundError(S2SError):
    """A --run reference matched no run (or more than one)."""


class StageError(S2SError):
    """A stage could not run or did not produce its declared outputs."""


class SpendDeclinedError(S2SError):
    """A paid call was not confirmed (declined, or no TTY and no --yes)."""
