"""
Contains custom exceptions for the NZGMDB.
"""


class SensitivityRemovalError(Exception):
    """Exception raised when sensitivity removal fails."""

    pass


class InventoryNotFoundError(Exception):
    """Exception raised when the inventory information is not found."""

    pass


class NoPWaveFoundError(Exception):
    """Exception raised when no P-wave is found in the phase arrival table."""

    pass


class TPNotInWaveformError(Exception):
    """Exception raised when the TP is not in the waveform bounds."""

    pass
