"""
Contains custom exceptions for the NZGMDB.
"""


class SensitivityRemovalError(Exception):
    """Exception raised when sensitivity removal fails."""

    pass


class InventoryNotFoundError(Exception):
    """Exception raised when the inventory information is not found."""

    pass


class RotationError(Exception):
    """Exception raised when the rotation fails."""

    pass


class All3ComponentsNotPresentError(Exception):
    """Exception raised when all 3 components are not present in the mseed file."""

    pass


class NoPWaveFoundError(Exception):
    """Exception raised when no P-wave is found in the phase arrival table."""

    pass


class TPNotInWaveformError(Exception):
    """Exception raised when the TP is not in the waveform bounds."""

    pass


class LowcutHighcutError(Exception):
    """Exception raised when the lowcut is greater than the highcut."""

    pass


class ComponentSelectionError(Exception):
    """Exception raised when the component selection is invalid."""

    pass


class InvalidNumberOfGeonetPicksException(Exception):
    """
    Exception raised if more than two phase picks
    from Geonet match a given mseed file as there should
    only be one P phase pick and sometimes one S phase pick.
    """

    pass
