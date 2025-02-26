"""
Contains custom exceptions for the NZGMDB.
"""


class EventIDNotFoundError(Exception):
    """Exception raised when the event ID is not found."""

    pass


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


class LowcutHighcutError(Exception):
    """Exception raised when the lowcut is greater than the highcut."""

    pass


class DigitalFilterError(Exception):
    """Exception raised when the digital filter fails."""

    pass


class ComponentSelectionError(Exception):
    """Exception raised when the component selection is invalid."""

    pass


class InvalidTraceLengthError(Exception):
    """Exception raised when the trace length is invalid for an mseed file."""

    pass


class NoStationsError(Exception):
    """Exception raised when no stations are computed for a given earthquake."""

    pass
