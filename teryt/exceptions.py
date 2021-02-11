""" Exceptions. """

# This is the part of *teryt* library.
# Author: Stim (stijm), 2021
# License: GNU GPLv3


class Error(Exception):
    """ An error. """


class MissingResourcesError(Error):
    """ Missing resources error. """


class ErroneousUnitName(Error):
    """ Erroneous unit name error. """


class UnpackError(Error):
    """ Unpack error. """


class EntryNotFoundError(Error):
    """ Entry not found – error. """


class UnitNotFound(EntryNotFoundError):
    """ Unit not found – error. """


class LocalityNotFound(EntryNotFoundError):
    """ Locality not found – error. """


class StreetNotFound(EntryNotFoundError):
    """ Street not found – error. """
