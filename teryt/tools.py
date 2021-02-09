""" Utilities. """

# This is the part of teryt library.
# Author: Stim (stijm), 2021
# License: GNU GPLv3

from pandas import (
    Series,
    DataFrame  # noqa: F401
)
from typing import Any
from re import escape
from functools import wraps
from .exceptions import (
    Error
)


def require(logic, error) -> Any:
    """
    If: logic: is not a positive logical value,
    raise :error: or the default error using :error:
    as an argument.
    """
    if not logic:
        if issubclass(error.__class__, BaseException):
            raise error
        raise Error(error)
    return logic


def ensure_column(item, frame) -> Series:
    """
    If :item: is not a :frame: column, create it and return it.
    Otherwise, return :item: as a :frame: column.
    """
    # Equivalent to:
    # ```
    # if isinstance(item, Series):
    #     return item
    # return frame[str(item)]
    # ```
    return (lambda: frame[str(item)],  # False → index 0
            lambda: item  # True → index 1
            )[isinstance(item, Series)]()


def set_sentinel(bound_priority_method) -> type(lambda: None):
    """
    Precede the function to be decorated with another function,
    e.g. to check the arguments given to the function.
    """
    def wrapper(sub):
        @wraps(sub)
        def inner_wrapper(self, *args, **kwargs):
            bound_priority_method(self, args, kwargs)
            return sub(self, *args, **kwargs)

        return inner_wrapper
    return wrapper


class StringCaseFoldedSetLikeTuple(tuple):
    """
    A tuple that behaves as if reduced to set consisting only of
    casefolded values.
    """
    def casefold(self):
        return tuple(set(map(str.casefold, self)))

    def __len__(self):
        return len(self.casefold())

    def count(self, __value: Any) -> int:
        return self.casefold().count(__value)

    def __contains__(self, item):
        return item.casefold() in self.casefold()


class FrameQuestioner(object):
    """
    Search broker for more understandable
    DataFrame searching.
    """

    def __init__(self, frame):
        """
        Constructor.

        Parameters
        ----------
        frame : DataFrame
            DataFrame to search in.
        """
        self.frame = frame

    def name(
            self,
            *,
            col: (Series, str),
            value: str,
            case: bool
    ):
        col = ensure_column(col, self.frame)
        return self.frame.loc[
            (col == value) if case or not isinstance(col, str)
            else (col.str.lower() == value.lower())
        ]

    equal = name

    def match(
            self,
            *,
            col: (Series, str),
            value: str,
            case: bool
    ):
        col = ensure_column(col, self.frame)
        return self.frame.loc[
            (col.str.match(value, case=case))
        ]

    def contains(
            self,
            *,
            col: (Series, str),
            value: str,
            case: bool
    ):
        value = escape(str(value))
        col = ensure_column(col, self.frame)
        return self.frame.loc[
            (col.str.contains(value, case=case, na=False))
        ]

    def startswith(
            self,
            *,
            col: (Series, str),
            value: str,
            case: bool
    ):
        col = ensure_column(col, self.frame)
        return self.frame.loc[
            (col.str.startswith(value, na=False))
            if case else
            (col.str.lower().str.startswith(value.lower()))
        ]

    def endswith(
            self,
            *,
            col: (Series, str),
            value: str,
            case: bool
    ):
        col = ensure_column(col, self.frame)
        return self.frame.loc[
            (col.str.endswith(value, na=False))
            if case else
            (col.str.lower().str.endswith(value.lower(), na=False))
        ]
