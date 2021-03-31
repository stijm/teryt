""" Utilities. """

# This is the part of *teryt* library.
# Author: Stim (stijm), 2021
# License: MIT

import re

from functools import wraps
from typing import (Any, Sequence, Union)

import pandas as pd


def require(logic, error, default=ValueError) -> Any:
    """
    If: logic: is not a positive logical value,
    raise :error: or the default error using :error:
    as an argument.
    """
    if not logic:
        if issubclass(error.__class__, BaseException):
            raise error
        raise default(error)
    return logic


def ensure_column(item, frame) -> pd.Series:
    """
    If :item: is not a :frame: column, create it and return it.
    Otherwise, return :item: as a :frame: column.
    """
    return (lambda: frame[str(item)],
            lambda: item)[isinstance(item, pd.Series)]()


def result_of(priority) -> type(lambda: None):
    """
    Precede the function to be decorated with another function,
    e.g. to check the arguments given to the function.
    """
    def decorator(func):

        @wraps(func)
        def prior(self, *args, **kwargs):
            priority(self, args, kwargs)
            result = func(self, *args, **kwargs)
            return result

        return prior
    return decorator


class StringCaseFoldTuple(tuple):
    """
    A tuple that behaves as if reduced to set and converted to tuple
    consisting of casefolded strings only.

    It cannot contain a non-string value.
    """
    def casefold(self):
        return tuple(set(map(str.casefold, self)))

    def __len__(self):
        return len(self.casefold())

    def count(self, __value: Any) -> int:
        return self.casefold().count(__value)

    def __contains__(self, item):
        return item.casefold() in self.casefold()


class StrSearch:
    """
    String DataFrame search broker, more understandable.
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

    def get_method(self, name, default=None):
        if hasattr(self, name):
            return object.__getattribute__(self, name)
        return default

    def name(
            self,
            *,
            root: (pd.Series, str),
            value: str,
            case: bool
    ):
        col = ensure_column(root, self.frame)
        return self.frame.loc[
                    (col == value)
                    if case or not isinstance(value, str)
                    else
                    (col.str.lower() == value.lower())
                    ]

    equal = name

    def match(
            self,
            *,
            root: (pd.Series, str),
            value: str,
            case: bool
    ):
        col = ensure_column(root, self.frame)
        return self.frame.loc[
                    (col.str.match(value, case=case))
                    ]

    def contains(
            self,
            *,
            root: (pd.Series, str),
            value: str,
            case: bool
    ):
        value = re.escape(str(value))
        col = ensure_column(root, self.frame)
        return self.frame.loc[
                   (col.str.contains(value, case=case, na=False))
                   ]

    def startswith(
            self,
            *,
            root: (pd.Series, str),
            value: str,
            case: bool
    ):
        col = ensure_column(root, self.frame)
        return self.frame.loc[
                (col.str.contains(value, na=False))
                if case else
                (col.str.lower().str.startswith(value.lower()))
                ]

    def endswith(
            self,
            *,
            root: (pd.Series, str),
            value: str,
            case: bool
    ):
        col = ensure_column(root, self.frame)
        return self.frame.loc[
                  (col.str.endswith(value, na=False))
                  if case else
                  (col.str.lower().str.endswith(value.lower(), na=False))
                  ]


class DisinheritanceError(KeyError):
    """ Disinheritance error. """


def disinherit(parent: type, klasses: Union[type, Sequence[type]]):
    """
    Stop classes (or one class) inheriting from a :parent: class.

    Parameters
    ----------
    parent : type
        Parent class to remove from :klasses:' bases.

    klasses : type or Sequence[type]
        Classes to disinherit.

    Raises
    ------
    KeyError, if any class from :klasses: does not inherit from :parent:.
    """
    if isinstance(klasses, type):
        klasses = [klasses]
    for klass in set(klasses):
        bases = list(klass.__bases__)
        if parent not in bases:
            raise DisinheritanceError(
                f"{klass.__name__} class does "
                f"not inherit from {parent.__name__}")
        bases.remove(parent)
        klass.__bases__ = tuple(bases)
