# pylint: disable=too-many-lines

""" The TERYT systems. """

# This is the part of *teryt* library.
# Author: Stim (stijm), 2021
# License: MIT


import abc
import concurrent.futures
import dataclasses
import itertools
import re
import warnings
from typing import Any, Callable, final, Hashable, Iterable, List, Union, Type

import pandas as pd

from simstring.feature_extractor.character_ngram import (
    CharacterNgramFeatureExtractor)
from simstring.measure.cosine import CosineMeasure
from simstring.database.dict import DictDatabase
from simstring.searcher import Searcher

from teryt.data import NA_FILLCHAR
from teryt.data.implement import (
    function_dict,
    inject_master,
    inject_slaves
)
from teryt.data.manager import simc_data, terc_data, ulic_data
from teryt.exceptions import (
    ErroneousUnitName, Error, LocalityNotFound,
    StreetNotFound, UnitNotFound, UnpackError
)
from teryt.tools import (
    disinherit,
    require,
    result_of,
    StringCaseFoldTuple,
    StrSearch
)

systems = StringCaseFoldTuple(('simc', 'terc', 'ulic'))
transfer_collector = {}


class _Missing:  # pylint: disable=too-few-public-methods
    """ Marker-class for missing information. """
    __call__ = None


MISSING = _Missing()


# -----------------------------------------------------------------------------
# Transferred searches
# -----------------------------------------------------------------------------


def transferred_searches(key):
    """ Access transferred searches by the key passed to the transfer. """
    for transferred_search in set(transfer_collector.get(key, ())):
        name = getattr(transferred_search, 'system')

        if isinstance(name, System):
            name = name.system

        yield name, transferred_search


# -----------------------------------------------------------------------------
# Semantic links and link manager classes
# -----------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class SMLink:
    """ TERYT semantic link. """

    code: str
    value: Any

    def __getitem__(self, item: (str, int)):
        choices = ({item: getattr(self, item, "")}, [*dict(self).values()])
        return choices[isinstance(item, int)][item]

    def __str__(self):
        return self.code or ""

    def __add__(self, other):
        return self.code.__add__(other)

    def __bool__(self):
        return all([self.value, self.code])

    def __iter__(self):
        if self.value:
            yield "value", self.value

        yield "code", self.code

        index_ = getattr(self, "index", MISSING)

        if index_ is not MISSING:
            yield "index", index_


@dataclasses.dataclass(frozen=True)
class UnitLink(SMLink):
    """ Unit link. """
    index: int

    @property
    def as_unit(self):
        """ Return the unit object that this link leads to. """
        if self.index or self.index == 0:
            with terc() as unit_mgr:
                return unit_mgr.index(self.index)

    as_entry = as_unit


@dataclasses.dataclass(frozen=True)
class LocalityLink(SMLink):
    """ Locality link. """
    index: int

    @property
    def as_loc(self):
        """ Return the locality object that this link leads to. """
        if self.index or self.index == 0:
            with simc() as loc_mgr:
                return loc_mgr.index(self.index)

    as_entry = as_loc


class _LinkManager:
    def __init__(self):
        self.frame_linked_args = {}
        self.other_args = {}

    @staticmethod
    def link(gl_mgr, arguments, _keywords):
        if arguments:
            field = next(iter(arguments[:]))
            if field != 'integral':
                require(
                    gl_mgr.has_link_mgr(field),
                    f'field {field!r} '
                    f'cannot be linked'
                )

    def link_names(self, gl_mgr, _arguments, keywords):
        expected = sorted(gl_mgr.system.fields.keys())

        for keyword in list(set(keywords)):
            if keyword in expected:
                err = gl_mgr.erroneous_argument
                TypeError(err % ("link_names", keyword, ', '.join(expected)))

        for name, value in keywords.items():
            if gl_mgr.has_frame_link_mgr(name):
                self.frame_linked_args.update({name: value})
                continue

            if gl_mgr.has_dict_link_mgr(name):
                link_mgr = gl_mgr.get(name, MISSING)

                if link_mgr is not MISSING:
                    if value.isnumeric() and value in link_mgr.values():
                        value = dict(
                            [*map(reversed, link_mgr.items())])[value]

                    if value not in link_mgr:
                        err = (
                            f"{value!r} is not a valid "
                            f"{name.replace('_', ' ')!r} non-ID value. "
                        )

                        raise ValueError(err)

                    value = link_mgr[value]
            self.other_args.update({name: value})


@dataclasses.dataclass(init=False)
class LinkManager:
    system: "System"
    dict_link_mgrs: "DictLinkManagers"
    frame_link_mgrs: "FrameLinkManagers"

    _backend = _LinkManager()

    def __init__(self, system, dict_link_mgrs, frame_link_mgrs):
        self.system = system
        self.dict_link_mgrs = dict_link_mgrs
        self.frame_link_mgrs = frame_link_mgrs
        self.lm_cache = self.system.lm_cache[self.system.system].update
        self.cache = self.system.cache[self.system.system].update
        self.indexes = {}

    @result_of(_backend.link_names)
    def link_names(self, **_keywords):
        frame_linked_args = self._backend.frame_linked_args
        other_args = self._backend.other_args
        extract = other_args.copy()

        for field in other_args.keys():
            if field not in self.system.terc.columns:
                extract.pop(field)

        fields = list(self.system.fields.keys())
        for field, value in frame_linked_args.items():
            # TODO: dict with all fields-preparation mappers
            if field == 'voivodship' and isinstance(value, str):
                value = value.upper()

            frame_link_mgr: pd.DataFrame = self.frame_link_mgrs.get(field)

            if frame_link_mgr.empty:
                warnings.warn(
                    f"no link managers available for {field + 's'}. "
                    "Search keywords are now updated with the provided value, "
                    "however results are possible not to be found if it "
                    "is not a valid ID."
                )
                other_args.update({field: value})
                continue

            if value.isnumeric() and value in frame_link_mgr.values:
                value = self.link(field, value)
                other_args.update({field: value})

            entry = Search(
                database=frame_link_mgr,
                system="terc",
                method="equal",
                fields=terc.fields,
                case=False,
                str_contains=terc.str_contains,
                str_startswith=terc.str_startswith,
                str_eq=value
            )(keywords=other_args)

            if entry.empty:
                case_fmt: Callable = name_case_descriptors[field]
                chunk = case_fmt(value)
                error = f"{chunk!r} is not a {field}"
                most_similar = terc().most_similar(
                    value, inplace=False).lstrip("(").rstrip(")").split("|")

                if most_similar != value:
                    if most_similar[0] in self.frame_link_mgrs.get(
                            field)[self.system.fields["name"]]:
                        error += (". Did you mean "
                                  f"{case_fmt(most_similar[0])!r}?")

                raise ErroneousUnitName(error)

            self.indexes[field] = entry.iat[0, 0]
            link_result = {field: entry.iat[0, entry.columns.get_loc(
                self.system.fields[field]
            )]}
            self.system.entry_helper.update(link_result)
            other_args.update(link_result)

            if field != fields[0]:
                for rotation in range(fields.index(field)):
                    prev = fields[fields.index(field) - 1 - rotation]
                    other_args[prev] = entry.iat[
                        0, entry.columns.get_loc(self.system.fields[prev])
                    ]

        return dict(**extract, **other_args)

    def has_dict_link_mgr(self, field: str) -> "bool":
        """ Check if a field has a dict link manager. """
        return hasattr(self.dict_link_mgrs, field + '_link_mgr')

    def has_frame_link_mgr(self, field: str) -> "bool":
        """ Check if a field has a frame link manager. """
        return hasattr(self.frame_link_mgrs, field + 's')

    def has_link_mgr(self, field: str) -> "bool":
        """ Check if a field has any link manager. """
        return (self.has_dict_link_mgr(field) or
                self.has_frame_link_mgr(field))

    @result_of(_backend.link)
    def link(self, field: str, value: str):
        """
        Resolve entry that value in field refers to,
        creating and returning a Link instance.

        Parameters
        ----------
        field : str
            field of :value:.

        value : value
            Value to link.

        Returns
        -------
        str
        """
        if self.has_dict_link_mgr(field):
            return dict.__call__(
                map(reversed, self.dict_link_mgrs.get(field).items()))[value]

        unit_mgr = terc()

        if field not in self.system.link_fields or str(value) == NA_FILLCHAR:
            return ""

        keywords = {'function': function_dict[self.system.fields[field]]}
        fields = list(self.system.fields.keys())
        helper = self.system.entry_helper

        if field != fields[0]:
            quantum = fields.index(field) - 1

            for rotation in range(quantum + 1):
                prev_value = fields[quantum - rotation]

                if prev_value not in helper:
                    raise ValueError(f"cannot link {field!r} as "
                                     f"its overriding field {prev_value!r} "
                                     f"is not declared")

                keywords[prev_value] = str(helper[prev_value])

        keywords[field] = value

        if field != fields[-1]:
            next_value = fields[fields.index(field) + 1]
            keywords[next_value] = NA_FILLCHAR

        hashable_keywords = tuple(keywords.items())

        cache = self.system.cache[self.system.system]
        lm_cache = self.system.lm_cache[self.system.system]

        if hashable_keywords in cache:
            cached_name = cache[hashable_keywords]
            cached_index = lm_cache[hashable_keywords]
            self.indexes[field] = cached_index
            return cached_name

        result = Search(
            database=unit_mgr.database,
            system=unit_mgr.system,
            method='no_name',
            fields=unit_mgr.fields,
            case=False,
            str_contains=unit_mgr.str_contains,
            str_startswith=unit_mgr.str_startswith,
        )(keywords=keywords).reset_index()

        result.drop(
            columns=["level_0"],
            inplace=True
        )

        name = result.iat[
            0, result.columns.get_loc(unit_mgr.fields["name"])
        ]

        self.indexes[field] = result.iat[0, 0]
        self.lm_cache({hashable_keywords: self.indexes[field]})
        self.cache({hashable_keywords: name})
        return name

    def get(self, item, default=None):
        if self.has_dict_link_mgr(item + "_link_mgr"):
            return self.dict_link_mgrs.get(item)

        if self.has_frame_link_mgr(item + "s"):
            return self.frame_link_mgrs.get(item)

        return default

    def __getattribute__(self, item):
        try:
            got_attr = object.__getattribute__(self, item)

        except AttributeError:
            _got_attr = self.get(item, MISSING)
            if _got_attr is MISSING:
                raise
            got_attr = _got_attr

        return got_attr


class DictLinkManagers:
    """ Internal helper class. """
    def __init__(self, register):
        for attr in filter(
                lambda s: s.endswith("_link_mgr"), dir(register)):
            setattr(self, attr, getattr(register, attr))

    def get(self, field, default=None):
        if not field.endswith("_link_mgr"):
            field += "_link_mgr"
        if not hasattr(self, field):
            return default
        return object.__getattribute__(self, field)


class FrameLinkManagers:
    """ Internal helper class. """
    def __init__(self):
        params = dict(unpack=False, unpacked=True)
        self._m = terc(link=False)
        self.voivodships = self._m.search(function='województwo', **params)
        self.powiats = self._m.search(function='powiat', **params)
        self.gminas = self._m.search(function='gmina', **params)

    def get(self, field, default=None):
        if not field.endswith("s"):
            field += "s"

        if not hasattr(self, field):
            return default

        return object.__getattribute__(self, field)

    def __repr__(self):
        return f"FrameLinkManagers({self._m!r})"


# -----------------------------------------------------------------------------
# Search
# -----------------------------------------------------------------------------


class Search:
    def __init__(
            self,
            *,
            database: pd.DataFrame,
            system: str,
            method: str,
            fields: Union[dict, property],
            case: bool,
            str_contains: Iterable,
            str_startswith: Iterable,
            str_eq='',
    ):
        """ Constructor. """
        self.database = database
        self.system = system
        self.method = method
        self.fields = fields
        self._case = case
        self._keywords = {}
        self._buffer = self.database.copy()
        self._frames = [self._buffer]
        self._ineffective_fields = []
        self._str_eq = str_eq
        self._str_contains = str_contains
        self._str_startswith = str_startswith
        self._attempts = 0
        self._max_attempts = 0
        self._done = False

    def _failure(self):
        """ Was anything found? """
        return self._buffer.empty or self._buffer.equals(self.database)

    def _lookup(self):
        for field, query in (*self._keywords.items(),):
            if field == "voivodship":
                query = query.upper()

            self._attempts += 1

            if field not in self.fields:
                continue

            root = self.fields[field]
            keyword_args = dict(
                root=root,
                value=str(query),
                case=self._case
            )
            method = "name"

            if field in self._str_contains:
                method = "contains"
            elif field in self._str_startswith:
                method = "startswith"

            self._buffer = StrSearch(self._buffer).get_method(
                method)(**keyword_args)

            if self._failure():
                if self._buffer.equals(self.database):
                    no_uniqueness = f'It seems that all values in ' \
                                    f'{field!r} field ' \
                                    f'are equal to {query!r}. Try ' \
                                    f'using more unique key words.'
                    warnings.warn(no_uniqueness, category=UserWarning)

                self._buffer = self._frames[-1]

                if self._attempts <= self._max_attempts:
                    ineffective = self._keywords.pop(field)

                    if ineffective not in self._ineffective_fields:
                        # Ineffective keyword goes to the end
                        self._ineffective_fields.append(field)
                        self._keywords[field] = ineffective
                        self._lookup()

                else:
                    self._done = True
                    break

            self._frames.append(self._buffer)

    def _name_lookup(self):
        self._buffer = StrSearch(self._buffer).get_method(self.method)(
            root=self.fields["name"],
            value=self._str_eq,
            case=self._case)

        self._frames.append(self._buffer)
        return self._buffer

    def _search(self, keywords) -> "pd.DataFrame":
        self._keywords = keywords
        self._max_attempts = len(self._keywords)

        if self.method != "no_name":
            self._name_lookup()
            if self._failure():
                return pd.DataFrame()

        if not self._done and self._keywords:
            self._lookup()
        return self._buffer

    def __call__(self, keywords: dict):
        return self._search(keywords=keywords)


# -----------------------------------------------------------------------------
# System classes.
# -----------------------------------------------------------------------------

class _System:
    @staticmethod
    def search(self, arguments, keywords):
        self.__init__(**self.modes)

        if len(arguments) == 1:
            keyword = arguments[0]

            if not isinstance(keyword, str):
                raise TypeError(
                    f"name must be str, not {keyword.__class__.__name__}")
            if "name" in keywords:
                raise ValueError(
                    "passed multiple values for 'name' parameter")

            keywords["name"] = keyword
            arguments = ()

        if arguments:
            raise ValueError("only one positional argument can be accepted")

        if not keywords:
            raise ValueError("no keyword arguments")

        search_keywords = tuple(
            set(self.name_fields + (*self.fields.keys(),))
        ) + self.optional_str_fields

        keywords = dict(map(  # roots -> fields
            lambda kv: (self.ensure_field(kv[0]), kv[1]),
            keywords.items())
        )

        if not any(map(keywords.__contains__, search_keywords)):
            raise ValueError(
                f'no keyword arguments for searching '
                f'(expected at least one from: '
                f'{", ".join(sorted(search_keywords))}'
                f')'
            )

        for conflicted in self.conflicts:
            conflict = []
            conflicted = sorted(conflicted)
            for keyword in conflicted:
                if keyword in keywords:
                    if conflict:
                        raise ValueError(
                            'setting more than one keyword argument '
                            'from %s in one search is impossible' %
                            (' and '.join(map('%s'.__mod__, conflicted))))
                    conflict.append(keyword)

        self.str_eq = None
        self.force_unpack = False
        self.exact = keywords.pop("exact", self.exact)
        modes = self.name_fields + ('no_name',)
        self.method = modes[-1]

        for keyword in keywords.copy():
            for mode in self.name_fields:
                if keyword == mode:
                    self.method = mode
                    self.str_eq = (keywords[mode] if self.exact else
                                   self.most_similar(keywords[mode]))
                    del keywords[mode]

        pop = keywords.pop

        self.raise_for_failure = pop("raise_for_failure",
                                     self.raise_for_failure)
        self.unpack_mode = pop("unpack", self.unpack_mode)
        self.link_mode = pop("link", self.link_mode)
        self.force_unpack = pop("force_unpack", self.force_unpack)
        self.unpacked = pop("unpacked", False)
        self.case = pop("case", self.case)
        terid = pop("terid", "")

        if not self.unpacked and self.link_mode:
            self.link_mgr.erroneous_argument = self.erroneous_argument
            keywords = self.link_mgr.link_names(**keywords)

        if terid:
            unpacked = self.unpack_terid(terid)
            for key, value in unpacked.items():
                if value:
                    keywords[key] = value

        self.keywords = keywords
        self.buffer = self.database[:]

        for field in self.keywords:
            root_name = self.fields.get(field, MISSING)
            # KeyError: 'secname' after using ULIC
            if root_name is MISSING:
                continue
            self.database[root_name] = self.database[
                self.fields[field]
            ].map(str)

    @staticmethod
    def unpack_row(system, args, keywords):
        if len(args) == 1:
            row = args[0]
        else:
            row = keywords.pop("row", MISSING)
        fields = system.fields

        if isinstance(row, pd.Series):
            row = pd.DataFrame([[*row]], columns=row.keys())

        if row is MISSING or row is None:
            raise UnpackError("nothing to unpack from")

        if row.empty:
            raise UnpackError("nothing to unpack from")

        if len(row) != 1:  # it's not a row then
            raise UnpackError(
                "cannot unpack from more "
                "than one TERYT row "
                f"(got {len(row)} rows)"
            )

        for field in fields:
            if fields[field] not in row:
                raise UnpackError(
                    f"field "
                    f"{field.replace('_', ' ')} "
                    f"(root name: "
                    f"{fields[field]!r}) "
                    f"not in source DataFrame"
                )
        system.buffer = row


class System(abc.ABC):
    __slots__ = tuple("""
        case
        exact
        force_unpack
        link_mode
        raise_for_failure
        unpack_mode
        modes
        simc
        terc
        ulic
        system
        database
        root_names
        str_eq
        method
        keywords
        found_results
        unpacked
        linked
        buffer
        _results
        transfer_target
        entry_helper
        store
        link_mgr
        """.split())

    # Cache
    # -----

    cache = {
        "simc": {},
        "terc": {},
        "ulic": {}
    }

    lm_cache = {
        "simc": {},
        "terc": {},
        "ulic": {}
    }

    # Several ways to look up the values
    # ----------------------------------

    str_contains = (
        "function",
    )
    str_startswith = (
        "date",
        # TODO: this should be a `datetime.date` argument, shouldn't it?
    )
    optional_bool_fields = (
        "raise_for_failure",
        "force_unpack",
        "unpack",
        "link",
        "unpacked",
        "case"
    )
    optional_str_fields = (
        "terid",
    )
    optional_bool_str_fields = (
        *optional_bool_fields,
        *optional_str_fields
    )

    # Fields, arguments
    # -----------------

    name_fields = (
        "name",
        "match",
        "startswith",
        "endswith",
        "contains"
    )
    conflicts = (
        name_fields,
        ("force_unpack", "unpack")
    )

    # Helpers
    # -------

    erroneous_argument = (
        f"%s() got an unexpected keyword argument %r. "
        f"Try looking for the proper argument name "
        f"in the following list:\n{' ' * 12}%s."
    )
    _backend = _System()
    _name_searcher = None

    # Initializer
    # -----------

    def __init__(
            self,
            case=False,
            exact=False,
            link=True,
            unpack=True,
            raise_for_failure=False,
    ):
        """
        Constructor.

        Parameters
        ----------
        case : bool
            Whether searched values are case-sensitive.

        exact : bool
            Whether to look for most similar names if a given name
            is not correct.

        link : bool
            Whether to link the values in
            search/(converting to list)/(converting to dict).

        unpack : bool
            Whether to unpack future results/processed rows.

        raise_for_failure : bool
            Whether to raise *NotFound error if no results are found.
        """
        self.__class__.conflicts += tuple(
            map(lambda ls: ('terid', ls), self.link_fields))

        # Modes
        self.case = case
        self.exact = exact
        self.force_unpack = False
        self.link_mode = link
        self.raise_for_failure = raise_for_failure
        self.unpack_mode = unpack
        self.modes = dict(
            case=self.case,
            exact=self.exact,
            link=self.link_mode,
            raise_for_failure=self.raise_for_failure,
            unpack=self.unpack_mode
        )

        # Data
        self.simc = simc_data
        self.terc = terc_data
        self.ulic = ulic_data

        self.system = self.__class__.__name__.replace(' ', '_').casefold()

        if self.system == "System".casefold():
            raise Error("abstract class")
        self.database = getattr(self, self.system, MISSING)

        if self.database is MISSING:
            raise ValueError(f"invalid system {self.system!r}")

        self.database = self.database.reset_index()

        # Root names
        self.root_names = [*self.database.columns]

        # Searching
        self.str_eq = None
        self.method = None
        self.keywords = {}
        self.found_results = False
        self.unpacked = False
        self.linked = False
        self.buffer = pd.DataFrame()
        self._results = EntryGroup(self)

        # Transferring
        self.transfer_target = None

        # Building links
        self.entry_helper = {}

        # Caching
        self.store = self.__class__.cache[self.system].update

        if link:
            self.link_mgr = LinkManager(
                dict_link_mgrs=DictLinkManagers(self),
                frame_link_mgrs=System.frame_link_mgrs,
                system=self
            )

    # System as a context manager
    # ---------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            raise

    # Other dunder methods
    # --------------------

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            return self.multi_unpack(
                self.database.iloc[item], link=self.link_mode)

        return self.to_list(item, link=self.link_mode)

    def __iter__(self):
        if self.entry_helper["terid"]:
            yield "terid", self.entry_helper["terid"]

        for field in self.fields:
            if self.entry_helper.get(field, ""):
                yield field, self.entry_helper.get(field, "")

    def __len__(self):
        return len(self.database)

    def __repr__(self):
        # F"{value=}" rule was ommitted in order to avoid
        # SyntaxError in older versions.
        init_args = ", ".join(map(str, itertools.compress(
            ["link=" + repr(self.link_mode),
             "unpack=" + repr(self.unpack_mode),
             "force_unpack=" + repr(self.force_unpack),
             "raise_for_failure=" + repr(self.raise_for_failure)],
            [not self.link_mode,
             not self.unpack_mode,
             bool(self.force_unpack),
             bool(self.raise_for_failure)]
        )))

        return (f"{(__name__ + '.', '')[__name__ == '__main__']}"
                f"{self.system.upper()}(" +
                init_args + ")" +
                (f"\nResults:\n{self.results}"
                 if self.found_results else ""))

    # Private System methods
    # ----------------------

    def _dispatcher(self):
        buffer = self.buffer.reset_index()

        if self._failure():
            not_found_err = error_types[self.system]
            if self.raise_for_failure:
                raise not_found_err("no results found")
            self.__init__(**self.modes)

        else:
            self.found_results = True
            self._results = EntryGroup(self, buffer)
            self._results.frame = self._results.frame.drop(columns=["level_0"])

            if (len(self.r) == 1 or self.force_unpack) and self.unpack_mode:
                return self.unpack_row(self.results)

        return self.results

    @final
    def _failure(self):
        return self.buffer.empty or self.buffer.equals(self.database)

    def _treat_data_chunk(self, field, root):
        code = self.buffer.iat[
            0, self.buffer.columns.get_loc(root)
        ]

        if code != NA_FILLCHAR:
            self.entry_helper[field] = code

            if self.link_mgr.has_link_mgr(field) and self.link_mode:
                value = self.link_mgr.link(field, code)
                entry_index = self.link_mgr.indexes.get(field, MISSING)

                if entry_index is not MISSING:
                    self.entry_helper[field] = UnitLink(
                        code=code, value=value, index=entry_index)

                else:
                    self.entry_helper[field] = SMLink(
                        code=code, value=value)

    # Public System attributes, methods and properties
    # ------------------------------------------------

    @final
    def ensure_field(self, name) -> str:
        """
        Find :name: in fields and return it if it occurs.

        Parameters
        ----------
        name : str
            Name to return field name of.

        Returns
        -------
        str
        """
        if name in self.fields:
            return name

        return self.roots.get(name.upper(), name)

    @property
    def fields(self):
        """ Fields. """
        raise NotImplementedError

    @property
    def roots(self):
        """ Roots. """
        (*items,) = map(reversed, self.fields.items())
        return dict(items)

    def index(self, i, /, *, link=True):
        """
        Return an entry by index.

        Parameters
        ----------
        i : int or slice
            Positional argument. Slice of an entry.

        link : bool
            Whether to link the result entry.

        Returns
        -------
        Entry
        """
        self.buffer = self.database.iloc[i]

        return self.unpack_row(_row=self.buffer, link=link)

    get_entries = index

    @property
    def is_entry(self):
        """
        Check if this instance is of Entry class.

        Returns
        -------
        bool
        """
        return False

    isentry = is_entry
    isunit = is_unit = is_entry
    isloc = is_loc = is_entry
    isstreet = is_street = is_entry

    @property
    def link_fields(self):
        """ Fields to be linked. """
        raise NotImplementedError

    def most_similar(self, name, inplace=True):
        if name.lower() not in names[self.system]:
            rank = self._name_searcher.ranked_search(name, 0.4)

            if not rank:
                return name

            pattern = ("(" + "|".join(
                itertools.islice(map(lambda res: res[1], rank), 3)) + ")")

            if inplace:
                self.method = "match"

            return pattern

        return name

    def pack_terid(self, **info) -> "str":
        """
        Pack information into teritorial ID.

        Parameters
        ----------
        **info
            Keyword arguments consisting of linkable fields.

        Returns
        -------
        str
        """
        return ''.join(map(str, map(
            lambda field: info.get(field, ""),
            filter(lambda x: str(x) != NA_FILLCHAR, self.link_fields)
        )))

    @property
    def results(self):
        """
        Results of the most recent search.

        Returns
        -------
        EntryGroup
        """
        return self._results

    r = res = results

    # Search
    # ------

    @result_of(_backend.search)
    def search(
            self,
            *args,  # noqa, pylint: disable=unused-argument
            **keywords  # noqa, pylint: disable=unused-argument
    ) -> Union["Entry", "EntryGroup"]:
        """
        Search for the most accurate entry using provided keywords.

        Parameters
        ----------
        *args
            Positional arguments. Can only contain name of
            the searched locality, street or unit (all systems).

        **keywords
            Keyword arguments.

        Other Parameters
        ----------------
        secname : str
            Second name of a street (ULIC).

        date : str
            State as of the date (all systems).

        name : str
            Name of the searched locality, street or unit (all systems).

        loctype : str
            Locality type (SIMC).

        gmina : str
            Gmina of the searched locality, street or unit (all systems).

        voivodship : str
            Voivodship of the searched locality, street or unit (all systems).

        function : str
            Unit function (TERC).

        streettype : str
            Street type (ULIC).

        powiat : str
            Voivodship of the searched locality, street or unit (all systems).

        cnowner : bool
            Whether a locality owns a common name (SIMC).

        id : str
            ID of a locality/street (SIMC, ULIC).

        integral_id : str
            Integral ID of a locality/street (SIMC, ULIC).

        gmitype : str
            Gmina type of the searched locality, street or unit (all systems).

        Column names as the above listed arguments are also acceptable.
        It means, you can pass "woj=value" instead of "voivodship=value".

        Examples
        --------
        >>> s = simc()
        >>> s.search("Poznań")
        SIMC()
        Results:
           index WOJ POW GMI  ...   NAZWA      SYM   SYMPOD     STAN_NA
        0  11907  06  11  06  ...  Poznań  0686397  0686397  2021-01-01
        1  76796  24  10  03  ...  Poznań  0217047  0216993  2021-01-01
        2  95778  30  64  01  ...  Poznań  0969400  0969400  2021-01-01

        Returns
        -------
        Union[Entry, EntryGroup]
            Entry, if one most accurate entry was found, otherwise EntryGroup
            if there were many results or not.

        """
        self.buffer = Search(
            database=self.database,
            system=self.system,
            method=self.method,
            fields=self.fields,
            case=self.case,
            str_eq=self.str_eq,
            str_contains=self.str_contains,
            str_startswith=self.str_startswith
        )(keywords=self.keywords)

        return self._dispatcher()

    # Dict exporter
    # -------------

    def to_dict(
            self,
            link: bool = True,
            root_names=False,
            indexes: bool = True
    ) -> "dict":
        """
        Return all values in a system as a dict.

        Parameters
        ----------
        indexes : bool
            Whether to return a dict with indexes.

        root_names : bool
            Whether to leave the database's column names or apply
            the fields names.

        link : bool
            Whether to link the linkable values. Defaults to True.

        Returns
        -------
        dict
        """
        return dict(EntryGroup(self, frame=self.database).to_dict(
            root_names=root_names, indexes=indexes, link=link
        ))

    # List exporter
    # -------------

    def to_list(self, field: str, link: bool = True) -> "list":
        """
        Return list of all values in :field: in the database.

        Parameters
        ----------
        field : str
            field to retrieve values of.

        link : bool
            Whether to link the linkable values. Defaults to True.

        Returns
        -------
        list
        """
        return list(EntryGroup(self, frame=self.database).to_list(
            field=field, link=link
        ))

    tolist = to_list

    @final
    def unique_field(self, field):
        """
        State whether a field is unique in comparison
        to other systems.

        Parameters
        ----------
        field : str
            Name of a field.

        Returns
        -------
        bool
        """
        other_fields = set(filter(
            lambda: not self.database.equals,
            [self.simc, self.terc, self.ulic])
        )

        return all([field in self.fields, field not in other_fields])

    # Unpacking
    # ---------

    @result_of(_backend.unpack_row)
    def unpack_row(
            self,
            _row: Union["EntryGroup", pd.Series, pd.DataFrame] = None,
            *,
            link=True
    ) -> "Entry":
        """
        Unpack one-row DataFrame to Entry instance.

        Parameters
        ----------
        _row : EntryGroup, Series or DataFrame
            Entry group/DataFrame with length 1 or Series to unpack.

        link : bool
            Whether to link the linkable values.

        Returns
        -------
        Entry
        """
        self.link_mode = link

        for chunk in self.fields.items():
            self._treat_data_chunk(*chunk)

        self.entry_helper["terid"] = self.pack_terid(**self.entry_helper)
        self.unpacked = True

        return entry_types[self.system].__call__(
            system=self,
            **self.entry_helper,
            row=self.buffer,
            index=self.buffer.iat[0, 0]
        )

    def multi_unpack(
            self,
            rows: Union["EntryGroup", "pd.DataFrame"] = None,
            *,
            link=True,
            ensure_list=False,
    ) -> Union["Entry", List["Entry"]]:
        """
        Unpack DataFrame entries to Entry instance and return a list of it.

        Parameters
        ----------
        rows : EntryGroup or DataFrame
            Entry group/DataFrame to unpack.

        link : bool
            Whether to link the linkable values.

        ensure_list : bool
            Whether to still return list of entries if there is only one.

        Returns
        -------
        Entry or List[Entry]
        """
        if isinstance(rows, pd.Series):
            rows = pd.DataFrame([[*rows.values]], columns=[*rows.keys()])

        def _no_index_unpack(pair):
            return self.__class__().unpack_row(pair[1], link=link)

        with concurrent.futures.ThreadPoolExecutor() as _multi:
            results = list(_multi.map(_no_index_unpack, rows.iterrows()))

        if len(results) == 1 and not ensure_list:
            return results[0]

        return results

    def unpack_terid(self, teritorial_id: str, errors: bool = True) -> "dict":
        """
        Unpack teritorial ID into information.

        Parameters
        ----------
        teritorial_id : str
            ID to unpack.

        errors : bool
            Whether to raise errors if :teritorial_id: is invalid.

        Returns
        -------
        dict
        """
        if not teritorial_id:
            raise ValueError("cannot unpack an empty teritorial ID string")

        chunks = {}
        frames = {}
        max_length = sum(self.link_fields.values())

        if len(teritorial_id) > max_length:
            raise ValueError(f"{self.system.upper()} teritorial ID length "
                             f"is expected to be at least {max_length}")

        entry_index = 0

        for link_mgr_field, proper_length in self.link_fields.items():

            if entry_index >= len(teritorial_id) - 1:
                break

            frames.update(
                {link_mgr_field: self.link_mgr.get(link_mgr_field)}
            )
            chunk = teritorial_id[entry_index:entry_index + proper_length]
            unpack = self.unpack_mode

            if errors:
                checker = self.__class__().search(
                    unpacked=True, unpack=False,
                    **{link_mgr_field: chunk})
                if checker.results.empty:
                    raise ValueError(
                        repr(chunk) +
                        f"is an incorrect teritorial code chunk "
                        f"(error at {link_mgr_field!r} field, "
                        f"root name "
                        f"{self.fields[link_mgr_field]!r})"
                    )

            self.unpack_mode = unpack
            chunks.update({link_mgr_field: chunk})
            entry_index += proper_length

        return chunks


# Inject the common data
# ----------------------
inject_master(System)


# -----------------------------------------------------------------------------
# EntryGroup classes.
# -----------------------------------------------------------------------------
class _EntryGroup:
    @staticmethod
    def to_keywords(self, arguments, _kwds):
        require(arguments, "to_keywords(): no arguments")
        target_name = arguments[0]

        if target_name in list(map(eval, systems)):
            target_name = target_name.__name__

        if target_name not in systems:
            raise ValueError(
                f"cannot evaluate transfer target using name {target_name!r}")

        self.transfer_target = eval_dict[target_name]()


class EntryGroup:
    """
    Group of entries.

    See also
    --------
    Entry
    """

    _backend = _EntryGroup()

    def __contains__(self, item):
        return self.frame.__contains__(item)

    def __getattribute__(self, item):
        try:
            return object.__getattribute__(self, item)

        except AttributeError:
            return getattr(self.frame, item)

    def __init__(self, system, frame=pd.DataFrame()):
        self.system = system
        self.frame = frame

    def __len__(self):
        return len(self.frame)

    def __repr__(self):
        return repr(self.frame)

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            return self.get_entries(item, link=self.system.link_mode)

        return self.to_list(item, link=self.system.link_mode)

    def get_entries(self, number, link=True, ensure_list=False):
        frame_or_series = self.frame.iloc[number]

        if ensure_list or link:
            return self.system.__class__().multi_unpack(
                frame_or_series, ensure_list=ensure_list, link=link)

        return frame_or_series

    get_entries.__doc__ = System.get_entries.__doc__

    def to_dict(
            self,
            link: bool = True,
            root_names=False,
            indexes: bool = True
    ) -> "dict":
        """
        Return dict of all values in search results.

        Parameters
        ----------
        indexes : bool
            Whether to return a dict with indexes.

        root_names : bool
            Whether to leave the database's column names or apply
            the fields names.

        link : bool
            Whether to link the linkable values. Defaults to True.

        Returns
        -------
        dict
        """
        frame = self.frame.copy()
        new = {}

        for field in self.system.fields:
            value = [*self.to_list(field, link=link)]
            name = (field, self.system.fields[field])[root_names]
            new.update({name: value[0] if len(value) == 1 else value})

        if indexes:
            new["index"] = [*range(0, len(frame))]

        return new

    todict = to_dict

    @result_of(_backend.to_keywords)
    def to_keywords(self, target: Union[str, Type]):  # noqa
        """
        Create and return keywords leading to current search results.

        Parameters
        ----------
        target: str or type
            Target class (SIMC, TERC or ULIC).

        Returns
        -------
        generator
        """
        target = self.transfer_target
        name = {}
        if self.system.str_eq:
            name = dict(match=re.escape(self.system.str_eq))

        yield dict(
            **name,
            **self.system.keywords,
            **self.system.modes
        )
        yield target

    # noinspection PyUnresolvedReferences
    def _form_root(self, root_index):
        """ Internal helper method. """
        entry = self.system.__class__().unpack_row(
            pd.DataFrame([self.frame.loc[self.frame.index[root_index]]]))

        self.__class__._form_root.new_list[
            root_index] = getattr(entry, self.__class__._form_root.field)

    def to_list(self, field: str, link: bool = True) -> "list":
        """
        Return list of all values in :field:.

        Parameters
        ----------
        field : str
            field to retrieve values of.

        link : bool
            Whether to link the linkable values. Defaults to True.

        Examples
        --------
        >>> warsaw = simc().search("Warszawa", gmitype="wiejska", woj="04")
        >>> warsaw
           index WOJ POW GMI  ...     NAZWA      SYM   SYMPOD     STAN_NA
        0   4810  04  14  05  ...  Warszawa  1030760  0090316  2021-01-01
        1   5699  04  04  03  ...  Warszawa  0845000  0844991  2021-01-01
        2   5975  04  14  07  ...  Warszawa  0093444  0093438  2021-01-01

        >>> warsaw.results.to_list("sym")  # equivalent to to_list("id")
        ['1030760', '0845000', '0093444']

        >>> warsaw.results.to_list("powiat")  # equivalent to_list("pow")
        [UnitLink(code='14', name='świecki', index=469),
         UnitLink(code='04', name='chełmiński', index=358),
         UnitLink(code='14', name='świecki', index=469)]

        >>> warsaw.results.to_list("pow", link=False)
        ['14', '04', '14']

        Returns
        -------
        list
        """
        field = self.system.ensure_field(field)

        if field not in self.system.fields:
            raise ValueError(f"{field!r} is not a valid field. "
                             f"Available fields: "
                             f"{', '.join(sorted(self.system.fields.keys()))}")

        new = getattr(self.frame, self.system.fields[field]).tolist()

        if link and self.system.link_mgr.has_link_mgr(field):
            self.__class__._form_root.new_list = new
            self.__class__._form_root.field = field

            with concurrent.futures.ThreadPoolExecutor() as exe:
                exe.map(self._form_root, range(len(new)))

        return new

    def transfer(
            self,
            key: Hashable,
            target: Union[str, Type],
            _kwt=MISSING,
            **other
    ) -> "System":
        """
        Search :target: system using search keywords
        and modes from this instance.
        """
        global transfer_collector

        if _kwt is not MISSING:
            keywords, transfer_target = _kwt(target)

        else:
            keywords, transfer_target = self.to_keywords(target)

        pop = transfer_collector.pop(key, ())
        transfer_collector[key] = pop + (self, transfer_target.search(
            **{**keywords, **other}))

        return transfer_collector[key][-1]


@dataclasses.dataclass(frozen=True)
class Entry:
    """
    System entry immutable dataclass.
    """

    system: System
    terid: str = None
    voivodship: str = None
    powiat: str = None
    gmina: str = None
    gmitype: str = None
    loctype: str = None
    streettype: str = None
    cnowner: str = None
    name: str = None
    secname: str = None
    function: str = None
    id: str = None  # pylint: disable=invalid-name
    integral_id: str = None
    row: pd.DataFrame = None
    date: str = None
    index: int = None

    _backend = _EntryGroup()  # inherit broker from EGB

    @property
    def is_entry(self):
        return True

    isentry = is_entry

    @property
    def integral(self):
        if self.integral_id:
            with simc() as integral_mgr:
                integral = integral_mgr.search(id=self.integral_id)
                return LocalityLink(
                    code=integral.id,
                    value=integral.name,
                    index=integral.index
                )

    @property
    def results(self):
        return self.system.results

    r = res = frame = results  # pylint: disable=invalid-name

    @result_of(_backend.to_keywords)
    def to_keywords(self, transfer_target: Union[str, type]):
        """
        Create and return keywords leading to current search results.

        Parameters
        ----------
        transfer_target: str or type
            Target class (SIMC, TERC or ULIC).

        Returns
        -------
        generator
        """
        transfer_target = (transfer_target, self.transfer_target)[True]
        properties = dict(self.system)
        str_eq = properties.pop('name')
        copy = properties.copy()

        for key, value in copy.items():
            if key in transfer_target.fields and str(value):
                properties[key] = str(value)
            else:
                properties.__delitem__(key)

        keywords = {
            **properties,
            'unpacked': True,
            'name': str_eq,
            'raise_for_failure': self.raise_for_failure,
            'case': self.case
        }

        yield keywords
        yield transfer_target

    def transfer(
            self,
            key: Hashable,
            target: Union[str, type],
            **other
    ) -> "System":
        return EntryGroup(
            self.system, frame=self.frame
        ).transfer(key, target, self.to_keywords, **other)

    def __getattribute__(self, item):
        try:
            return object.__getattribute__(self, item)
        except AttributeError as attribute_error:
            try:
                return object.__getattribute__(self.system, item)
            except AttributeError:
                raise AttributeError from attribute_error

    def __repr__(self, indent=4):
        conjunc = "\n" + " " * indent if indent > 0 else ""

        # F"{value=}" rule was ommitted in order to avoid
        # SyntaxError in older versions.
        return (f"{self.type}({conjunc if indent else ''}" +
                (f"name={self.name!r}, {conjunc}" if self.name else "") +
                (f"secname={self.secname!r}, {conjunc}"
                 if self.secname else "") +
                f"terid={self.terid!r}, {conjunc}"
                f"system={self.system.system.upper()}, {conjunc}"
                f"voivodship={self.voivodship!r}, {conjunc}" +
                (f"powiat={self.powiat!r}, {conjunc}" if self.powiat else "") +
                (f"gmina={self.gmina!r}, {conjunc}" if self.gmina else "") +
                (f"gmitype={self.gmitype!r}, {conjunc}"
                 if self.gmitype else "") +
                (f"loctype={self.loctype!r}, {conjunc}"
                 if self.loctype else "") +
                (f"streettype={self.streettype!r}, {conjunc}"
                 if self.streettype else "") +
                (f"cnowner={self.cnowner!r}, {conjunc}"
                 if self.cnowner else "") +
                (f"function={self.function!r}, {conjunc}"
                 if self.function else "") +
                (f"id={self.id!r}, {conjunc}" if self.id else "") +
                (f"integral_id={self.integral_id!r}, {conjunc}" if
                 self.integral_id else "") +
                f"date={self.date!r}, {conjunc}"
                f"index={self.index}" + ('\n' if indent else '') +
                ")")

    repr = __repr__


class Unit(Entry):
    """ TERC entry. """
    type = "Unit"

    @property
    def is_unit(self):
        """ State whether this object is a unit. """
        return True

    isunit = is_unit


class Locality(Entry):
    """ SIMC entry. """
    type = "Locality"

    @property
    def is_loc(self):
        return True

    isloc = is_loc


class Street(Entry):
    """ ULIC entry. """
    type = "Street"

    @property
    def is_street(self):
        """ State whether this object is a street. """
        return True

    isstreet = is_street

    @property
    def fullname(self):
        chunks = [self.streettype, self.secname, self.name]
        selectors = [True, bool(self.secname), True]

        return " ".join(map(str, itertools.compress(chunks, selectors)))

    @staticmethod
    def unpack_fullname(fullname, sep=" "):
        chunks = (fullname.split()
                  if not sep or sep.isspace()
                  else
                  fullname.split(sep))
        data = {}
        if len(chunks) not in [2, 3]:
            raise UnpackError("fullname has to be 3-part composision of "
                              "street type[, second name] and name. Length "
                              f"of the chunks given equals {len(chunks)}")
        data["streettype"] = chunks[0]
        if len(chunks) == 3:
            data["secname"] = chunks[1]
        data["name"] = chunks[-1]
        return data


entry_types = {
    "simc": Locality,
    "terc": Unit,
    "ulic": Street,
}

link_types = {
    "simc": LocalityLink,
    "terc": UnitLink,
}

error_types = {
    "simc": LocalityNotFound,
    "terc": UnitNotFound,
    "ulic": StreetNotFound,
}


# -----------------------------------------------------------------------------
#                                   Systems
# -----------------------------------------------------------------------------

# -----------
#    SIMC
# -----------

class SIMC(System, abc.ABC):
    """ SIMC system. """


# -----------
#    TERC
# -----------

class TERC(System, abc.ABC):
    """ TERC system. """


# -----------
#    ULIC
# -----------

class ULIC(System, abc.ABC):
    """ ULIC system. """


# Inject slaves and disinherit from abc.ABC as they're no more abstract classes
inject_slaves(SIMC, TERC, ULIC)
disinherit(parent=abc.ABC, klasses=[SIMC, TERC, ULIC])

# Internal helpers for the module-level API
# -----------------------------------------

MOST_RECENT = None
""" Most recent instance. """


def _make_recent(sys,
                 err=ValueError("system must be a valid "
                                "TERYT system name, "
                                "instance or type; "
                                "no proper was found/provided")):
    global MOST_RECENT  # pylint: disable=global-statement

    if sys is None:
        if MOST_RECENT is None:
            raise err

    elif isinstance(sys, str) and sys in systems:
        MOST_RECENT = eval_dict[sys]()

    elif isinstance(sys, type):
        if not issubclass(sys, System):
            raise err
        MOST_RECENT = sys()

    elif isinstance(sys, System):
        MOST_RECENT = sys.__class__()

    else:
        raise err

    return MOST_RECENT


# -----------------------------------------------------------------------------
#                              Module-level API
# -----------------------------------------------------------------------------


def reset_recent():
    """
    Reset the most recent instance of System created by a module-level
    API function.
    """
    global MOST_RECENT  # pylint: disable=global-statement
    MOST_RECENT = None


def get_entries(number, system=None, link=True, from_results=True):
    # doc: see System.get_entries.__doc__
    # pylint: disable=missing-function-docstring
    recent = _make_recent(system)

    if from_results:
        recent = recent.results

    return recent.get_entries(number, link=link)


def search(name=None, *, system=None, **keywords):
    # doc: see System.search.__doc__
    # pylint: disable=missing-function-docstring
    if name is not None:
        keywords["name"] = name

    return _make_recent(system).search(**keywords)


def index(i, system=None, **params):
    # doc: see System.index.__doc__
    # pylint: disable=missing-function-docstring
    return _make_recent(system).index(i, **params)


def transfer(results, to_system=None, **keywords):
    # doc: see Entry.transfer.__doc__
    # pylint: disable=missing-function-docstring
    if isinstance(to_system, type):
        raise TypeError("target system must be a "
                        "System instance or name, not type")

    keywords = {'target': results.database, **keywords}
    recent = _make_recent(results.database)

    return recent.results.transfer(to_system, **keywords)


def to_list(system=None, from_results=True, **params):
    # doc: see System.to_list.__doc__
    # pylint: disable=missing-function-docstring
    recent = _make_recent(system)

    if from_results:
        if recent.results:
            recent = recent.results

    return recent.to_list(**params)


tolist = to_list


def to_dict(system=None, from_results=True, **params):
    # doc: see System.to_dict.__doc__
    # pylint: disable=missing-function-docstring
    recent = _make_recent(system)

    if from_results:
        if recent.results:
            recent = recent.results

    return recent.to_dict(**params)


todict = to_dict


def ensure_field(root_name, system=None):
    # doc: see System.ensure_field.__doc__
    # pylint: disable=missing-function-docstring
    if isinstance(system, type):
        raise TypeError("system must be a System instance")

    return _make_recent(system).ensure_field(root_name)


# Copy the docs
# -------------
search.__doc__ = System.search.__doc__
index.__doc__ = System.index.__doc__
transfer.__doc__ = Entry.transfer.__doc__ = EntryGroup.transfer.__doc__
to_list.__doc__ = System.to_list.__doc__
to_dict.__doc__ = System.to_dict.__doc__
ensure_field.__doc__ = System.ensure_field.__doc__
get_entries.__doc__ = System.get_entries.__doc__

# Casefold system names
# ---------------------
terc = Terc = TERC  # pylint: disable=invalid-name
simc = Simc = SIMC  # pylint: disable=invalid-name
ulic = Ulic = ULIC  # pylint: disable=invalid-name

# -----------------------------------------------------------------------------
#                                    Setup
# -----------------------------------------------------------------------------
_terc = TERC(link=False)
_simc = SIMC(link=False)
_ulic = ULIC(link=False)

System.frame_link_mgrs = FrameLinkManagers()

eval_dict = {
    "terc": terc,
    "simc": simc,
    "ulic": ulic
}

names = {
    "terc": {*_terc["name"]},
    "simc": {*_simc["name"]},
    "ulic": {*_ulic["name"],
             *_ulic["secname"]},
}

name_case_descriptors = {
    "voivodship": str.upper,
    "powiat": str.casefold,
    "gmina": str.title,
}

_db = DictDatabase(CharacterNgramFeatureExtractor())

for _type, _name_set in names.items():
    _system = eval_dict[_type]()

    for _string in _name_set:
        _db.add(_string)

    _system.__class__._name_searcher = Searcher(_db, CosineMeasure())

globals().pop("_db", ...)
globals().pop("_system", ...)
globals().pop("_type", ...)
globals().pop("_name_set", ...)

globals().pop("_simc", ...)
globals().pop("_terc", ...)
globals().pop("_ulic", ...)

# The end. :)
