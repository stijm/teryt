""" Search TERYT. """

# This is the part of *teryt* library.
# Author: Stim (stijm), 2021
# License: GNU GPLv3

# Future features
# ---------------
# - System.filter
# - System.to_xml
# - System.results.to_xml

# TODO: rewrite the searching algorithm,
#  improve caching,
#  add teritorial division unit hierarchy and put it in a tree.

import dataclasses
import re
import typing
from abc import ABC
from concurrent.futures import ThreadPoolExecutor
from itertools import compress
from math import factorial
from pandas import DataFrame, Series
from typing import (
    final,
    Hashable,
    Union,
    Type,
)
from warnings import warn

from .data import na_char
from .data.implement import (
    function_dict,
    implement_common_data,
    implement_specific_data
)
from .data.manager import (
    simc_data,
    terc_data,
    ulic_data
)
from .exceptions import (
    ErroneousUnitName,
    Error,
    LocalityNotFound,
    StreetNotFound,
    UnitNotFound,
    UnpackError,
)
from .tools import (
    disinherit,
    require,
    set_broker,
    StringCaseFoldTuple,
    FrameSearch
)

systems = StringCaseFoldTuple(('simc', 'terc', 'ulic'))
transfer_collector = {}

_sentinel = object()


def transferred_searches(key):
    for transferred_search in set(transfer_collector.get(key, ())):
        name = getattr(transferred_search, 'system')
        if isinstance(name, System):
            name = name.system
        yield name, transferred_search


@dataclasses.dataclass(frozen=True)
class SemanticLink(object):
    """ TERYT semantic Link. """

    code: str
    value: typing.Any

    def __getitem__(self, item: (str, int)):
        return (
            {item: getattr(self, item, '')},
            [*dict(self).values()]
        )[isinstance(item, int)][item]

    def __str__(self):
        return str(self.code or '')

    def __add__(self, other):
        return str(self).__add__(other)

    def __bool__(self):
        return all([self.value, self.code])

    def __iter__(self):
        if self.value:
            yield 'value', self.value
        yield 'code', self.code
        i = getattr(self, 'index', None)
        if i:
            yield 'index', i


@dataclasses.dataclass(frozen=True)
class UnitLink(SemanticLink):
    """ Link to Unit. """
    index: int

    @property
    def as_unit(self):
        if self.index or self.index == 0:
            with terc() as unit_mgr:
                return unit_mgr.index(self.index)

    as_entry = as_unit


@dataclasses.dataclass(frozen=True)
class LocalityLink(SemanticLink):
    """ Link to Locality. """
    index: int

    @property
    def as_loc(self):
        if self.index or self.index == 0:
            with simc() as loc_mgr:
                return loc_mgr.index(self.index)

    as_entry = as_loc


class Search(object):
    """ TERYT searching algorithm class. """

    def __init__(
            self,
            *,
            database: DataFrame,
            system: str,
            method: str,
            fields: Union[dict, property],
            case: bool,
            str_contains: typing.Iterable,
            str_startswith: typing.Iterable,
            str_eq='',
    ):
        """ Constructor. """
        self.database = database
        self.system = system
        self._cur = self.database.copy()
        self._frames = [self._cur]
        self.method = method
        self.fields = fields
        self.case = case
        self.keywords = {}
        self._ineffective = ''
        self._str_eq = str_eq
        self._str_contains = str_contains
        self._str_startswith = str_startswith
        self.attempts = 0
        self.max_attempts = 0
        self._done = False

    def _failure(self):
        """ Was anything found? """
        return self._cur.empty or self._cur.equals(self.database)

    def _lookup(self):
        for field, query in (*self.keywords.items(),):
            if field == "voivodship":
                query = query.upper()
            self.attempts += 1
            if field not in self.fields:
                continue
            root = self.fields[field]
            keyword_args = dict(
                root=root,
                value=str(query),
                case=self.case
            )
            method = "name"
            if field in self._str_contains:
                method = "contains"
            elif field in self._str_startswith:
                method = "startswith"

            self._cur = getattr(
                FrameSearch(self._cur),
                method
            )(**keyword_args)

            if self._failure() and field != [*self.keywords][-1]:
                if self._cur.equals(self.database):
                    no_uniqueness = f'It seems that all values in ' \
                                    f'{field!r} field ' \
                                    f'are equal to {query!r}. Try ' \
                                    f'using more unique key words.'
                    warn(no_uniqueness, category=UserWarning)

                self._cur = self._frames[-1]
                if self.attempts <= self.max_attempts:
                    self.keywords[field] = self.keywords.pop(field)
                    self._lookup()
                else:
                    self._done = True
                    break

            self._frames.append(self._cur)

    def _name_lookup(self):
        print(self.method)
        self._cur = getattr(
            FrameSearch(self._cur),
            self.method
        )(root=self.fields["name"],
          value=self._str_eq,
          case=self.case)

        self._frames.append(self._cur)

    def search(self, keywords) -> "DataFrame":
        self.keywords = keywords
        self.max_attempts = factorial(len(self.keywords))

        if self.method != "NO_NAME":
            self._name_lookup()
            if self._failure():
                return DataFrame()

        if not self._done:
            self._lookup()
        return self._cur

    def __call__(self, keywords: dict):
        return self.search(keywords=keywords)


class GenericLinkManagerBroker(object):
    def __init__(self):
        self.largs = {}
        self.uargs = {}

    @staticmethod
    def link(gl_mgr, arguments, _keywords):
        if arguments:
            field = next(iter(arguments[:]))
            if field != 'integral':
                require(
                    gl_mgr.has_lm(field),
                    f'field {field!r} '
                    f'cannot be linked'
                )

    def link_names(self, gl_mgr, _arguments, keywords):
        expected_kw = sorted(gl_mgr.system.fields.keys())

        for keyword in list(set(keywords)):
            if keyword in expected_kw:
                e = gl_mgr.erroneous_argument
                TypeError(e % ("link_names", keyword, ', '.join(expected_kw)))

        for name, value in keywords.items():
            if gl_mgr.has_frame_lm(name):
                self.largs.update({name: value})
                continue
            elif gl_mgr.has_dict_lm(name):
                link_mgr = getattr(gl_mgr, name + "_link_manager")
                if value.isnumeric() and value in link_mgr.values():
                    value = dict.__call__(
                        map(reversed, link_mgr.items()))[value]
                if value not in link_mgr:
                    e = f"{value!r} is not a valid " \
                        f"{name.replace('_', ' ')!r} non-ID value"
                    raise ValueError(e)
                else:
                    value = link_mgr[value]
            self.uargs.update({name: value})


@dataclasses.dataclass(init=False)
class GenericLinkManager(object):
    system: "System"
    dict_link_managers: "DictLinkManagers"
    frame_link_managers: "FrameLinkManagers"

    broker = GenericLinkManagerBroker()

    def __init__(self, system, dict_link_managers, frame_link_managers):
        self.system = system
        self.dict_link_managers = dict_link_managers
        self.frame_link_managers = frame_link_managers
        self.store = self.system.cache[self.system.system].update
        self.link_indexes = {}

    @set_broker(broker.link_names)
    def link_names(self, **_keywords):
        largs = self.broker.largs
        uargs = self.broker.uargs
        extract = uargs.copy()
        for field in uargs.keys():
            if field not in self.system.terc.columns:
                extract.pop(field)

        fields = list(self.system.fields.keys())
        for field, value in largs.items():
            # TODO: implement a dict with all fields mappers
            if field == 'voivodship' and isinstance(value, str):
                value = value.upper()

            frame_lmname = field + 's'
            frame_link_manager: DataFrame = getattr(
                self.frame_link_managers, frame_lmname)

            if frame_link_manager.empty:
                warn(
                    f'no links available for {frame_lmname}. '
                    f'Updating search keywords with the provided value, '
                    f'however results are possible not to be found if it '
                    f'is not a valid ID.'
                )
                uargs.update({field: value})
                continue

            if value.isnumeric() and value in frame_link_manager.values:
                value = self.link(field, value)
                uargs.update({field: value})

            entry = Search(
                database=frame_link_manager,
                system="terc",
                method="equal",
                fields=terc.fields,
                case=False,
                str_contains=terc.posmethod,
                str_startswith=terc.prefmethod,
                str_eq=value
            )(keywords=uargs)

            if entry.empty:
                raise ErroneousUnitName(f"{value!r} is not a {field}")
            index = entry.iat[0, 0]
            self.link_indexes[field] = index
            link_result = {field: entry.iat[0, entry.columns.get_loc(
                self.system.fields[field]
            )]}
            self.system.entry_helper.update(link_result)
            uargs.update(link_result)

            if field != fields[0]:
                quantum = fields.index(field) - 1
                for rot in range(quantum + 1):
                    prev = fields[quantum - rot]
                    uargs[prev] = entry.iat[
                        0, entry.columns.get_loc(self.system.fields[prev])
                    ]

        return dict(**extract, **uargs)

    def has_dict_lm(self, field: str) -> "bool":
        return hasattr(self.dict_link_managers, field + '_link_manager')

    def has_frame_lm(self, field: str) -> "bool":
        return hasattr(self.frame_link_managers, field + 's')

    def has_lm(self, field: str) -> "bool":
        return self.has_dict_lm(
            field
        ) or self.has_frame_lm(
            field
        )

    @set_broker(broker.link)
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

        if self.has_dict_lm(field):
            return dict.__call__(map(
                reversed,
                getattr(self.dict_link_managers, field + '_link_manager'
                        ).items()))[value]

        if field == "integral":  # special case
            new = simc()
            integral = new.search
            return integral

        unit_mgr = terc()

        if field not in self.system.link_fields or str(value) == na_char:
            return ""

        keywords = {'function': function_dict[self.system.fields[field]]}
        fields = list(self.system.fields.keys())
        helper = self.system.entry_helper

        if field != fields[0]:
            quantum = fields.index(field) - 1
            for rot in range(quantum + 1):
                prev_value = fields[quantum - rot]
                if prev_value not in helper:
                    raise ValueError(f"cannot link {field!r} as "
                                     f"its overriding field {prev_value!r} "
                                     f"is not declared")
                keywords[prev_value] = str(helper[prev_value])

        keywords[field] = value

        if field != fields[-1]:
            next_value = fields[fields.index(field) + 1]
            keywords[next_value] = na_char

        cache = self.system.cache[self.system.system]

        if tuple(keywords.items()) in cache:
            return cache[tuple(keywords.items())]

        result = Search(
            database=unit_mgr.database,
            system=unit_mgr.system,
            method='NO_NAME',
            fields=unit_mgr.fields,
            case=False,
            str_contains=unit_mgr.posmethod,
            str_startswith=unit_mgr.prefmethod,
        )(keywords=keywords)

        self.link_indexes[field] = result.iat[0, 0]
        name = result.iat[
            0, result.columns.get_loc(unit_mgr.fields["name"])
        ]
        self.store({tuple(keywords.items()): name})
        return name

    def __getattribute__(self, item):
        try:
            return object.__getattribute__(self, item)
        except AttributeError:
            if item.endswith("s"):
                return getattr(self.frame_link_managers, item)
            elif item.endswith("_link_manager"):
                return getattr(self.dict_link_managers, item)
            raise


class SystemBroker(object):
    @staticmethod
    def search(system, arguments, keywords):
        system.__init__(**system.modes)

        if len(arguments) == 1:
            keyword = next(iter(arguments[:]))

            if not isinstance(keyword, str):
                raise TypeError(
                    f"name must be str, not {keyword.__class__.__name__}")
            if "name" in keywords:
                raise ValueError(
                    "passed multiple values for 'name' parameter")

            keywords["name"] = keyword
            arguments = ()

        if arguments:
            raise ValueError(
                'cannot perform searching: '
                'only one positional argument can be accepted')

        if not keywords:
            raise ValueError(
                'cannot perform searching: '
                'no keyword arguments')

        search_keywords = tuple(
            set(system.locname_keywords + (*system.fields.keys(),))
        ) + system.optional_str_arguments

        keywords = dict(map(  # roots -> fields
            lambda kv: (system.ensure_field(kv[0]), kv[1]),
            keywords.items())
        )

        if not any(map(keywords.__contains__, search_keywords)):
            raise ValueError(
                f'no keyword arguments for searching '
                f'(expected at least one from: '
                f'{", ".join(sorted(search_keywords))}'
                f')'
            )

        for conflicted in system.conflicts:
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

        system.name_field, system.force_unpack = None, False
        modes = system.locname_keywords + ('NO_NAME',)
        system.method = modes[-1]

        for keyword in keywords.copy():
            for mode in system.locname_keywords:
                if keyword == mode:
                    system.method = mode
                    system.name_field = keywords[mode]
                    del keywords[mode]

        system.raise_for_failure = keywords.pop("raise_for_failure",
                                                system.raise_for_failure)
        system.unpack_mode = keywords.pop("unpack", system.unpack_mode)
        system.link_mode = keywords.pop("link", system.link_mode)
        system.force_unpack = keywords.pop("force_unpack", system.force_unpack)
        system.unpacked = keywords.pop("unpacked", False)
        system.case = keywords.pop("case", system.case)
        terid = keywords.pop("terid", '')

        if not system.unpacked:
            system.link_manager.erroneous_argument = system.erroneous_argument
            keywords = system.link_manager.link_names(**keywords)
        if terid:
            unpacked = system.unpack_terid(terid)
            [keywords.__setitem__(n, v) for n, v in unpacked.items() if v]

        system.keywords = keywords
        system._cur = system.database[:]

        for field in system.keywords:
            root_name = system.fields.get(field, None)
            # KeyError: 'secname' after using ULIC
            if not root_name:
                continue
            system.database[root_name] = system.database[
                system.fields[field]
            ].map(str)

    @staticmethod
    def unpack_row(system, args, keywords):
        if len(args) == 1:
            row = args[0]
        else:
            row = keywords.pop("row", _sentinel)
        fields = system.fields
        if isinstance(row, Series):
            row = DataFrame([[*row]], columns=row.keys())
        if row is _sentinel:
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
        system._cur = row


class DictLinkManagers(object):
    def __init__(self, register):
        for attr in filter(
                lambda a: a.endswith("link_manager"), dir(register)):
            setattr(self, attr, getattr(register, attr))


class FrameLinkManagers(object):
    def __init__(self):
        om = dict(unpack=False, unpacked=True)
        self._m = terc(link=False)
        self.voivodships = self._m.search(function='województwo', **om)
        self.powiats = self._m.search(function='powiat', **om)
        self.gminas = self._m.search(function='gmina', **om)

    def __repr__(self):
        return f"FrameLinkManagers({self._m!r})"


class System(ABC):
    __slots__ = ()

    cache = {
        "simc": {},
        "terc": {},
        "ulic": {}
    }

    posmethod = (
        "function",
    )
    prefmethod = (
        "date",  # TODO: this should be a datetime argument
    )
    locname_keywords = (
        "name",
        "match",
        "startswith",
        "endswith",
        "contains"
    )
    conflicts = (
        locname_keywords,
        ("force_unpack", "unpack")
    )
    optional_bool_arguments = (
        "raise_for_failure",
        "force_unpack",
        "unpack",
        "link",
        "unpacked",
        "case"
    )
    optional_str_arguments = (
        "terid",
    )
    optional_bool_str_arguments = (
        *optional_bool_arguments,
        *optional_str_arguments
    )
    erroneous_argument = \
        f"%s() got an unexpected keyword argument %r. " \
        f"Try looking for the proper argument name " \
        f"in the following list:\n{' ' * 12}%s."

    broker = SystemBroker()

    def __init__(
            self,
            case=False,
            link=True,
            unpack=True,
            raise_for_failure=False,
    ):
        """
        Constructor.

        Parameters
        ----------
        link : bool
            Whether to link the values in
            search/(converting to list)/(converting to dict).

        unpack : bool
            Whether to unpack future results/processed rows.
        """
        self.__class__.conflicts += tuple(
            map(lambda ls: ('terid', ls), self.link_fields))

        # Modes
        self.case = case
        self.force_unpack = False
        self.link_mode = link
        self.raise_for_failure = raise_for_failure
        self.unpack_mode = unpack
        self.modes = dict(
            case=self.case,
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
        self.database: DataFrame = getattr(
            self, self.system, None
        )
        if self.database is None:
            raise ValueError(f"invalid system {self.system!r}")
        self.database = self.database.reset_index()
        self._cur = None

        # Root names
        self.root_names = self.roots = [*self.database.columns]

        # Searching
        self.name_field = None
        self.method = None
        self.search_keywords = {}
        self.found_results = False
        self.unpacked = False
        self.linked = False
        self.current = DataFrame()
        self._results = EntryGroup(self)

        # Transferring
        self.transfer_target = None

        # Building links
        self.entry_helper = {}

        # Caching
        self.store = self.__class__.cache[self.system].update

        if link:
            self.link_manager = GenericLinkManager(
                dict_link_managers=DictLinkManagers(self),
                frame_link_managers=System.frame_link_managers,
                system=self
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            raise

    def __getitem__(self, item):
        return dict(self)[item]

    def __iter__(self):
        if self.entry_helper["terid"]:
            yield "terid", self.entry_helper["terid"]
        for field in self.fields:
            if self.entry_helper.get(field, ""):
                yield field, self.entry_helper.get(field, "")

    def __len__(self):
        return len(self.database)

    def __repr__(self):
        init_args = ", ".join(map(str, compress(
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

    def _dispatcher(self):
        current = self._cur.reset_index()
        if self._failure():
            not_found_err = error_types[self.system]
            if self.raise_for_failure:
                raise not_found_err("no results found")
            self.__init__(**self.modes)
        else:
            self.found_results = True
            self._results = EntryGroup(self, current)
            self._results.frame = self._results.frame.drop(columns=["level_0"])
            if (len(self.r) == 1 or self.force_unpack) and self.unpack_mode:
                return self.unpack_row(self.results)
        return self.results

    @final
    def _failure(self):
        return self._cur.empty or self._cur.equals(self.database)

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
        fields = self.fields
        if name in fields:
            return name
        root_name_upper = name.upper()  # root names are upper
        return dict([*map(reversed, fields.items())]).get(
            root_name_upper, name
        )

    @property
    def fields(self):
        """ Fields. """
        raise NotImplementedError

    def index(self, i, /, *, link=True):
        """
        Return an entry by index.

        Parameters
        ----------
        i : int
            Positional argument. Index of an entry.

        link : bool
            Whether to link the result entry.

        inplace : bool
            Whether to perform on the current instance or new.

        Returns
        -------
        Entry
        """
        if (len(self.database) - 1) < i:
            raise ValueError(f"index too large (max: {len(self.database) - 1})")
        return self.unpack_row(row=self.database.iloc[i], link=link)

    get_entry = index

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
            filter(lambda x: str(x) != na_char, self.link_fields)
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

    @set_broker(broker.search)
    def search(
            self,
            *args,  # noqa for autocompletion
            **keywords  # noqa for autocompletion
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
        #
        # TODO: Unit, Locality and Street objects should be also legal
        #       as search keywords. (09-02-2021)
        #
        self._cur = Search(
            database=self.database,
            system=self.system,
            method=self.method,
            fields=self.fields,
            case=self.case,
            str_eq=self.name_field,
            str_contains=self.posmethod,
            str_startswith=self.prefmethod
        )(keywords=self.search_keywords)

        return self._dispatcher()

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

    def _treat_data_chunk(self, field, root):
        code: str = self._cur.iat[
            0, self._cur.columns.get_loc(root)
        ]

        if code != na_char:
            self.entry_helper[field] = code
            if self.link_manager.has_lm(field) and self.link_mode:
                value = self.link_manager.link(field, code)

                index = self.link_manager.link_indexes.get(
                    field, _sentinel)
                if index is not _sentinel:
                    self.entry_helper[field] = UnitLink(
                        code=code, value=value, index=index)
                else:
                    self.entry_helper[field] = SemanticLink(
                        code=code, value=value)

    @set_broker(broker.unpack_row)
    def unpack_row(
            self,
            row: Union["EntryGroup", "Series", "DataFrame"] = None,  # noqa
            *,
            link=True
    ) -> "Entry":
        """
        Unpack one-row DataFrame to Entry instance.

        Parameters
        ----------
        row : EntryGroup, Series or DataFrame
            Entry group/DataFrame with length 1 or Series to unpack.

        link : bool
            Whether to link the linkable values.

        Returns
        -------
        Entry
        """
        print(self._cur)
        self.link_mode = link
        for chunk in self.fields.items():
            self._treat_data_chunk(*chunk)
        self.entry_helper["terid"] = self.pack_terid(**self.entry_helper)
        self.unpacked = True
        return entry_types[self.system].__call__(
            system=self,
            **self.entry_helper,
            row=self.current,
            index=self.current.iat[0, 0]
        )

    def multi_unpack(self,
                     rows: Union["EntryGroup", "DataFrame"] = None,
                     *,
                     link=True
                     ):
        for row in rows.iterrows():
            print(row)

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
            f"{self.system.upper()} teritorial ID length "
            f"is expected to be maximally {max_length}"
        index = 0

        for link_manager_field, proper_length in self.link_fields.items():
            if index >= len(teritorial_id) - 1:
                break
            frames.update(
                {link_manager_field: getattr(
                    self.link_manager, link_manager_field + 's'
                )}
            )
            chunk = teritorial_id[index:index + proper_length]
            unpack = self.unpack_mode
            if errors:
                checker = type(self)().search(
                    unpacked=True, unpack=False,
                    **{link_manager_field: chunk})
                if checker.results.empty:
                    raise ValueError(
                        repr(chunk) +
                        f"is an incorrect teritorial code chunk "
                        f"(error at {link_manager_field!r} field, "
                        f"root name "
                        f"{self.fields[link_manager_field]!r})"
                    )
            self.unpack_mode = unpack
            chunks.update({link_manager_field: chunk})
            index += proper_length

        return chunks


implement_common_data(System)


class EntryGroupBroker(object):
    @staticmethod
    def to_keywords(klass, arguments, _kwds):
        require(arguments, "to_keywords(): no arguments")
        target_name = arguments[0]
        if target_name in list(map(eval, systems)):
            target_name = target_name.__name__
        if target_name not in systems:
            raise ValueError(
                f"cannot evaluate transfer target using name {target_name!r}")

        klass.transfer_target = eval(target_name)()


class EntryGroup(object):
    broker = EntryGroupBroker()

    def __contains__(self, item):
        return self.frame.__contains__(item)

    def __getattribute__(self, item):
        try:
            return object.__getattribute__(self, item)
        except AttributeError:
            return getattr(self.frame, item)

    def __init__(self, system, frame=DataFrame()):
        self.system = system
        self.frame = frame

    def __len__(self):
        return len(self.frame)

    def __repr__(self):
        return repr(self.frame)

    def get_entry(self, number, link=True):
        series = self.frame.iloc[number]
        return (lambda: series,
                lambda: self.system.__class__().unpack_row
                (series)
                )[link]()

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
        new_dict = {}
        for field in self.system.fields:
            value = [*self.to_list(field, link=link)]
            name = (field, self.system.fields[field])[root_names]
            new_dict.update(
                {name: value[0] if len(value) == 1 else value}
            )
        if indexes:
            new_dict["index"] = [*range(0, len(frame))]
        return new_dict

    todict = to_dict

    @set_broker(broker.to_keywords)
    def to_keywords(self, transfer_target: Union[str, Type]):
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
        name = {}
        if self.system.name_field:
            name = dict(match=re.escape(self.system.name_field))
        yield dict(
            **name,
            **self.system.keywords,
            **self.system.modes
        )
        yield transfer_target

    def _form_root(self, root_index):
        new = self.system.__class__()
        entry = new.unpack_row(DataFrame([
            self.frame.loc[self.frame.index[root_index]]
        ]))
        self.__class__._form_root.new_list[root_index] = getattr(  # noqa
            entry, self.__class__._form_root.field)  # noqa

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
        new_list = getattr(self.frame, self.system.fields[field]).tolist()
        if link and self.system.link_manager.has_lm(field):
            self.__class__._form_root.new_list = new_list
            self.__class__._form_root.field = field
            with ThreadPoolExecutor() as exe:
                exe.map(self._form_root, range(len(new_list)))

        return new_list

    def transfer(
            self,
            key: Hashable,
            target: Union[str, Type],
            _kwt=None,
            **other
    ) -> "System":
        """
        Search :target: system using search keywords
        and modes from this instance.
        """
        global transfer_collector
        if _kwt:
            keywords, transfer_target = _kwt(target)
        else:
            keywords, transfer_target = self.to_keywords(target)
        pop = transfer_collector.pop(key, ())
        transfer_collector[key] = pop + (self, transfer_target.search(
            **{**keywords, **other}))
        return transfer_collector[key][-1]


class EntryBroker(object):
    to_keywords = EntryGroupBroker.to_keywords


@dataclasses.dataclass(frozen=True)
class Entry(object):
    """
    System entry class.
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
    id: str = None
    integral_id: str = None
    row: DataFrame = None
    date: str = None
    index: int = None

    broker = EntryBroker()

    @property
    def is_entry(self):
        return True

    isentry = is_entry

    @property
    def integral(self):
        if self.integral_id:
            with simc() as integral_manager:
                integral = integral_manager.search(id=self.integral_id)
                return LocalityLink(
                    code=integral.id,
                    value=integral.name,
                    index=integral.index
                )

    @property
    def results(self):
        return self.system.results

    r = res = frame = results

    @set_broker(broker.to_keywords)
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
        name_field_value = properties.pop('name')
        copy = properties.copy()
        for k, v in copy.items():
            if k in transfer_target.fields and str(v):
                properties[k] = str(v)
            else:
                properties.__delitem__(k)
        keywords = {
            **properties,
            'unpacked': True,
            'name': name_field_value,
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
        # System is mutable;
        # Entry isn't.
        try:
            return object.__getattribute__(self, item)
        except AttributeError as a:
            try:
                return object.__getattribute__(self.system, item)
            except AttributeError:
                raise a from a

    def __repr__(self, indent=True):
        joiner = '\n    ' if indent else ''
        # TODO: compress it maybe?
        return (f"{self.type}({joiner if indent else ''}" +
                (f"name={self.name!r}, {joiner}" if self.name else "") +
                (f"secname={self.secname!r}, {joiner}"
                 if self.secname else "") +
                f"terid={self.terid!r}, {joiner}"
                f"system={self.system.system.upper()}, {joiner}"
                f"voivodship={self.voivodship!r}, {joiner}" +
                (f"powiat={self.powiat!r}, {joiner}" if self.powiat else "") +
                (f"gmina={self.gmina!r}, {joiner}" if self.gmina else "") +
                (f"gmitype={self.gmitype!r}, {joiner}"
                 if self.gmitype else "") +
                (f"loctype={self.loctype!r}, {joiner}"
                 if self.loctype else "") +
                (f"streettype={self.streettype!r}, {joiner}"
                 if self.streettype else "") +
                (f"cnowner={self.cnowner!r}, {joiner}"
                 if self.cnowner else "") +
                (f"function={self.function!r}, {joiner}"
                 if self.function else "") +
                (f"id={self.id!r}, {joiner}" if self.id else "") +
                (f"integral_id={self.integral_id!r}, {joiner}" if
                 self.integral_id else "") +
                f"date={self.date!r}, {joiner}"
                f"index={self.index}" + ('\n' if indent else '') +
                ")")


class Unit(Entry):
    """ TERC entry. """
    type = "Unit"

    @property
    def is_unit(self):
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
        return True

    isstreet = is_street

    @property
    def fullname(self):
        return " ".join(
            map(str, compress(
                [self.streettype, self.secname, self.name],
                [True, bool(self.secname), True]
            )))


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


class SIMC(System, ABC):
    """ SIMC system. """


class TERC(System, ABC):
    """ TERC system. """


class ULIC(System, ABC):
    """ ULIC system. """


implement_specific_data(SIMC, TERC, ULIC)
disinherit(parent=ABC, klasses=[SIMC, TERC, ULIC])  # not abstract classes

most_recent = None
""" Most recent instance. """


def _make_recent(sys,
                 err=ValueError("system must be a valid "
                                "TERYT system name, "
                                "instance or type; "
                                "no proper was found/provided")):
    global most_recent
    if sys is None:
        if most_recent is None:
            raise err
    elif isinstance(sys, str) and sys in systems:
        most_recent = eval(sys)()
    elif isinstance(sys, type):
        if not issubclass(sys, System):
            raise err
        most_recent = sys()
    elif isinstance(sys, System):
        most_recent = sys.__class__()
    else:
        raise err
    return most_recent


def reset_recent():
    global most_recent
    most_recent = None


def get_entry(number, system=None, link=True, from_results=True):
    recent = _make_recent(system)
    if from_results:
        recent = recent.results
    return recent.get_entry(number, link=link)


def search(name=None, *, system=None, **keywords):
    if name is not None:
        keywords["name"] = name
    return _make_recent(system).search(**keywords)


def sys_index(i, system=None, **params):
    return _make_recent(system).index(i, **params)


def transfer(results, to_system=None, **keywords):
    if isinstance(to_system, type):
        raise TypeError("target system must be a "
                        "System instance or name, not type")
    keywords = {'target': results.database, **keywords}
    recent = _make_recent(results.database)
    return recent.transfer(to_system, **keywords)


def to_list(system=None, from_results=True, **params):
    recent = _make_recent(system)
    if from_results:
        if recent.results:
            recent = recent.results
    return recent.to_list(**params)


tolist = to_list


def to_dict(system=None, from_results=True, **params):
    recent = _make_recent(system)
    if from_results:
        if recent.results:
            recent = recent.results
    return recent.to_dict(**params)


todict = to_dict


def ensure_field(root_name, system=None):
    if isinstance(system, type):
        raise TypeError("system must be a System instance")
    return _make_recent(system).ensure_field(root_name)


search.__doc__ = System.search.__doc__
sys_index.__doc__ = System.index.__doc__
transfer.__doc__ = Entry.transfer.__doc__ = EntryGroup.transfer.__doc__
to_list.__doc__ = System.to_list.__doc__
to_dict.__doc__ = System.to_dict.__doc__
ensure_field.__doc__ = System.ensure_field.__doc__

terc = Terc = TERC
simc = Simc = SIMC
ulic = Ulic = ULIC

System.frame_link_managers = FrameLinkManagers()
