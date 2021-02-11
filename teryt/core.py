""" Search TERYT. """

# This is the part of *teryt* library.
# Author: Stim (stijm), 2021
# License: GNU GPLv3

# Future features
# ---------------
# - Register.filter
# - Register.to_xml
# - Register.results.to_xml

import dataclasses
import pandas
import re
import typing
from abc import ABC
from math import factorial
from itertools import compress
from typing import (
    final,
    Hashable,
    Union,
)
from warnings import warn
from .data.implement import (
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
    set_sentinel,
    StringCaseFoldTuple,
    FrameSearch
)

systems = StringCaseFoldTuple(('simc', 'terc', 'ulic'))
transfer_collector = {}


def transferred_searches(key):
    for transferred_search in set(transfer_collector.get(key, ())):
        name = getattr(transferred_search, 'system')
        if isinstance(name, Register):
            name = name.system
        yield name, transferred_search


@dataclasses.dataclass(frozen=True)
class Link(object):
    """ TERYT Link. """

    code: str
    name: typing.Any

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
        return all([self.name, self.code])

    def __iter__(self):
        if self.name:
            yield 'name', self.name
        yield 'code', self.code
        i = getattr(self, 'index', None)
        if i:
            yield 'index', i


@dataclasses.dataclass(frozen=True)
class UnitLink(Link):
    """ Link to Unit. """
    index: int

    @property
    def as_unit(self):
        if self.index or self.index == 0:
            with terc() as unit_manager:
                return unit_manager.index(self.index)

    as_entry = as_unit


@dataclasses.dataclass(frozen=True)
class LocalityLink(Link):
    """ Link to Locality. """
    index: int

    @property
    def as_loc(self):
        if self.index or self.index == 0:
            with simc() as loc_manager:
                return loc_manager.index(self.index)

    as_entry = as_loc


class Search(object):
    """ TERYT searching algorithm class. """

    def __init__(
            self,
            *,
            dataframe: pandas.DataFrame,
            system: str,
            search_mode: str,
            fields: Union[dict, property],
            case: bool,
            by_possession: typing.Iterable,
            by_prefix: typing.Iterable,
            locname: str = '',
    ):
        """ Constructor. """
        self.dataframe = dataframe
        self.candidate = self.dataframe.copy()
        self.frames = [self.candidate]
        self.database_name = system
        self.search_mode = search_mode
        self.fields = fields
        self.case = case
        self.search_keywords = {}
        self.ineffective = ''
        self.locname = locname
        self.by_possession = by_possession
        self.by_prefix = by_prefix

    def failure(self):
        """ Was anything found? """
        return self.candidate.empty or self.candidate.equals(self.dataframe)

    def move_key(self):
        keys = [*self.search_keywords.keys()]
        values = [*self.search_keywords.values()]
        ineffective_key_index = keys.index(self.ineffective)
        ineffective_value = values[ineffective_key_index]
        del keys[ineffective_key_index], values[ineffective_key_index]
        keys.insert(ineffective_key_index + 1, self.ineffective)
        values.insert(ineffective_key_index + 1, ineffective_value)
        self.search_keywords = dict(zip(keys, values))
        return self.search_keywords

    def search(self, search_keywords) -> "pandas.DataFrame":
        # TODO: no shuffling by itself, it should be itertools.product.
        self.search_keywords = search_keywords

        def locname_search():
            """
            Search for locality name.
            """
            self.candidate = getattr(
                FrameSearch(self.candidate),
                self.search_mode
            )(col=self.fields['name'],
              value=self.locname,
              case=self.case)

            self.frames.append(self.candidate)

        if self.search_mode != 'no_locname':
            locname_search()
            if self.failure():
                return pandas.DataFrame()

        attempts = 0
        max_attempts = factorial(len(self.search_keywords))
        done = False

        def search_loop():
            nonlocal self, attempts, done
            for field, query in self.search_keywords.items():
                if field == 'voivodship':
                    query = query.upper()
                attempts += 1
                if field not in self.fields:
                    continue
                col = self.fields[field]
                keyword_args = dict(
                    col=col,
                    value=str(query),
                    case=self.case
                )
                frame_query = 'name'
                if field in self.by_possession:
                    frame_query = 'contains'
                elif field in self.by_prefix:
                    frame_query = 'startswith'

                self.candidate = getattr(
                    FrameSearch(self.candidate),
                    frame_query
                )(**keyword_args)

                if self.failure():
                    if self.candidate.equals(self.dataframe):
                        no_uniqueness = f'It seems that all values in ' \
                                        f'{field!r} field ' \
                                        f'are equal to {query!r}. Try ' \
                                        f'using more unique key words.'
                        warn(no_uniqueness, category=UserWarning)
                    self.candidate = self.frames[-1]
                    if attempts <= max_attempts:
                        self.ineffective = field
                        self.move_key()
                        search_loop()
                    else:
                        done = True
                        break
                self.frames.append(self.candidate)

        if not done:
            search_loop()
        return self.candidate

    def __call__(self, search_keywords: dict):
        return self.search(search_keywords=search_keywords)


class GenericLinkManagerSentinel(object):
    def __init__(self):
        self.linkable_args = {}
        self.unlinkable_args = {}

    @staticmethod
    def link(klass, arguments, _keywords):
        if arguments:
            field = next(iter(arguments[:]))
            if field != 'integral':
                require(
                    klass.has_lm(field),
                    f'field {field!r} '
                    f'cannot be linked'
                )

    def link_names(self, klass, _args, keywords):
        expected_kw = sorted(klass.klass.fields.keys())
        # self.__init__()

        for keyword in list(set(keywords)):
            if keyword in expected_kw:
                e = klass.erroneous_argument
                TypeError(e % ('link_names', keyword, ', '.join(expected_kw)))

        for name, value in keywords.items():
            if klass.has_frame_lm(name):
                self.linkable_args.update({name: value})
                continue
            elif klass.has_dict_lm(name):
                link_manager = getattr(klass, name + '_link_manager')
                if value.isnumeric() and value in link_manager.values():
                    value = dict.__call__(map(reversed, link_manager.items()))[value]
                elif value not in link_manager:
                    e = f'{value!r} is not a valid ' \
                        f'{name.replace("_", " ")!r} non-ID value'
                    raise ValueError(e)
                else:
                    value = link_manager[value]
            self.unlinkable_args.update({name: value})


@dataclasses.dataclass(init=False)
class GenericLinkManager(object):
    klass: "Register"
    dict_link_managers: "DictLinkManagers"
    frame_link_managers: "FrameLinkManagers"
    cache: dict

    sentinel = GenericLinkManagerSentinel()

    def __init__(self, klass, dict_link_managers, frame_link_managers, cache):
        self.klass = klass
        self.dict_link_managers = dict_link_managers
        self.frame_link_managers = frame_link_managers
        self.cache = cache
        self.store = self.cache.update
        self.link_indexes = {}

    @set_sentinel(sentinel.link_names)
    def link_names(self, **_keywords):
        linkable = self.sentinel.linkable_args
        unlinkable = self.sentinel.unlinkable_args
        extract = unlinkable.copy()
        for field in unlinkable.keys():
            if field not in self.klass.terc.columns:
                extract.pop(field)

        fields = list(self.klass.fields.keys())
        for field, value in linkable.items():
            # TODO: implement a dict with all fields mappers
            if field == 'voivodship':
                value = value.upper()

            frame_lmname = field + 's'
            frame_link_manager = getattr(self.frame_link_managers,
                                         frame_lmname)

            if frame_link_manager.empty:
                warn(
                    f'no links available for {frame_lmname}. '
                    f'Updating search keywords with the provided value, '
                    f'however results are possible not to be found if it '
                    f'is not a valid ID.'
                )
                unlinkable.update({field: value})
                continue

            entry = Search(
                dataframe=frame_link_manager,
                system="terc",
                search_mode="equal",
                fields=terc.fields,
                case=False,
                by_possession=terc.by_possession,
                by_prefix=terc.by_prefix,
                locname=value
            )(search_keywords=unlinkable)

            require(not entry.empty,
                    ErroneousUnitName(f"{value!r} is not a {field}"))
            index = entry.iat[0, 0]
            link_result = {field: entry.iat[0, entry.columns.get_loc(
                self.klass.fields[field]
            )]}
            unlinkable.update(link_result)
            self.link_indexes[field] = index

            if field != fields[0]:
                quantum = fields.index(field) - 1
                for rot in range(quantum + 1):
                    prev = fields[quantum - rot]
                    unlinkable[prev] = entry.iat[
                        0, entry.columns.get_loc(self.klass.fields[prev])
                    ]

        return dict(**extract, **unlinkable)

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

    @set_sentinel(sentinel.link)
    def link(self, field: str, value: str):
        """
        Resolve locality that value in field refers to,
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

        if field == 'integral':  # special case
            new = simc()
            integral = new.search
            return integral

        unit_link_manager = terc()

        if field not in self.klass.link_fields or str(value) == 'nan':
            return ''

        keywords = {'function': self.klass.fields[field]}
        fields = list(self.klass.fields.keys())
        helper = self.klass.entry_helper

        if field != fields[0]:
            quantum = fields.index(field) - 1
            for rot in range(quantum + 1):
                prev_value = fields[quantum - rot]
                keywords[prev_value] = str(helper[prev_value])

        keywords[field] = value

        if field != fields[-1]:
            next_value = fields[fields.index(field) + 1]
            keywords[next_value] = 'nan'

        if tuple(keywords.items()) in self.cache:
            return self.cache[tuple(keywords.items())]

        result = Search(
            dataframe=unit_link_manager.database,
            system=unit_link_manager.system,
            search_mode='no_locname',
            fields=unit_link_manager.fields,
            case=False,
            by_possession=unit_link_manager.by_possession,
            by_prefix=unit_link_manager.by_prefix,
        )(search_keywords=keywords)

        name = result.iat[
            0, result.columns.get_loc(unit_link_manager.fields["name"])
        ]
        self.link_indexes[field] = result.iat[0, 0]
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


class RegisterSentinel(object):
    @staticmethod
    def search(klass, arguments, keywords):
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
            set(klass.locname_keywords + (*klass.fields.keys(),))
        ) + klass.optional_str_arguments

        keywords = dict(map(  # roots -> fields
            lambda kv: (klass.ensure_field(kv[0]), kv[1]),
            keywords.items())
        )

        if not any(map(keywords.__contains__, search_keywords)):
            raise ValueError(
                f'no keyword arguments for searching '
                f'(expected at least one from: '
                f'{", ".join(sorted(search_keywords))}'
                f')'
            )

        for conflicted in klass.conflicts:
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

        klass.locname_field, klass.force_unpack = None, False
        modes = klass.locname_keywords + ('no_locname',)
        klass.search_mode = modes[-1]

        for keyword in keywords.copy():
            for mode in klass.locname_keywords:
                if keyword == mode:
                    klass.search_mode = mode
                    klass.locname_field = keywords[mode]
                    del keywords[mode]

        klass.raise_for_failure = keywords.pop("raise_for_failure",
                                               klass.raise_for_failure)
        klass.unpack_mode = keywords.pop("unpack", klass.unpack_mode)
        klass.link_mode = keywords.pop("link", klass.link_mode)
        klass.force_unpack = keywords.pop("force_unpack", klass.force_unpack)
        klass.unpacked = keywords.pop("unpacked", False)
        klass.case = keywords.pop("case", klass.case)
        terid = keywords.pop("terid", '')

        if not klass.unpacked:
            klass.link_manager.erroneous_argument = klass.erroneous_argument
            keywords = klass.link_manager.link_names(**keywords)
        if terid:
            unpacked = klass.unpack_terid(terid)
            [keywords.__setitem__(n, v) for n, v in unpacked.items() if v]

        klass.search_keywords = keywords
        klass._candidate = klass.database[:]

        for field in klass.search_keywords:
            root_name = klass.fields[field]
            klass.database[root_name] = klass.database[
                klass.fields[field]
            ].map(str)

    @staticmethod
    def unpack_row(klass, _args, keywords):
        row = keywords.pop("dataframe", klass.results)
        fields = klass.fields
        if row is None:
            raise UnpackError("nothing to unpack from")
        if row.empty:
            raise UnpackError("nothing to unpack from")
        if len(row) != 1:  # it's not a row then
            raise UnpackError(
                "cannot unpack from more "
                "than one TERYT entry "
                f"(got {len(row)} entries)"
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
        klass.current_row = row


class DictLinkManagers(object):
    def __init__(self, register):
        for attr in filter(
                lambda a: a.endswith("link_manager"), dir(register)):
            setattr(self, attr, getattr(register, attr))


class FrameLinkManagers(object):
    def __init__(self):
        om = dict(unpack=False, unpacked=True)
        self._m = terc(link=False)
        self.voivodships = self._m.search(function='województwo', **om).r
        self.powiats = self._m.search(function='powiat', **om).r
        self.gminas = self._m.search(function='gmina', **om).r

    def __repr__(self):
        return f"FrameLinkManagers({self._m!r})"


class Register(ABC):
    __slots__ = ()

    by_possession = (
        "function",
    )
    by_prefix = (
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

    sentinel = RegisterSentinel()

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
            search/converting to list/converting to dict.

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
        if self.system == "Register".casefold():
            raise Error("abstract class")
        self.database: pandas.DataFrame = getattr(
            self, self.system, None
        )
        if self.database is None:
            raise ValueError(f"invalid system {self.system!r}")
        self.database = self.database.reset_index()
        self._candidate = None

        # Root names
        self.root_names = self.roots = [*self.database.columns]

        # Searching
        self.locname_field = None
        self.search_mode = None
        self.search_keywords = {}
        self.found_results = False
        self.unpacked = False
        self.linked = False
        self.current_row = pandas.DataFrame()
        self._results = EntryGroup(self)

        # Transferring
        self.transfer_target = None

        # Building links
        self.entry_helper = {}

        # Caching
        self.cache = {}
        self.store = self.cache.update

        if link:
            self.link_manager = GenericLinkManager(
                dict_link_managers=DictLinkManagers(self),
                frame_link_managers=Register.frame_link_managers,
                cache=self.cache,
                klass=self
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
        if self._failure():
            not_found_exctype = error_types[self.system.lower()]
            if self.raise_for_failure:
                raise not_found_exctype('no results found')
            self.__init__()
        else:
            self.found_results = True
            self._results = EntryGroup(
                self, self._candidate.reset_index())
            self._results.frame = self._results.frame.drop(columns=["level_0"])
            if (len(self.r) == 1 or self.force_unpack) and self.unpack_mode:
                return self.unpack_row()
        return self

    @final
    def _failure(self):
        return self._candidate.empty or self._candidate.equals(self.database)

    @final
    def ensure_field(self, name):
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

        Returns
        -------
        Entry
        """
        dataframe = self.database.loc[self.database["index"] == int(i)]
        return self.unpack_row(dataframe, link=link)

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
            filter(lambda x: str(x) != 'nan', self.link_fields)
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

    r = results

    @set_sentinel(sentinel.search)
    def search(
            self,
            *args,  # noqa for autocompletion
            **keywords  # noqa for autocompletion
    ) -> Union["Entry", "Register"]:
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
           index WOJ POW GMI RODZ_GMI  RM MZ   NAZWA      SYM   SYMPOD     STAN_NA
        0  11907  06  11  06        2  01  1  Poznań  0686397  0686397  2021-01-01
        1  76796  24  10  03        2  00  1  Poznań  0217047  0216993  2021-01-01
        2  95778  30  64  01        1  96  1  Poznań  0969400  0969400  2021-01-01

        >>> s.search("Poznań", woj="06")

        Returns
        -------
        Union[Entry, Register]
            Entry, if one most accurate entry was found, otherwise Register
            if there were many results or no.

        """
        #
        # TODO: Unit, Locality and Street objects should be also legal
        #       as search keywords. (09-02-2021)
        #
        self._candidate = Search(
            dataframe=self.database,
            system=self.system,
            search_mode=self.search_mode,
            fields=self.fields,
            case=self.case,
            locname=self.locname_field,
            by_possession=self.by_possession,
            by_prefix=self.by_prefix
        )(search_keywords=self.search_keywords)

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

    @set_sentinel(sentinel.unpack_row)
    def unpack_row(
            self,
            dataframe: pandas.DataFrame = None,
            *,
            link=True
    ) -> "Entry":
        """
        Unpack one-row DataFrame to Entry instance.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            DataFrame with length 1 to unpack.

        link : bool
            Whether to link the linkable values.

        Returns
        -------
        Entry
        """
        self.link_mode = (dataframe, link)[True]
        for field, colname in self.fields.items():
            value = self.current_row.iat[
                0, self.current_row.columns.get_loc(colname)
            ]

            if str(value) != 'nan':
                self.entry_helper[field] = value
                if self.link_manager.has_lm(field) and self.link_mode:
                    name = self.link_manager.link(field, value)
                    index = self.link_manager.link_indexes.get(
                        field, None)
                    if index:
                        self.entry_helper[field] = UnitLink(
                            code=value, name=name, index=index)
                    else:
                        self.entry_helper[field] = Link(
                            code=value, name=name)

        # TODO: move this somewhere else…
        if "integral_id" in self.entry_helper:
            self.entry_helper[
                "integral_func"] = self.link_manager.link(
                "integral", self.entry_helper["integral_id"])

        self.entry_helper["terid"] = self.pack_terid(**self.entry_helper)
        self.unpacked = True
        return entry_types[self.system].__call__(
            system=self,
            **self.entry_helper,
            row=self.current_row,
            index=self.current_row.iat[0, 0]
        )

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


implement_common_data(Register)


class EntryGroupSentinel(object):
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
    sentinel = EntryGroupSentinel()

    def __contains__(self, item):
        return self.frame.__contains__(item)

    def __getattribute__(self, item):
        try:
            return object.__getattribute__(self, item)
        except AttributeError:
            return getattr(self.frame, item)

    def __init__(self, reg, frame=pandas.DataFrame()):
        self.system = reg
        self.frame = frame

    def __len__(self):
        return len(self.frame)

    def __repr__(self):
        return repr(self.frame)

    def get_entry(self, number, link=True):
        dataframe = self.frame.reset_index()
        dataframe = dataframe.loc[dataframe["level_0"] == number]
        dataframe = dataframe.drop(columns=["level_0"])
        return (lambda: dataframe,
                lambda: self.system.__class__().unpack_row
                (dataframe)
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

    @set_sentinel(sentinel.to_keywords)
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
        name = {}
        if self.system.locname_field:
            name = dict(match=re.escape(self.system.locname_field))
        yield dict(
            **name,
            **self.system.search_keywords,
            **self.system.modes
        )
        yield transfer_target

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
        >>> warsaw = simc().search("Warszawa", gmitype="wiejska", woj="kujawsko-pomorskie")
        >>> warsaw
        SIMC()
        Results:
           index WOJ POW GMI RODZ_GMI  RM MZ     NAZWA      SYM   SYMPOD     STAN_NA
        0   4810  04  14  05        2  00  1  Warszawa  1030760  0090316  2021-01-01
        1   5699  04  04  03        2  00  1  Warszawa  0845000  0844991  2021-01-01
        2   5975  04  14  07        2  00  1  Warszawa  0093444  0093438  2021-01-01

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
        dataframe = self.frame
        field = self.system.ensure_field(field)
        require(
            field in self.system.fields,
            f"{field!r} is not a valid field. "
            f"Available fields: "
            f"{', '.join(sorted(self.system.fields.keys()))}"
        )
        new_list = getattr(dataframe, self.system.fields[field]).tolist()
        if link and self.system.link_manager.has_lm(field):
            for key_index in range(len(new_list)):
                new = self.system.__class__()
                entry = new.unpack_row(pandas.DataFrame([
                    dataframe.loc[dataframe.index[key_index]]
                ]))
                new_list[key_index] = getattr(entry, field)

        return new_list

    def transfer(
            self,
            key: Hashable,
            target: Union[str, type],
            _kwt=None,
            **other
    ) -> "Register":
        """ Search :target: system using keywords from this instance. """
        global transfer_collector
        if _kwt:
            keywords, transfer_target = _kwt(target)
        else:
            keywords, transfer_target = self.to_keywords(target)
        pop = transfer_collector.pop(key, ())
        transfer_collector[key] = pop + (self, transfer_target.search(
            **{**keywords, **other}))
        return transfer_collector[key][-1]


class EntrySentinel(object):
    to_keywords = EntryGroupSentinel.to_keywords


@dataclasses.dataclass(frozen=True)
class Entry(object):
    """
    TERYT register entry class.
    """

    system: Register
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
    integral_func: type(lambda: None) = (lambda: None)
    integral_id: str = None
    row: pandas.DataFrame = None
    date: str = None
    index: int = None

    sentinel = EntrySentinel()

    @property
    def is_entry(self):
        return True

    isentry = is_entry

    @property
    def integral(self):
        if self.integral_id:
            return self.integral_func.__call__(id=self.integral_id)

    @property
    def results(self):
        return self.system.results

    r = frame = results

    @set_sentinel(sentinel.to_keywords)
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
        prop_copy = properties.copy()
        for k, v in prop_copy.items():
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

    def transfer(self, key: Hashable, target: Union[str, type], **other) -> "Register":
        return EntryGroup(
            self.system, frame=self.frame
        ).transfer(key, target, self.to_keywords, **other)

    def __getattribute__(self, item):
        # Register is mutable;
        # Entry isn't.
        try:
            return object.__getattribute__(self, item)
        except AttributeError:
            try:
                return object.__getattribute__(self.system, item)
            except AttributeError:
                raise

    def __repr__(self, indent=True):
        joiner = '\n    ' if indent else ''
        # TODO: compress it
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


class SIMC(Register, ABC):
    """ SIMC system. """


class TERC(Register, ABC):
    """ TERC system. """


class ULIC(Register, ABC):
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
        if not issubclass(sys, Register):
            raise err
        most_recent = sys()
    elif isinstance(sys, Register):
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
        raise TypeError("target system must be a Register instance or name, not type")
    keywords = {'target': results.system, **keywords}
    recent = _make_recent(results.system)
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
        raise TypeError("system must be a Register instance")
    return _make_recent(system).ensure_field(root_name)


search.__doc__ = Register.search.__doc__
sys_index.__doc__ = Register.index.__doc__
transfer.__doc__ = Entry.transfer.__doc__ = EntryGroup.transfer.__doc__
to_list.__doc__ = Register.to_list.__doc__
to_dict.__doc__ = Register.to_dict.__doc__
ensure_field.__doc__ = Register.ensure_field.__doc__

terc = Terc = TERC
simc = Simc = SIMC
ulic = Ulic = ULIC

Register.frame_link_managers = FrameLinkManagers()
