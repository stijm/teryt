""" Search TERYT. """

# This is the part of teryt library.
# Author: Stim, 2021
# License: GNU GPLv3

import dataclasses
import pandas
import typing
from warnings import warn
from abc import ABC
from math import factorial
from typing import Union
from .data.manager import (
    simc_data,
    terc_data,
    ulic_data
)
from .tools import (
    require,
    subsequent,
    CaseFoldedSetLikeTuple,
    FrameQuestioner
)
from .exceptions import (
    ErroneousUnitName,
    Error as _Error,
    EntryNotFoundError,
    LocalityNotFound,
    StreetNotFound,
    UnitNotFound,
    UnpackError,
)

systems = CaseFoldedSetLikeTuple((
    'simc', 'Simc', 'SIMC',
    'terc', 'Terc', 'TERC',
    'ulic', 'Ulic', 'ULIC'
))
__all__ = (
    'all_transferred_searches',
    'ensure_value_space',
    'EntryNotFoundError',
    'entry_types',
    'index',
    'search',
    'simc_data',
    'terc_data',
    'ulic_data',
    *systems,
    'to_dict',
    'to_list',
    'transferred_searches',
    'transfer',
)


all_transferred_searches = {}


def transferred_searches(name):
    for transferred_search in set(all_transferred_searches.get(name, ())):
        yield getattr(transferred_search, 'system'), transferred_search


@dataclasses.dataclass(frozen=True)
class Link(object):
    """ TERYT Link. """
    
    id: str
    name: typing.Any

    def __getitem__(self, item: (str, int)):
        return (
            {item: getattr(self, item, '')},
            [*dict(self).values()]
        )[isinstance(item, int)][item]

    def __str__(self):
        return str(self.id or '')

    def __add__(self, other):
        return (str(self.id) + other) if self else ('' + other)

    def __bool__(self):
        return all([self.name, self.id])

    def __iter__(self):
        if self.name:
            yield 'name', self.name
        yield 'ID', self.id
        i = getattr(self, 'index', None)
        if i:
            yield 'index', i


@dataclasses.dataclass(frozen=True)
class UnitLink(Link):
    """ Link to a unit. """
    index: int

    @property
    def as_unit(self):
        if self.index or self.index == 0:
            with terc() as unit_manager:
                return unit_manager.index(self.index)

    as_entry = as_unit


@dataclasses.dataclass(frozen=True)
class LocalityLink(Link):
    """ Link to a locality. """
    index: int

    @property
    def as_loc(self):
        if self.index or self.index == 0:
            with simc() as loc_manager:
                return loc_manager.index(self.index)

    as_entry = as_loc


class Search:
    """
    TERYT searching algorithm class.
    """
    def __init__(
            self,
            *,
            dataframe: pandas.DataFrame,
            field_name: str,
            search_mode: str,
            value_spaces: dict,
            case: bool,
            containers: typing.Iterable,
            startswiths: typing.Iterable,
            locname: str = '',
    ):
        self.dataframe = dataframe
        self.candidate = self.dataframe[:]
        self.frames = [self.candidate]
        self.field_name = field_name
        self.search_mode = search_mode
        self.value_spaces = value_spaces
        self.case = case
        self.search_keywords = {}
        self.ineffective = ''
        self.locname = locname
        self.containers = containers
        self.startswiths = startswiths

    def failure(self):
        return self.candidate.empty or self.candidate.equals(self.dataframe)

    def shuffle(self):
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
        self.search_keywords = search_keywords

        def locname_search():
            """
            Search for locality name.
            """
            self.candidate = getattr(
                FrameQuestioner(self.candidate),
                self.search_mode
            )(col=self.value_spaces['name'],
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
            for value_space, query in self.search_keywords.items():
                if value_space == 'voivodship':
                    query = query.upper()
                attempts += 1
                if value_space not in self.value_spaces:
                    continue
                col = self.value_spaces[value_space]
                keyword_args = dict(
                    col=col,
                    value=str(query),
                    case=self.case
                )
                frame_query = 'name'
                if value_space in self.containers:
                    frame_query = 'contains'
                elif value_space in self.startswiths:
                    frame_query = 'startswith'

                self.candidate = getattr(
                        FrameQuestioner(self.candidate),
                        frame_query
                    )(**keyword_args)

                if self.failure():
                    if self.candidate.equals(self.dataframe):
                        no_uniqueness = f'It seems that all values in ' \
                                        f'{value_space!r} value space ' \
                                        f'are equal to {query!r}. Try ' \
                                        f'using more unique key words.'
                        warn(no_uniqueness, category=UserWarning)
                    self.candidate = self.frames[-1]
                    if attempts <= max_attempts:
                        self.ineffective = value_space
                        self.shuffle()
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


class FrameLinkManagers(object):
    def __init__(self, linkmanager):
        omitter = dict(unpack=False, unpacked=True)
        self.linkmanager = linkmanager
        self.voivodships = linkmanager.search(
            function='województwo',
            **omitter
        ).results
        self.powiats = linkmanager.search(function='powiat', **omitter).results
        self.gminas = linkmanager.search(function='gmina', **omitter).results

    def __repr__(self):
        return f"FrameLinkManagers({self.linkmanager!r})"


class Register(ABC):
    _containers = (
        'function',
    )
    _startswiths = (
        'date',
    )
    _locname_keywords = (
        'name',
        'match',
        'startswith',
        'endswith',
        'contains'
    )
    _optional_bool_arguments = (
        'raise_for_failure',
        'force_unpack',
        'unpack',
        'link',
        'unpacked',
        'case'
    )
    _optional_str_arguments = (
        'terid',
    )
    _boolstr_arguments = (
        *_optional_bool_arguments,
        *_optional_str_arguments
    )
    _erroneus_argument = \
        f'%s() got an unexpected keyword argument %r. ' \
        f'Try looking for the proper argument name ' \
        f'in the following list:\n{" " * 12}%s.'
    _repr_not_str = '%r is not a %s'

    def __init__(self):
        self.simc = simc_data
        self.terc = terc_data
        self.ulic = ulic_data
        self.system = self.__class__.__name__.replace(' ', '_').casefold()
        if self.system == "Register".casefold():
            raise _Error("abstract class")
        self.field: pandas.DataFrame = getattr(
            self, self.system, None
        )
        require(
            self.field is not None,
            f'invalid system {self.system!r}'
        )
        self.field = self.field.reset_index()
        self._candidate = None  # auxiliary
        self._linkable_args = {}
        self._unlinkable_args = {}
        self.link_indexes = {}
        self.cols = [*self.field.columns]
        self.len = len(self.field)
        self.columns = self.cols
        self.conflicts = (self._locname_keywords, ('force_unpack', 'unpack'))
        self.case = None
        self.force_unpack = None
        self.locname_value_space = None
        self.valid_keywords = None
        self.unpack = None
        self.search_keywords = None
        self._search_keywords = None
        self.search_mode = None
        self.raise_for_failure = None
        self._link = True
        self.results_found = False
        self.unpacked = False
        self.linked = False
        self._results = None
        self._transfer_target = None
        self._locality_buf = {}
        self.cache = {}
        self.store = self.cache.update
        if not hasattr(self, 'frames'):
            Register.frames = FrameLinkManagers(self)

    @property
    def results(self):
        return self._results

    def __repr__(self):
        return f"{self.system.upper()}()" +\
               (f"\n{''.center(len(self.system) + 2, '-')} "
                f"Results:\n{self.results}"
                if self.results_found else "")

    @property
    def link_spaces(self):
        """ Spaces to be linked. """
        raise NotImplementedError

    @property
    def value_spaces(self):
        """ Value spaces. """
        raise NotImplementedError

    def unique_value_space(self, value_space):
        """
        Return if a value space is unique in comparison
        to other systems.
        """
        other_value_spaces = set(
            filter(lambda: not self.field.equals,
                   [self.simc, self.terc, self.ulic])
        )

        return all([value_space in self.value_spaces,
                    value_space not in other_value_spaces])

    def __iter__(self):
        if self._locality_buf['terid']:
            yield 'terid', self._locality_buf['terid']
        for value_space in self.value_spaces:
            if self._locality_buf.get(value_space, ""):
                yield value_space, self._locality_buf.get(value_space, "")

    def __getitem__(self, item):
        return dict(self)[item]

    def __del__(self):
        # self.__init__()
        # -1073741571 (0xC00000FD)
        pass

    clear = __del__

    def _has_dict_linkmanager(self, value_space: str) -> "bool":
        return hasattr(self, value_space + '_linkmanager')

    def _has_data_linkmanager(self, value_space: str) -> "bool":
        return hasattr(getattr(self, "frames"), value_space + 's')

    def _has_linkmanager(self, value_space: str) -> "bool":
        return self._has_dict_linkmanager(
            value_space
        ) or self._has_data_linkmanager(
            value_space
        )

    def _on_keywords(self, arguments, _kwds):
        require(arguments, 'as_keywords(): no arguments')
        target_name = arguments[0]
        require(
            target_name in systems,
            f'cannot evaluate transfer target using'
            f' name {target_name!r}'
        )
        self._transfer_target = \
            eval(f'{target_name!s}')
        self._transfer_target = self._transfer_target()
        require(
            self.unpacked,
            'cannot perform generating keywords from '
            'properties if search results were not unpacked'
        )

    def _on_link_names(self, _args, keywords):
        valid_keywords = sorted(self.value_spaces.keys())
        [require(
            keyword in valid_keywords,
            TypeError(self._erroneus_argument % (
                '_link_names', keyword, ', '.join(valid_keywords)
            ))
        ) for keyword in list(set(keywords))]

        for name, value in keywords.items():
            if self._has_data_linkmanager(name):
                self._linkable_args.update({name: value})
                continue
            elif self._has_dict_linkmanager(name):
                linkmanager = getattr(self, name + '_linkmanager')
                require(
                    value in linkmanager,
                    self._repr_not_str % (
                        value,
                        f'valid \'{name.replace("_", " ")}'
                        f'\' non-ID value'
                    )
                )
                value = linkmanager[value]
            self._unlinkable_args.update({name: value})

    def _on_link(self, arguments, _kwargs):
        if arguments:
            value_space = next(iter(arguments[:]))
            require(
                self._has_linkmanager(value_space),
                f'value space {value_space!r}'
                f' does not have a name-ID link'
            )

    def _on_unpack_entry(self, _args, keywords):
        dataframe = keywords.pop("dataframe", self.results)
        value_spaces = self.value_spaces
        require(
            dataframe is not None,
            UnpackError('nothing to unpack from')
        )
        require(
            not dataframe.empty,
            UnpackError('nothing to unpack from')
        )
        require(
            len(dataframe) == 1,
            UnpackError(
                f'cannot unpack from more '
                f'than one TERYT entry (got {len(dataframe)} entries)'
            )
        )
        for value_space in value_spaces:
            require(
                value_spaces[value_space] in dataframe,
                UnpackError(
                    f'value space '
                    f'{value_space.replace("_", " ")} '
                    f'(the real column is named {value_spaces[value_space]!r})'
                    f' not in source DataFrame'
                )
            )
        self._entry_data = dataframe

    def _on_search(self, arguments, keywords):
        self.clear()

        if len(arguments) == 1:
            argument = next(iter(arguments[:]))
            if not isinstance(argument, str):
                raise TypeError(f"name must be str, "
                                f"not {argument.__class__.__name__}")
            if "name" in keywords:
                raise ValueError("passed multiple values for 'name' parameter")
            keywords["name"] = argument
            arguments = ()

        require(
            not arguments,
            ValueError(
                'cannot perform searching: '
                'only one positional argument can be accepted'
            )
        )
        require(
            keywords,
            ValueError(
                'cannot perform searching: '
                'no keyword arguments'
            )
        )

        self._search_keywords = tuple(
            set(self._locname_keywords + (*self.value_spaces.keys(),))
        ) + self._optional_str_arguments
        self._valid_keywords = self._search_keywords + self._boolstr_arguments
        self.unpacked = keywords.copy().pop('unpacked', False)

        keywords = dict(map(
            lambda kv: (self.ensure_value_space(kv[0]), kv[1]),
            keywords.items())
        )

        require(
            any(map(keywords.__contains__, self._search_keywords)),
            ValueError(
                f'no keyword arguments for searching '
                f'(expected at least one from: '
                f'{", ".join(sorted(self._search_keywords))}'
                f')'
            )
        )

        self.conflicts += tuple(map(lambda x: ('terid', x), self.link_spaces))

        for conflicted in self.conflicts:
            conflict = []
            for argument in conflicted:
                if argument in keywords:
                    require(
                        not conflict,
                        'setting more than one keyword argument '
                        'from %s in one search is impossible' %
                        ' and '.join(
                            map('%s'.__mod__,
                                sorted(conflicted))
                        )
                    )
                    conflict.append(argument)

        self.locname_value_space, self.force_unpack = None, True
        modes = self._locname_keywords + ('no_locname',)
        self.search_mode = modes[-1]

        for keyword in keywords.copy():
            for mode in self._locname_keywords:
                if keyword == mode:
                    self.search_mode = mode
                    self.locname_value_space = keywords[mode]
                    del keywords[mode]

        self.raise_for_failure = keywords.pop('raise_for_failure', False)
        self.unpack = keywords.pop('unpack', True)
        self._link = keywords.pop('link', True)
        self.force_unpack = keywords.pop('force_unpack', False)
        self.unpacked = keywords.pop('unpacked', False)
        self.case = keywords.pop('case', False)
        terid = keywords.pop('terid', '')

        if not self.unpacked:
            keywords = self._link_names(**keywords)
        if terid:
            unpacked = self.unpack_terid(terid)
            [keywords.__setitem__(n, v) for n, v in unpacked.items() if v]

        self.search_keywords = keywords
        self._candidate = self.field.copy()

        for value_space in self.search_keywords:
            column = self.value_spaces[value_space]
            self.field[column] = self.field[
                self.value_spaces[value_space]
            ].map(str)  # map all to strings

    @subsequent(_on_link_names)
    def _link_names(self, **_kwargs):
        id_keywords = self._unlinkable_args.copy()
        for value_space in self._unlinkable_args.keys():
            if value_space not in self.terc.columns:
                id_keywords.pop(value_space)

        spaces = [*self.value_spaces.keys()]
        for value_space, value in self._linkable_args.items():
            if value_space == 'voivodship':
                value = value.upper()

            space_linkmanager = value_space + 's'
            id_dataframe = getattr(getattr(self, "frames"), space_linkmanager)

            if id_dataframe.empty:
                warn(
                    f'no name-ID link available for {space_linkmanager}. '
                    f'Updating search keywords with the provided value, '
                    f'however results are possible not to be found if it '
                    f'is not a valid ID.'
                )
                id_keywords.update({value_space: value})
                continue

            entry = Search(
                dataframe=id_dataframe,
                field_name="terc",
                search_mode="equal",
                value_spaces=terc.value_spaces,
                case=False,
                containers=self._containers,
                startswiths=self._startswiths,
                locname=value
            )(search_keywords=id_keywords)

            require(not entry.empty,
                    ErroneousUnitName(self._repr_not_str % (
                        value, value_space)))
            index = entry.iat[0, 0]  # noqa
            link_result = {value_space: entry.iat[0, entry.columns.get_loc(
                self.value_spaces[value_space]
            )]}
            id_keywords.update(link_result)
            self.link_indexes[value_space] = index

            if value_space != spaces[0]:
                quantum = spaces.index(value_space) - 1
                for rot in range(quantum + 1):
                    prev = spaces[quantum - rot]
                    id_keywords[prev] = entry.iat[
                        0, entry.columns.get_loc(self.value_spaces[prev])
                    ]

        return {**id_keywords, **self._unlinkable_args}

    def _dispatch_results(self):
        if self._failure():
            not_found_exctype = error_types[self.system.lower()]
            require(
                not self.raise_for_failure,
                not_found_exctype('no results found')
            )
            self.__del__()
        else:
            self.results_found = True
            self._results: pandas.DataFrame = self._candidate.reset_index()
            self._results = self._results.drop(columns=["level_0"])
            if (len(self._candidate
                    ) == 1 or self.force_unpack) and self.unpack:
                return self.unpack_entry()
        return self

    @subsequent(_on_link)
    def link(self, value_space: str, value: str):
        """
        Resolve locality that value in value space refers to,
        creating and returning a Link instance.

        Parameters
        ----------
        value_space : str
            Value space of :value:.

        value : value
            Value to link.

        Returns
        -------
        dict or str
        """

        if self._has_dict_linkmanager(value_space):
            return dict(map(
                lambda pair: (pair[1], pair[0]),
                getattr(self, value_space + '_linkmanager').items()
            ))[value]

        linkmanager = terc()

        if value_space == 'integral_id':  # special case
            new = simc()
            new.search("integral_id")

        if any(
                [value_space not in self.link_spaces,
                 str(value) == 'nan']
        ):
            return ''

        keywords = {'function': self.value_spaces[value_space]}
        spaces = [*self.value_spaces.keys()]

        if value_space != spaces[0]:
            quantum = spaces.index(value_space) - 1
            for rot in range(quantum + 1):
                prev_value = spaces[quantum - rot]
                keywords[prev_value] = str(self._locality_buf[prev_value])

        keywords[value_space] = value

        if value_space != spaces[-1]:
            next_value = spaces[spaces.index(value_space) + 1]
            keywords[next_value] = 'nan'

        if tuple(keywords.items()) in self.cache:
            return self.cache[tuple(keywords.items())]

        result = Search(
            dataframe=linkmanager.field,
            field_name=linkmanager.system,
            search_mode='no_locname',
            value_spaces=linkmanager.value_spaces,
            case=False,
            containers=linkmanager._containers,
            startswiths=linkmanager._startswiths,
        )(search_keywords=keywords)

        name = result.iat[
            0, result.columns.get_loc(linkmanager.value_spaces['name'])
        ]
        self.link_indexes[value_space] = result.iat[0, 0]
        self.store({tuple(keywords.items()): name})
        return name

    def _failure(self):
        return self._candidate.empty or self._candidate.equals(self.field)

    @subsequent(_on_keywords)
    def as_keywords(self, _transfer_target_name: str):
        """
        Create and return keywords leading to current search results.

        Parameters
        ----------
        _transfer_target_name: str
            Target class (simc, terc or ulic).

        Returns
        -------
        generator
        """
        transfer_target = self._transfer_target
        properties = dict(self)
        name_space_value = properties.pop('name')
        prop_copy = properties.copy()
        [
            properties.__setitem__(k, str(v))
            if k in transfer_target.value_spaces and str(v)
            else properties.__delitem__(k)
            for k, v in prop_copy.items()
        ]
        keywords = {
            **properties,
            'unpacked': True,
            'name': name_space_value,
            'raise_for_failure': self.raise_for_failure,
            'case': self.case
        }
        yield keywords
        yield transfer_target

    def ensure_value_space(self, column_name):
        """
        Find :column_name: in value spaces and return it if it occurs.

        Parameters
        ----------
        column_name

        Returns
        -------
        str
        """
        spaces = self.value_spaces
        if column_name in spaces:
            return column_name
        column_name_upper = column_name.upper()  # columns are upper
        return dict([*map(reversed, spaces.items())]).get(
            column_name_upper, column_name
        )

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
        dataframe = self.field.loc[self.field["index"] == int(i)]
        self._link = link
        return self.unpack_entry(dataframe=dataframe)

    @property
    def is_entry(self):
        """
        Check if this instance is of Entry class.

        Returns
        -------
        bool
        """
        return False

    is_unit = is_loc = is_street = is_entry

    def pack_terid(self, **info) -> "str":
        """
        Pack information into teritorial ID.

        Parameters
        ----------
        **info
            Keyword arguments consisting of linkable value spaces.

        Returns
        -------
        str
        """
        return ''.join(map(str, map(
            lambda space: info.get(space, ""),
            filter(lambda x: str(x) != 'nan', self.link_spaces)
        )))

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
        require(
            teritorial_id,
            'teritorial ID to be unpacked cannot'
            ' be an empty string')
        code_keywords = {}
        frames = {}
        max_len = sum(self.link_spaces.values())
        require(
            len(teritorial_id) <= max_len,
            f'{self.system.upper()} teritorial ID length'
            f' is expected to be maximally {max_len}'
        )
        i = 0

        for linkmanager_value_space, valid_length in self.link_spaces.items():
            if i >= len(teritorial_id) - 1:
                break
            frames.update(
                {linkmanager_value_space: getattr(
                    getattr(self, "frames"), linkmanager_value_space + 's'
                )}
            )
            partial = teritorial_id[i:i + valid_length]
            unpack = self.unpack
            if errors:
                require(
                    not self.search(  # noqa
                        unpacked=True,
                        unpack=False,
                        **{linkmanager_value_space: partial}
                    ).results.empty,
                    'unpack_terid(…, errors=True, …): ' +
                    self._repr_not_str % (
                        '…' + partial,
                        f'valid teritorial ID part '
                        f'(error at {linkmanager_value_space!r} value space'
                        f', column '
                        f'{self.value_spaces[linkmanager_value_space]!r})'
                    )
                )
            self.unpack = unpack
            code_keywords.update({linkmanager_value_space: partial})
            i += valid_length

        return code_keywords

    @subsequent(_on_unpack_entry)
    def unpack_entry(self, *, dataframe: pandas.DataFrame = None) -> "Entry":  # noqa
        """
        Unpack one-row DataFrame to Entry instance.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            DataFrame to unpack.

        Returns
        -------
        Entry
        """
        for value_space, colname in self.value_spaces.items():
            value = self._entry_data.iat[
                0, self._entry_data.columns.get_loc(colname)
            ]

            if str(value) != 'nan':
                self._locality_buf[value_space] = value
                if self._has_linkmanager(value_space) and self._link:
                    id = value  # noqa
                    name = self.link(value_space, value)
                    i = self.link_indexes.get(value_space, None)
                    if i:
                        self._locality_buf[value_space] = UnitLink(
                            id=id, name=name, index=i)
                    else:
                        self._locality_buf[value_space] = Link(
                            id=id, name=name)

        self._locality_buf['terid'] = self.pack_terid(**self._locality_buf)
        self.unpacked = True
        return entry_types[self.system].__call__(
            system=self,
            **self._locality_buf,
            entry_data=self._entry_data,
            index=self._entry_data.iat[0, 0]
        )

    @subsequent(_on_search)
    def search(self, *_args, **_kwargs) -> Union["Register", "Locality"]:
        """
        Search for the most accurate entry using provided keywords.

        Parameters
        ----------
        *_args
            Positional arguments.
            Only one positional argument is accepted: as an equivalent
            to "name=" keyword parameter.

        **_kwargs
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
            Gmina ()
        voivodship : str
            voivodship
        function : str
            function
        streettype : str
            streettype
        powiat : str
            powiat
        cnowner : str
            cnowner
        id : str
            id
        integral_id : str
            integral_id
        gmitype : str
            gmitype

        Examples
        --------
        >>> s = simc()
        >>> s.search("Poznań")
        SIMC()
        ------ Results:
           index WOJ POW GMI RODZ_GMI  RM MZ   NAZWA      SYM   SYMPOD     STAN_NA
        0  11907  06  11  06        2  01  1  Poznań  0686397  0686397  2021-01-01
        1  76796  24  10  03        2  00  1  Poznań  0217047  0216993  2021-01-01
        2  95778  30  64  01        1  96  1  Poznań  0969400  0969400  2021-01-01

        >>> s.search(s.as_keywords)

        Returns
        -------
        Entry
            If one most accurate entry was found.

        Search
            If there were many results.

        """
        self._candidate = Search(
            dataframe=self.field,
            field_name=self.system,
            search_mode=self.search_mode,
            value_spaces=self.value_spaces,
            case=self.case,
            locname=self.locname_value_space,
            containers=self._containers,
            startswiths=self._startswiths
        )(search_keywords=self.search_keywords)
        return self._dispatch_results()

    def to_list(self, value_space: str, link: bool = True) -> list:  # noqa
        """
        Return list of all values in :value space:.

        Parameters
        ----------
        value_space : str
            Value space to retrieve values of.

        link : bool
            Whether to link the linkable values. Defaults to True.

        Examples
        --------
        >>> warsaw = simc().search("Warszawa", gmitype="wiejska", woj="kujawsko-pomorskie")  # noqa
        >>> warsaw
        SIMC()
        ------ Results:
           index WOJ POW GMI RODZ_GMI  RM MZ     NAZWA      SYM   SYMPOD     STAN_NA
        0   4810  04  14  05        2  00  1  Warszawa  1030760  0090316  2021-01-01
        1   5699  04  04  03        2  00  1  Warszawa  0845000  0844991  2021-01-01
        2   5975  04  14  07        2  00  1  Warszawa  0093444  0093438  2021-01-01

        >>> warsaw.to_list("sym")
        ['1030760', '0845000', '0093444']

        >>> warsaw.to_list("powiat")  # equivalent to warsaw.to_list("pow")
        [UnitLink(id='14', name='świecki', index=469),
         UnitLink(id='04', name='chełmiński', index=358),
         UnitLink(id='14', name='świecki', index=469)]

        >>> warsaw.to_list("powiat", link=False)
        ['14', '04', '14']

        Returns
        -------
        list
        """
        dataframe = (
            self.field.copy(),
            ((self.results, {})[self.results is None]).copy(),
        )[self.results_found]
        value_space = self.ensure_value_space(value_space)
        require(
            value_space in self.value_spaces,
            f'{value_space!r} is not a valid value space.'
            f' Available value spaces: '
            f'{", ".join(sorted(self.value_spaces.keys()))}'
        )
        new_list = getattr(dataframe, self.value_spaces[value_space]).tolist()
        if link and self._has_linkmanager(value_space):
            for key_index in range(len(new_list)):
                new = self.__class__()
                entry = new.unpack_entry(dataframe=pandas.DataFrame([
                    dataframe.loc[dataframe.index[key_index]]
                ]))
                print(entry)
                new_list[key_index] = getattr(entry, value_space)

        return new_list

    def to_dict(self, link: bool = True) -> "dict":  # noqa
        results = self.results.copy()  # don't lose the current results
        new_dict = {}
        for value_space in self.value_spaces:
            value = [*self.to_list(value_space, link=link)]
            new_dict.update(
                {value_space: value[0] if len(value) == 1 else value}
            )
        self.clear()
        self._results = results
        return new_dict

    def transfer(self, target: str, **other) -> "Register":
        global all_transferred_searches
        keywords, transfer_target = self.as_keywords(target)
        name = keywords['name']
        pop = all_transferred_searches.pop(name, ())
        all_transferred_searches[name] = pop + (
            self,
            transfer_target.search(
                **{**{vs: v for vs, v in other.items()
                      if any(
                        [vs in transfer_target.value_spaces,
                         vs in getattr(transfer_target, '_boolstr_arguments')]
                    )}, **keywords}
            ))
        return all_transferred_searches[name][-1]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            raise

    gmitype_linkmanager = {
        'miejska': '1',
        'gmina miejska': '1',
        'wiejska': '2',
        'gmina wiejska': '2',
        'miejsko-wiejska': '3',
        'gmina miejsko-wiejska': '3',
        'miasto w gminie miejsko-wiejskiej': '4',
        'obszar wiejski w gminie miejsko-wiejskiej': '5',
        'dzielnice m. st. Warszawy': '8',
        'dzielnice Warszawy': '8',
        'dzielnica Warszawy': '8',
        'dzielnica': '8',
        'delegatury w miastach: Kraków, Łódź, Poznań i Wrocław': '9',
        'delegatura': '9'
    }


@dataclasses.dataclass(frozen=True)
class Entry(object):
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
    integral: str = None
    integral_id: str = None
    entry_data: pandas.DataFrame = None
    date: str = None
    index: int = None

    @property
    def is_entry(self):
        return True

    def __getattribute__(self, item):
        # Reason:
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
    type = "Unit"

    @property
    def is_unit(self):
        return True


class Locality(Entry):
    type = "Locality"

    @property
    def is_loc(self):
        return True


class Street(Entry):
    type = "Street"

    @property
    def is_street(self):
        return True


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


class SIMC(Register):
    value_spaces = {
        'voivodship': 'WOJ',
        'powiat': 'POW',
        'gmina': 'GMI',
        'gmitype': 'RODZ_GMI',
        'loctype': 'RM',
        'cnowner': 'MZ',
        'name': 'NAZWA',
        'id': 'SYM',
        'integral_id': 'SYMPOD',
        'date': 'STAN_NA'
    }
    link_spaces = {
        'voivodship': 2,
        'powiat': 2,
        'gmina': 2,
        'gmitype': 1
    }
    cnowner_linkmanager = {
        True: '1',
        False: '0'
    }

    # This is WMRODZ, in a dict…
    loctype_linkmanager = {
        'miasto': '96',
        'delegatura': '98',
        'dzielnica m. st. Warszawy': '95',
        'część miasta': '99',
        'wieś': '01',
        'przysiółek': '03',
        'kolonia': '02',
        'osada': '04',
        'osada leśna': '05',
        'osiedle': '06',
        'schronisko turystyczne': '07',
        'część miejscowości': '00',
    }


class TERC(Register):
    value_spaces = {
        'voivodship': 'WOJ',
        'powiat': 'POW',
        'gmina': 'GMI',
        'gmitype': 'RODZ',
        'name': 'NAZWA',
        'function': 'NAZWA_DOD',
        'date': 'STAN_NA',
    }
    link_spaces = {
        'voivodship': 2,
        'powiat': 2,
        'gmina': 2,
        'gmitype': 1
    }


class ULIC(Register):
    value_spaces = {
        'voivodship': 'WOJ',
        'powiat': 'POW',
        'gmina': 'GMI',
        'gmitype': 'RODZ_GMI',
        'integral_id': 'SYM',
        'id': 'SYM_UL',
        'streettype': 'CECHA',
        'name': 'NAZWA_1',
        'secname': 'NAZWA_2',
        'date': 'STAN_NA'
    }
    link_spaces = {
        'voivodship': 2,
        'powiat': 2,
        'gmina': 2,
        'gmitype': 1
    }


inst = None
""" Most recent instance. """


def make_recent(sys,
                err=ValueError("system must be a valid "
                               "TERYT system name, "
                               "instance or type; "
                               "no proper was found/provided")):
    global inst
    if sys is None:
        if inst is None:
            raise err
    elif isinstance(sys, str) and sys in systems:
        inst = eval(sys)()
    elif isinstance(sys, type):
        if not issubclass(sys, Register):
            raise err
        inst = sys()
    elif isinstance(sys, Register):
        inst = sys.__class__()
    else:
        raise err
    return inst


def reset_recent():
    global inst
    inst = None


def search(name=None, *, system=None, **keywords):
    if name is not None:
        keywords["name"] = name
    return make_recent(system).search(**keywords)


def index(i, system=None, **params):
    return make_recent(system).index(i, **params)


def transfer(target, system=None, **keywords):
    if isinstance(system, type):
        raise TypeError("system must be a Register instance")
    keywords = {'target': target, **keywords}
    return make_recent(system).transfer(**keywords)


def to_list(system=None, **params):
    return make_recent(system).to_list(**params)


def to_dict(system=None, **params):
    return make_recent(system).to_dict(**params)


def ensure_value_space(column_name, system=None):
    if isinstance(system, type):
        raise TypeError("system must be a Register instance")
    return make_recent(system).ensure_value_space(column_name)


search.__doc__ = Register.search.__doc__
index.__doc__ = Register.index.__doc__
transfer.__doc__ = Register.transfer.__doc__
to_list.__doc__ = Register.to_list.__doc__
to_dict.__doc__ = Register.to_dict.__doc__
ensure_value_space.__doc__ = Register.ensure_value_space.__doc__


terc = Terc = TERC
simc = Simc = SIMC
ulic = Ulic = ULIC

terc()
simc()
