""" Search TERYT. """

# This is the part of teryt library.
# Author: Stim (stijm), 2021
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
    set_sentinel,
    StringCaseFoldedSetLikeTuple,
    FrameQuestioner
)
from .exceptions import (
    ErroneousUnitName,
    Error,
    LocalityNotFound,
    StreetNotFound,
    UnitNotFound,
    UnpackError,
)

systems = StringCaseFoldedSetLikeTuple(('simc', 'terc', 'ulic'))
transfer_collector = {}


def transferred_searches(name):
    for transferred_search in set(transfer_collector.get(name, ())):
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


class Search(object):
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
            by_possession: typing.Iterable,
            by_prefix: typing.Iterable,
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
        self.by_possession = by_possession
        self.by_prefix = by_prefix

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
                if value_space in self.by_possession:
                    frame_query = 'contains'
                elif value_space in self.by_prefix:
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


class GenericLinkManagerSentinel(object):
    def __init__(self):
        self.linkable_args = {}
        self.unlinkable_args = {}

    @staticmethod
    def link(klass, arguments, _kwargs):
        if arguments:
            value_space = next(iter(arguments[:]))
            require(
                klass.has_lm(value_space),
                f'value space {value_space!r}'
                f' does not have a name-ID link'
            )
    
    def link_names(self, klass, _args, keywords):
        expected_kw = sorted(klass.klass.value_spaces.keys())
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
                linkmanager = getattr(self, name + '_linkmanager')
                if value not in linkmanager:
                    e = f'{value!r} is not a valid ' \
                        f'{name.replace("_", " ")!r} non-ID value'
                    raise ValueError(e)
                value = linkmanager[value]
            self.unlinkable_args.update({name: value})


@dataclasses.dataclass(init=False)
class GenericLinkManager(object):
    klass: "Register"
    dict_linkmanagers: "DictLinkManagers"
    frame_linkmanagers: "FrameLinkManagers"
    cache: dict

    sentinel = GenericLinkManagerSentinel()

    def __init__(self, klass, dict_linkmanagers, frame_linkmanagers, cache):
        self.klass = klass
        self.dict_linkmanagers = dict_linkmanagers
        self.frame_linkmanagers = frame_linkmanagers
        self.cache = cache
        self.store = self.cache.update
        self.link_indexes = {}

    @set_sentinel(sentinel.link_names)
    def link_names(self, **_kwargs):
        linkable = self.sentinel.linkable_args
        unlinkable = self.sentinel.unlinkable_args
        extract = unlinkable.copy()
        for value_space in unlinkable.keys():
            if value_space not in self.klass.terc.columns:
                extract.pop(value_space)

        spaces = list(self.klass.value_spaces.keys())
        for value_space, value in linkable.items():
            # TODO: implement a dict with all value spaces mappers
            if value_space == 'voivodship':
                value = value.upper()

            frame_lmname = value_space + 's'
            frame_linkmanager = getattr(self.frame_linkmanagers, frame_lmname)

            if frame_linkmanager.empty:
                warn(
                    f'no name-ID links available for {frame_lmname}. '
                    f'Updating search keywords with the provided value, '
                    f'however results are possible not to be found if it '
                    f'is not a valid ID.'
                )
                unlinkable.update({value_space: value})
                continue

            entry = Search(
                dataframe=frame_linkmanager,
                field_name="terc",
                search_mode="equal",
                value_spaces=terc.value_spaces,
                case=False,
                by_possession=terc.by_possession,
                by_prefix=terc.by_prefix,
                locname=value
            )(search_keywords=unlinkable)

            require(not entry.empty,
                    ErroneousUnitName(f"{value!r} is not a {value_space}"))
            index = entry.iat[0, 0]  # noqa
            link_result = {value_space: entry.iat[0, entry.columns.get_loc(
                self.klass.value_spaces[value_space]
            )]}
            unlinkable.update(link_result)
            self.link_indexes[value_space] = index

            if value_space != spaces[0]:
                quantum = spaces.index(value_space) - 1
                for rot in range(quantum + 1):
                    prev = spaces[quantum - rot]
                    unlinkable[prev] = entry.iat[
                        0, entry.columns.get_loc(self.klass.value_spaces[prev])
                    ]

        return dict(**extract, **unlinkable)

    def has_dict_lm(self, value_space: str) -> "bool":
        return hasattr(self.dict_linkmanagers, value_space + '_linkmanager')

    def has_frame_lm(self, value_space: str) -> "bool":
        return hasattr(self.frame_linkmanagers, value_space + 's')

    def has_lm(self, value_space: str) -> "bool":
        return self.has_dict_lm(
            value_space
        ) or self.has_frame_lm(
            value_space
        )

    @set_sentinel(sentinel.link)
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
        str
        """

        if self.has_dict_lm(value_space):
            return dict(map(
                lambda pair: (pair[1], pair[0]),
                getattr(self.dict_linkmanagers, 
                        value_space + '_linkmanager').items()))[value]

        if value_space == 'integral_id':  # special case
            new = simc()
            integral = new.search("integral_id")
            return integral

        unit_linkmanager = terc()
        
        if value_space not in self.klass.link_spaces or str(value) == 'nan':
            return ''

        keywords = {'function': self.klass.value_spaces[value_space]}
        spaces = list(self.klass.value_spaces.keys())
        helper = self.klass.entry_helper

        if value_space != spaces[0]:
            quantum = spaces.index(value_space) - 1
            for rot in range(quantum + 1):
                prev_value = spaces[quantum - rot]
                keywords[prev_value] = str(helper[prev_value])

        keywords[value_space] = value

        if value_space != spaces[-1]:
            next_value = spaces[spaces.index(value_space) + 1]
            keywords[next_value] = 'nan'

        if tuple(keywords.items()) in self.cache:
            return self.cache[tuple(keywords.items())]

        result = Search(
            dataframe=unit_linkmanager.field,
            field_name=unit_linkmanager.system,
            search_mode='no_locname',
            value_spaces=unit_linkmanager.value_spaces,
            case=False,
            by_possession=unit_linkmanager.by_possession,
            by_prefix=unit_linkmanager.by_prefix,
        )(search_keywords=keywords)

        name = result.iat[
            0, result.columns.get_loc(unit_linkmanager.value_spaces["name"])
        ]
        self.link_indexes[value_space] = result.iat[0, 0]
        self.store({tuple(keywords.items()): name})
        return name

    def __getattribute__(self, item):
        try:
            return object.__getattribute__(self, item)
        except AttributeError:
            if item.endswith("s"):
                return getattr(self.frame_linkmanagers, item)
            elif item.endswith("_linkmanager"):
                return getattr(self.dict_linkmanagers, item)
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
           set(klass.locname_keywords + (*klass.value_spaces.keys(),))
        ) + klass.optional_str_arguments

        keywords = dict(map(  # woj -> voivodship
            lambda kv: (klass.ensure_value_space(kv[0]), kv[1]),
            keywords.items())
        )

        if not any(map(keywords.__contains__, search_keywords)):
            raise ValueError(
                f'no keyword arguments for searching '
                f'(expected at least one from: '
                f'{", ".join(sorted(search_keywords))}'
                f')'
            )

        klass.conflicts += tuple(
            map(lambda ls: ('terid', ls), klass.link_spaces))

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

        klass.locname_value_space, klass.force_unpack = None, True
        modes = klass.locname_keywords + ('no_locname',)
        klass.search_mode = modes[-1]

        for keyword in keywords.copy():
            for mode in klass.locname_keywords:
                if keyword == mode:
                    klass.search_mode = mode
                    klass.locname_value_space = keywords[mode]
                    del keywords[mode]

        klass.raise_for_failure = keywords.pop('raise_for_failure', False)
        klass.unpack = keywords.pop('unpack', klass.unpack)
        klass._link = keywords.pop('link', True)
        klass.force_unpack = keywords.pop('force_unpack', klass.force_unpack)
        klass.unpacked = keywords.pop('unpacked', False)
        klass.case = keywords.pop('case', False)
        terid = keywords.pop('terid', '')

        if not klass.unpacked:
            klass.link_manager.erroneous_argument = klass.erroneous_argument
            keywords = klass.link_manager.link_names(**keywords)
        if terid:
            unpacked = klass.unpack_terid(terid)
            [keywords.__setitem__(n, v) for n, v in unpacked.items() if v]

        klass.search_keywords = keywords
        klass._candidate = klass.field[:]

        for value_space in klass.search_keywords:
            column = klass.value_spaces[value_space]
            klass.field[column] = klass.field[
               klass.value_spaces[value_space]
            ].map(str)

    @staticmethod
    def to_keywords(klass, arguments, _kwds):
        require(arguments, 'to_keywords(): no arguments')
        target_name = arguments[0]
        if target_name in list(map(eval, systems)):
            target_name = target_name.__name__
        if target_name not in systems:
            raise ValueError(
                f'cannot evaluate transfer target using name {target_name!r}')

        klass.transfer_target = eval(target_name)()
        require(
            klass.unpacked,
            'cannot perform generating keywords from '
            'properties if search results were not unpacked'
        )

    @staticmethod
    def unpack_row(klass, _args, keywords):
        row = keywords.pop("dataframe", klass.results)
        value_spaces = klass.value_spaces
        if row is None:
            raise UnpackError('nothing to unpack from')
        if row.empty:
            raise UnpackError('nothing to unpack from')
        if len(row) != 1:  # it's not a row then
            raise UnpackError(
                'cannot unpack from more '
                'than one TERYT entry '
                f'(got {len(row)} entries)'
            )
        for value_space in value_spaces:
            if value_spaces[value_space] not in row:
                raise UnpackError(
                    f'value space '
                    f'{value_space.replace("_", " ")} '
                    f'(the real column is named'
                    f' {value_spaces[value_space]!r})'
                    f' not in source DataFrame'
                )
        klass.row = row


class DictLinkManagers(object):
    def __init__(self, register):
        for attr in filter(lambda a: a.endswith("linkmanager"), dir(register)):
            setattr(self, attr, getattr(register, attr))


class FrameLinkManagers(object):
    def __init__(self):
        omitter = dict(unpack=False, unpacked=True)
        self.manager = terc(link_manager=False)
        self.voivodships = self.manager.search(
            function='województwo',
            **omitter
        ).results
        self.powiats = self.manager.search(function='powiat', **omitter).results
        self.gminas = self.manager.search(function='gmina', **omitter).results

    def __repr__(self):
        return f"FrameLinkManagers({self.manager!r})"


class Register(ABC):
    by_possession = (
        'function',
    )
    by_prefix = (
        'date',
    )
    locname_keywords = (
        'name',
        'match',
        'startswith',
        'endswith',
        'contains'
    )
    optional_bool_arguments = (
        'raise_for_failure',
        'force_unpack',
        'unpack',
        'link',
        'unpacked',
        'case'
    )
    optional_str_arguments = (
        'terid',
    )
    optional_bool_str_arguments = (
        *optional_bool_arguments,
        *optional_str_arguments
    )
    erroneous_argument = \
        f'%s() got an unexpected keyword argument %r. ' \
        f'Try looking for the proper argument name ' \
        f'in the following list:\n{" " * 12}%s.'

    sentinel = RegisterSentinel()

    def __init__(self, link_manager=True, link=True):
        self.simc = simc_data
        self.terc = terc_data
        self.ulic = ulic_data
        self.system = self.__class__.__name__.replace(' ', '_').casefold()
        if self.system == "Register".casefold():
            raise Error("abstract class")
        self.field: pandas.DataFrame = getattr(
            self, self.system, None
        )
        require(
            self.field is not None,
            f'invalid system {self.system!r}'
        )
        self.field = self.field.reset_index()
        self._candidate = None

        self.cols = [*self.field.columns]
        self.len = len(self.field)
        self.columns = self.cols
        self.conflicts = (self.locname_keywords, ('force_unpack', 'unpack'))
        self.case = None
        self.force_unpack = None
        self.locname_value_space = None
        self.valid_keywords = None
        self.unpack = None
        self.search_keywords = None
        self._search_keywords = None
        self.search_mode = None
        self.raise_for_failure = None
        self._link = link
        self.results_found = False
        self.unpacked = False
        self.linked = False
        self._results = ResultFrameWrapper(self)
        self.transfer_target = None
        self.entry_helper = {}
        self.row = pandas.DataFrame()
        self.cache = {}
        self.store = self.cache.update
        if link_manager:
            self.link_manager = GenericLinkManager(
                dict_linkmanagers=DictLinkManagers(self),
                frame_linkmanagers=Register.frame_linkmanagers,
                cache=self.cache,
                klass=self
            )

    @property
    def results(self):
        return self._results

    def __repr__(self):
        return f"{self.system.upper()}()" + \
               (f"\nResults:\n{self.results}"
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
        if self.entry_helper['terid']:
            yield 'terid', self.entry_helper['terid']
        for value_space in self.value_spaces:
            if self.entry_helper.get(value_space, ""):
                yield value_space, self.entry_helper.get(value_space, "")

    def __getitem__(self, item):
        return dict(self)[item]

    def __del__(self):
        # self.__init__()
        # -1073741571 (0xC00000FD)
        pass

    def _dispatcher(self):
        if self._failure():
            not_found_exctype = error_types[self.system.lower()]
            if self.raise_for_failure:
                raise not_found_exctype('no results found')
            self.__del__()
        else:
            self.results_found = True
            self._results = ResultFrameWrapper(self, self._candidate.reset_index())
            self._results.frame = self._results.frame.drop(columns=["level_0"])
            if (len(self._candidate
                    ) == 1 or self.force_unpack) and self.unpack:
                return self.unpack_row()
        return self

    def _failure(self):
        return self._candidate.empty or self._candidate.equals(self.field)

    @set_sentinel(RegisterSentinel.to_keywords)
    def to_keywords(self, _transfer_target_name: Union[str, type]):
        """
        Create and return keywords leading to current search results.

        Parameters
        ----------
        _transfer_target_name: str or type
            Target class (simc, terc or ulic).

        Returns
        -------
        generator
        """
        transfer_target = self.transfer_target
        properties = dict(self)
        name_space_value = properties.pop('name')
        prop_copy = properties.copy()
        for k, v in prop_copy.items():
            if k in transfer_target.value_spaces and str(v):
                properties[k] = str(v)
            else:
                properties.__delitem__(k)
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
        return self.unpack_row(dataframe=dataframe, link=link)

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
        if not teritorial_id:
            raise ValueError('cannot unpack an empty teritorial ID string')
        chunks = {}
        frames = {}
        max_length = sum(self.link_spaces.values())
        if len(teritorial_id) > max_length:
            f'{self.system.upper()} teritorial ID length'
            f' is expected to be maximally {max_length}'
        index = 0

        for linkmanager_value_space, proper_length in self.link_spaces.items():
            if index >= len(teritorial_id) - 1:
                break
            frames.update(
                {linkmanager_value_space: getattr(
                    self.link_manager, linkmanager_value_space + 's'
                )}
            )
            chunk = teritorial_id[index:index + proper_length]
            unpack = self.unpack
            if errors:
                checker = type(self)().search(unpacked=True, unpack=False,
                                              **{linkmanager_value_space: chunk})
                if checker.results.empty:
                    raise ValueError(
                        repr(chunk) +
                        f'is an incorrect teritorial code chunk '
                        f'(error at {linkmanager_value_space!r} value space'
                        f', column '
                        f'{self.value_spaces[linkmanager_value_space]!r})'
                    )
            self.unpack = unpack
            chunks.update({linkmanager_value_space: chunk})
            index += proper_length

        return chunks

    @set_sentinel(sentinel.unpack_row)
    def unpack_row(self, *, dataframe: pandas.DataFrame = None, link=True) -> "Entry":  # noqa
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
        self._link = link

        for value_space, colname in self.value_spaces.items():
            value = self.row.iat[
                0, self.row.columns.get_loc(colname)
            ]

            if str(value) != 'nan':
                self.entry_helper[value_space] = value
                if self.link_manager.has_lm(value_space) and self._link:
                    name = self.link_manager.link(value_space, value)
                    index = self.link_manager.link_indexes.get(value_space, None)
                    if index:
                        self.entry_helper[value_space] = UnitLink(
                            id=value, name=name, index=index)
                    else:
                        self.entry_helper[value_space] = Link(
                            id=value, name=name)

        self.entry_helper['terid'] = self.pack_terid(**self.entry_helper)
        self.unpacked = True
        return entry_types[self.system].__call__(
            system=self,
            **self.entry_helper,
            row=self.row,
            index=self.row.iat[0, 0]
        )

    @set_sentinel(sentinel.search)
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

        Returns
        -------
        Entry
            If one most accurate entry was found.

        Register
            If there were many results.

        """
        self._candidate = Search(
            dataframe=self.field,
            field_name=self.system,
            search_mode=self.search_mode,
            value_spaces=self.value_spaces,
            case=self.case,
            locname=self.locname_value_space,
            by_possession=self.by_possession,
            by_prefix=self.by_prefix
        )(search_keywords=self.search_keywords)
        return self._dispatcher()

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
        Results:
           index WOJ POW GMI RODZ_GMI  RM MZ     NAZWA      SYM   SYMPOD     STAN_NA
        0   4810  04  14  05        2  00  1  Warszawa  1030760  0090316  2021-01-01
        1   5699  04  04  03        2  00  1  Warszawa  0845000  0844991  2021-01-01
        2   5975  04  14  07        2  00  1  Warszawa  0093444  0093438  2021-01-01

        >>> warsaw.to_list("sym")  # equivalent to warsaw.to_list("id")
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
        if link and self.link_manager.has_lm(value_space):
            for key_index in range(len(new_list)):
                new = self.__class__()
                entry = new.unpack_row(dataframe=pandas.DataFrame([
                    dataframe.loc[dataframe.index[key_index]]
                ]))
                new_list[key_index] = getattr(entry, value_space)

        return new_list

    def to_dict(self, link: bool = True) -> "dict":  # noqa
        results = self.results.copy()
        new_dict = {}
        for value_space in self.value_spaces:
            value = [*self.to_list(value_space, link=link)]
            new_dict.update(
                {value_space: value[0] if len(value) == 1 else value}
            )
        self.__init__()
        self._results = results
        return new_dict

    def transfer(self, target: Union[str, type], **other) -> "Register":
        global transfer_collector
        keywords, transfer_target = self.to_keywords(target)
        name = keywords['name']
        pop = transfer_collector.pop(name, ())
        transfer_collector[name] = pop + (
            self,
            transfer_target.search(
                **{**{vs: v for vs, v in other.items()
                      if any(
                        [vs in transfer_target.value_spaces,
                         vs in getattr(transfer_target, 'bool_and_str_arguments')]
                    )}, **keywords}
            ))
        return transfer_collector[name][-1]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            raise


class ResultFrameWrapper(object):
    def __init__(self, reg, frame=pandas.DataFrame()):
        self.reg = reg
        self.frame = frame

    def __repr__(self):
        return repr(self.frame)

    def __getattribute__(self, item):
        try:
            return object.__getattribute__(self, item)
        except AttributeError:
            return getattr(self.frame, item)

    def row(self, number, link=True):
        dataframe = self.frame[number]
        return (lambda: dataframe,
                lambda: self.reg.__class__.unpack_row(dataframe))[link]


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
    row: pandas.DataFrame = None
    date: str = None
    index: int = None

    @property
    def is_entry(self):
        return True

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


def index_(i, system=None, **params):
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
index_.__doc__ = Register.index.__doc__
transfer.__doc__ = Register.transfer.__doc__
to_list.__doc__ = Register.to_list.__doc__
to_dict.__doc__ = Register.to_dict.__doc__
ensure_value_space.__doc__ = Register.ensure_value_space.__doc__

terc = Terc = TERC
simc = Simc = SIMC
ulic = Ulic = ULIC

Register.frame_linkmanagers = FrameLinkManagers()
