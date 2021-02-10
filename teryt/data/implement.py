""" Implement TERYT bindings between columns and value spaces and links data. """

# This is the part of teryt library.
# Author: Stim (stijm), 2021
# License: GNU GPLv3

from types import SimpleNamespace

SIMC = SimpleNamespace(
    value_spaces={
        "voivodship": 'WOJ',
        "powiat": 'POW',
        "gmina": 'GMI',
        "gmitype": 'RODZ_GMI',
        "loctype": 'RM',
        "cnowner": 'MZ',
        "name": 'NAZWA',
        "id": 'SYM',
        "integral_id": 'SYMPOD',
        "date": 'STAN_NA'
    },
    link_spaces={
        "voivodship": 2,
        "powiat": 2,
        "gmina": 2,
        "gmitype": 1
    },
    cnowner_link_manager={
        True: '1',
        False: '0'
    },

    # This is WMRODZ, in a dict…
    loctype_link_manager={
        "miasto": '96',
        "delegatura": '98',
        "dzielnica m. st. Warszawy": '95',
        "część miasta": '99',
        "wieś": '01',
        "przysiółek": '03',
        "kolonia": '02',
        "osada": '04',
        "osada leśna": '05',
        "osiedle": '06',
        "schronisko turystyczne": '07',
        "część miejscowości": '00',
    }
)

TERC = SimpleNamespace(
    value_spaces={
        "voivodship": 'WOJ',
        "powiat": 'POW',
        "gmina": 'GMI',
        "gmitype": 'RODZ',
        "name": 'NAZWA',
        "function": 'NAZWA_DOD',
        "date": 'STAN_NA',
    },
    link_spaces={
        "voivodship": 2,
        "powiat": 2,
        "gmina": 2,
        "gmitype": 1,
    }
)

ULIC = SimpleNamespace(
    value_spaces={
        "voivodship": 'WOJ',
        "powiat": 'POW',
        "gmina": 'GMI',
        "gmitype": 'RODZ_GMI',
        "integral_id": 'SYM',
        "id": 'SYM_UL',
        "streettype": 'CECHA',
        "name": 'NAZWA_1',
        "secname": 'NAZWA_2',
        "date": 'STAN_NA'
    },
    link_spaces={
        'voivodship': 2,
        'powiat': 2,
        'gmina': 2,
        'gmitype': 1
    }
)

klass_name_dict = {
    "SIMC": SIMC,
    "TERC": TERC,
    "ULIC": ULIC
}


def data_implement(simc, terc, ulic):
    """ Internal helper function for implementing data on Register subclasses. """
    global SIMC, TERC, ULIC

    for klass in [simc, terc, ulic]:
        namespace = klass_name_dict[klass.__name__]
        for attr in filter(lambda x: not x.startswith("__"), namespace.__dict__):
            setattr(klass, attr, getattr(namespace, attr))
