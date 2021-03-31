""" Implement TERYT bindings between columns and fields and links data. """

# This is the part of *teryt* library.
# Author: Stim (stijm), 2021
# License: MIT

from types import SimpleNamespace as Namespace


COMMON = Namespace(
    gmitype_link_mgr={
        "miejska": "1",
        "gmina miejska": "1",
        "wiejska": "2",
        "gmina wiejska": "2",
        "miejsko-wiejska": "3",
        "gmina miejsko-wiejska": "3",
        "miasto w gminie miejsko-wiejskiej": "4",
        "obszar wiejski w gminie miejsko-wiejskiej": "5",
        "dzielnice m. st. Warszawy": "8",
        "dzielnice Warszawy": "8",
        "dzielnica Warszawy": "8",
        "dzielnica": "8",
        "delegatury w miastach: Kraków, Łódź, Poznań i Wrocław": "9",
        "delegatura": "9"
    })


SIMC = Namespace(
    fields={
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
    link_fields={
        "voivodship": 2,
        "powiat": 2,
        "gmina": 2,
        "gmitype": 1
    },
    cnowner_link_mgr={
        True: '1',
        False: '0'
    },

    # See also: WMRODZ
    # ----------------
    loctype_link_mgr={
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

TERC = Namespace(
    fields={
        "voivodship": 'WOJ',
        "powiat": 'POW',
        "gmina": 'GMI',
        "gmitype": 'RODZ',
        "name": 'NAZWA',
        "function": 'NAZWA_DOD',
        "date": 'STAN_NA',
    },
    link_fields={
        "voivodship": 2,
        "powiat": 2,
        "gmina": 2,
        "gmitype": 1,
    }
)

ULIC = Namespace(
    fields={
        "voivodship": "WOJ",
        "powiat": "POW",
        "gmina": "GMI",
        "gmitype": "RODZ_GMI",
        "integral_id": "SYM",
        "id": "SYM_UL",
        "streettype": "CECHA",
        "name": "NAZWA_1",
        "secname": "NAZWA_2",
        "date": "STAN_NA"
    },
    link_fields={
        'voivodship': 2,
        'powiat': 2,
        'gmina': 2,
        'gmitype': 1
    }
)


function_dict = {
    "WOJ": "województwo",
    "POW": "powiat",
    "GMI": "gmina",
}


namespaces = {
    "COMMON": COMMON,
    "SIMC": SIMC,
    "TERC": TERC,
    "ULIC": ULIC
}


def update_vars_from_namespace(
        namespace,
        klass,
        key=(lambda x: not x.startswith("__"))
):
    """ Set filtered :namespace:'s attributes on a :klass:. """
    for attr in filter(key, dir(namespace)):
        setattr(
            klass,
            attr,
            object.__getattribute__(namespace, attr)
        )


def inject_master(register):
    """
    Internal helper function for implementing data on the System class.
    """
    namespace = namespaces["COMMON"]
    update_vars_from_namespace(namespace, register)


def inject_slaves(simc, terc, ulic):
    """
    Internal helper function for implementing data on the System subclasses.
    """
    global SIMC, TERC, ULIC  # pylint: disable=global-statement

    for klass in [simc, terc, ulic]:
        namespace = namespaces[klass.__name__]
        update_vars_from_namespace(namespace, klass)
