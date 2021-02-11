""" Manage TERYT data. """

# This is the part of *teryt* library.
# Author: Stim (stijm), 2021
# License: GNU GPLv3

import os
import pandas
from ..exceptions import MissingResourcesError as MRes

directory = os.path.abspath(os.path.dirname(__file__))


def resource_file(system):
    """
    Resolve the path of a TERYT system .csv resource file and return it.

    Parameters
    ----------
    system : str
        The TERYT system, e.g. "simc".

    Returns
    -------
    str
    """
    path = os.path.join(directory, system.upper())
    files = os.listdir(path)

    if len(files) > 1:
        raise MRes(f"resource file path is indistinct; leave only 1 file "
                   f"in {path} to continue")
    elif not files:
        raise MRes(f"{system} resource not found in {path}")

    file = os.path.join(path, files[0])

    if not file.endswith(".csv"):
        raise MRes(f"only .csv is supported ({file})")

    return file


read_csv_params = {
    "dtype": "str",
    "encoding": "utf-8",
    "sep": ";",
}

simc_data = pandas.read_csv(resource_file("simc"), **read_csv_params)
terc_data = pandas.read_csv(resource_file("terc"), **read_csv_params)
ulic_data = pandas.read_csv(resource_file("ulic"), **read_csv_params)
