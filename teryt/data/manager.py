""" Manage TERYT data. """

# This is the part of *teryt* library.
# Author: Stim (stijm), 2021
# License: MIT

import os
import pandas as pd

from teryt.data import NA_FILLCHAR
from teryt.exceptions import ResourceError, MissingResourcesError


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
        raise ResourceError("resource file path "
                            "is indistinct; leave "
                            f"only 1 file in {path} "
                            "to continue")
    if not files:
        raise MissingResourcesError(
            f"{system} resource not found in {path}")

    file = os.path.join(path, files[0])

    if not file.endswith(".csv"):
        raise ResourceError(f"only .csv is supported ({file})")

    return file


read_csv_params = {
    "dtype": "str",
    "encoding": "utf-8",
    "sep": ";",
}

simc_data = pd.read_csv(resource_file("simc"), **read_csv_params)
terc_data = pd.read_csv(resource_file("terc"), **read_csv_params)
ulic_data = pd.read_csv(resource_file("ulic"), **read_csv_params)

databases = [simc_data, ulic_data, terc_data]

for database in databases:
    database.fillna(NA_FILLCHAR, inplace=True)
