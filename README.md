# teryt
**teryt** library – efficient search engine for TERC, SIMC and ULIC systems.

# User guide (WIP)
## Step 1: Migrate official CSV databases
If you have not yet downloaded official TERYT databases,
visit [this TERYT website](
https://eteryt.stat.gov.pl/eTeryt/rejestr_teryt/udostepnianie_danych/baza_teryt/uzytkownicy_indywidualni/pobieranie/pliki_pelne.aspx?contrast=default
) and download SIMC, TERC and ULIC databases in `.csv` format.

Warning: Only `.csv` extensions and official (not statistical) versions are supported.

Now, place SIMC database file in `teryt/data/SIMC`, and so on with other systems.
Make sure there's only one file in each `teryt/data/<SYSTEM>` directory. 

## Step 2: Choose a TERYT system to begin with
In this example I choose TERC, which is a system containing
identifiers and names of Polish units of territorial division.

```python
import teryt
terc = teryt.TERC()
```

## Step 3: Usage!

### Searching system entries
To search for entries, simply use system's `Register.search` method.
For example:

```py 
>>> terc.search("Warszawa")
```
```
TERC()
Results:
   index WOJ POW  ...     NAZWA                             NAZWA_DOD     STAN_NA
0   2057  14  65  ...  Warszawa  miasto stołeczne, na prawach powiatu  2021-01-01
1   2058  14  65  ...  Warszawa       gmina miejska, miasto stołeczne  2021-01-01
```
----
### Search keywords
`Register.search` accepts plenty of keyword arguments for searching,
called **fields**,  which are the source database's columns representants.

#### `ULIC` secname (`str`)
Second name of a street.

Example use:
```py 
>>> ulic = teryt.terc()
>>> ulic.search(secname="Księcia")
```
```

```

#### `COMMON` date (`str`)
"State as of", the date in `STAN_NA` column.

#### `COMMON` name (`str`)
Name of the searched locality, street or unit.

#### `SIMC` loctype (`str`)
Locality type.

#### `COMMON` gmina (`str`)
Gmina of the searched locality, street or unit.

#### `COMMON` voivodship (`str`)
Voivodship of the searched locality, street or unit.

#### `TERC` function (`str`)
Unit function.

#### `ULIC` streettype (`str`)
Street type.

#### `COMMON` powiat (`str`)
Voivodship of the searched locality, street or unit.

#### `SIMC` cnowner (`bool`)
States whether a locality owns a common name.
As of 09.02, all Polish localities are "cnowners". Using this keyword 
may result in a kind warning of no uniqueness.

#### `SIMC, ULIC` id (`str`)
ID of a locality or street.

#### `SIMC, ULIC` integral_id (`str`)
Integral ID of a locality/street.

#### `COMMON` gmitype (`str`)
Gmina type of the searched locality, street or unit.

----

Column names as the above listed arguments are also acceptable.
It means that you can pass database's columns names 
(called **root names**) instead of passing the field name.
Fields were involved in order to unify columns of the systems' databases.

### Working on search results


