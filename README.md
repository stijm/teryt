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
To search for entries, simply use system's `System.search` method.
For example:

```py 
>>> terc.search("Warszawa")
```
```
   index WOJ POW  ...     NAZWA                             NAZWA_DOD     STAN_NA
0   2057  14  65  ...  Warszawa  miasto stołeczne, na prawach powiatu  2021-01-01
1   2058  14  65  ...  Warszawa       gmina miejska, miasto stołeczne  2021-01-01
```
----
### Search keywords
`System.search` accepts plenty of keyword arguments for searching,
called **fields**,  which are the source database's columns representants.

#### `ULIC` secname (`str`)
Second name of a street.

Example use:
```py 
>>> ulic = teryt.ulic()
>>> ulic.search(secname="Księcia")
```
```
      index WOJ POW GMI  ...  CECHA            NAZWA_1  NAZWA_2     STAN_NA
0       429  02  14  08  ...    ul.   Henryka Wiernego  Księcia  2021-02-05
1      3235  02  03  01  ...    ul.            Jana II  Księcia  2021-02-05
2      4199  02  64  06  ...    ul.            Witolda  Księcia  2021-02-05
3      5628  06  61  01  ...    ul.            Witolda  Księcia  2021-02-05
4      7603  04  61  01  ...    ul.            Witolda  Księcia  2021-02-05
..      ...  ..  ..  ..  ...    ...                ...      ...         ...
123  252104  14  65  03  ...    ul.             Jaremy  Księcia  2021-02-05
124  253965  14  61  01  ...  rondo      Siemowita III  Księcia  2021-02-05
125  260928  14  18  04  ...    ul.  Janusza I Starego  Księcia  2021-02-05
126  266940  24  64  01  ...    ul.     Leszka Białego  Księcia  2021-02-05
127  268012  20  62  01  ...    ul.         Stanisława  Księcia  2021-02-05
[128 rows x 11 columns]

```

#### `COMMON` date (`str`)
"State as of", the date in `STAN_NA` column.

#### `COMMON` name (`str`)
Name of the searched locality, street or unit.

Example use:
```py 
>>> terc.search(name="Piła")
```
```
Unit(
    name='Piła', 
    terid='3019011', 
    system=TERC, 
    voivodship=UnitLink(code='30', value='WIELKOPOLSKIE', index=3510), 
    powiat=UnitLink(code='19', value='pilski', index=3769), 
    gmina=UnitLink(code='01', value='Piła', index=3770), 
    gmitype=Link(code='1', value='gmina miejska'), 
    function='gmina miejska', 
    date='2021-01-01', 
    index=3770
)
```

#### `SIMC` loctype (`str`)
Locality type.

Example use:
```py 
>>> simc = teryt.simc()
>>> st = simc.search(loctype="schronisko turystyczne")
>>> st
```
```
    index WOJ POW GMI  ...                 NAZWA      SYM   SYMPOD     STAN_NA
0    2780  02  06  07  ...            Szwajcarka  0191402  0191402  2021-01-01
1    2781  02  06  08  ...            Odrodzenie  0192123  0192123  2021-01-01
2   17637  08  11  09  ...              Grabówek  0916147  0916147  2021-01-01
3   39446  12  15  08  ...     Markowe Szczawiny  0078114  0078114  2021-01-01
4   39450  12  17  03  ...  Dolina Pięciu Stawów  0418410  0418410  2021-01-01
5   39451  12  17  03  ...           Morskie Oko  0418432  0418432  2021-01-01
6   39452  12  17  03  ...       Polana-Głodówka  0418449  0418449  2021-01-01
7   39453  12  17  03  ...               Roztoka  0418455  0418455  2021-01-01
8   39454  12  17  03  ...            Włosienica  0418478  0418478  2021-01-01
9   39455  12  15  04  ...          Krupowa Hala  0419704  0419704  2021-01-01
10  39456  12  10  08  ...              Łabowiec  0442873  0442873  2021-01-01
11  39457  12  11  09  ...               Turbacz  0457432  0457432  2021-01-01
12  39458  12  17  04  ...           Hala Pisana  0468648  0468648  2021-01-01
13  39459  12  17  04  ...                 Ornak  0468654  0468654  2021-01-01
14  39460  12  17  04  ...   Dolina Chochołowska  0468915  0468915  2021-01-01
15  66751  20  12  07  ...                Słupie  0769166  0769166  2021-01-01
16  78067  24  02  10  ...       Kamieńska Płyta  1001332  1001332  2021-01-01
17  78081  24  17  02  ...        Chrobacza Łąka  0051262  0051262  2021-01-01
18  78082  24  17  04  ...                Pilsko  0055573  0055573  2021-01-01
19  78083  24  02  10  ...              Klimczok  0076569  0076569  2021-01-01
20  78084  24  02  10  ...          Szyndzielnia  0076598  0076598  2021-01-01
21  78085  24  02  10  ...               Magurka  0076629  0076629  2021-01-01
[22 rows x 11 columns]
```

Simple check:
```py 
>>> st.get_entry(1).loctype
```
```
Link(code='07', value='schronisko turystyczne')
```


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

### Search results
Results of a search returned from `System.search` are not in fact DataFrame.
It's `Entry` or `EntryGroup`, synced with fields.

That's what you can do with them:


