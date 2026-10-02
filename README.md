# Pčelinja riječ

Hrvatska igra u stilu NYT Spelling Bee. Otvori `index.html` u pregledniku.
Gumb **Nova zagonetka** generira novu zagonetku iz ugrađenog popisa riječi.

Popis riječi: frekvencije iz hermitdave/FrequencyWords (hr), filtrirano LibreOffice hr_HR rječnikom.
Ponovna izgradnja: `python3 tools/build_wordlist.py` (stvara `good.txt`), zatim `python3 tools/build.py good.txt`.
