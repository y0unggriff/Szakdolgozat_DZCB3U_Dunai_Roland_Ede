# Dunai Roland Ede szakdolgozatának forráskódja
A fájlok Visual Studio Code-ban lettek elkészítve

merge_seasons.py: Egységesíti a 8 különböző szezon CSV-jét, amelyek mind eltérő formátumúak voltak.

tisztitas.py: Sor szintű adattisztítást végez. Hozzáadja az MD5 hash-alapú player_id egyedi kulcsot, kiszűri a 750 percnél kevesebbet játszott játékosokat, megtisztítja a Nation oszlopot reguláris kifejezéssel, egységesíti a pozíció formátumot, és bajnokság/csapat/szezon szerint rendez.

osszevonas.py: Kezeli a szezon közbeni klubváltásokat. Az aggregate_df() függvény „Player” + „Season” párosítás alapján egy sorba aggregálja az ilyen játékosokat: összeadja a kumulatív statisztikákat, átlagolja a per-90 metrikákat, és a két klub nevét „Barcelona / Getafe” formában fűzi össze.

ketteszedes.py: Pozíció alapján két különálló táblába választja szét az adatokat: a Pos == "GK" kapusok külön kerülnek, a mezőnyjátékosok külön, mindkettőhöz saját oszlopokkal.

nevkereso.py: A hiányzó xG és xA értékeket pótolja vissza az Understat adataiból. Mivel a két forrás eltérő neveket használ, a best_match() függvény háromszintű hasonlósági módszer  maximumát veszi, és csak 70% feletti egyezést fogad el. Csapatnévnormalizálás is itt történik.

Szakdolgozat.py: Fő gépi tanulási szkript. Feature engineeringet végez, majd betanítja a három modellt: Ridge regresszió, Random Forest és XGBoost, és ezek súlyozott átlagából képezi az Ensemble modellt. Az evaluate() függvény számolja az MAE, RMSE és R² metrikákat, és a végén CSV-be írja a predikciókat a Power BI dashboard számára.

tesztek.py: Teszteli az előbb említett fájlok függvényeit.
