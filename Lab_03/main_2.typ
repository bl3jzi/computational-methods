#set document(title: "Laboratorium - Interpolacja")
#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.5cm, left: 2.5cm, right: 2.5cm),
)
#set text(font: "New Computer Modern", size: 11pt, lang: "pl")
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")
#set math.equation(numbering: "(1)")

// ─── Nagłówek ───────────────────────────────────────────────────────────────

#align(center)[
  #text(size: 14pt, weight: "bold")[Laboratorium 3]\
  #v(0.1em)
  Interpolacja

  #v(0.4em)
  *Autorzy:*\

  Kacper Hawro, Jan Ślosarczyk \
  #datetime.today().display("[day].[month].[year]")
]

#line(length: 100%, stroke: 0.5pt)
#v(0.5em)

// ─── Zadanie 1 ──────────────────────────────────────────────────────────────

= Zadanie 1

== Treść zadania

Dane są następujące punkty opisujące populację Stanów Zjednoczonych na przestrzeni lat:

#align(center)[
  #table(
    columns: (auto, auto),
    align: center,
    stroke: 0.5pt,
    [*Rok*], [*Populacja*],
    [1900], [76 212 168],
    [1910], [92 228 496],
    [1920], [106 021 537],
    [1930], [123 202 624],
    [1940], [132 164 569],
    [1950], [151 325 798],
    [1960], [179 323 175],
    [1970], [203 302 031],
    [1980], [226 542 199],
  )
]

Rozważamy cztery zbiory funkcji bazowych $phi_j (t)$, $j = 1, dots, 9$:

$
phi_j (t) = t^(j-1) quad quad quad quad quad quad quad
phi_j (t) = (t - 1900)^(j-1)
$

$
phi_j (t) = (t - 1940)^(j-1) quad quad quad
phi_j (t) = ((t - 1940)/40)^(j-1)
$

Zadanie obejmuje: (a) budowę macierzy Vandermonde'a dla każdej bazy, (b) obliczenie współczynnika uwarunkowania, (c) wyznaczenie współczynników wielomianu przy użyciu najlepiej uwarunkowanej bazy i narysowanie wykresu z użyciem schematu Hornera, (d) ekstrapolację do roku 1990 i obliczenie błędu względnego, (e) wyznaczenie wielomianu Lagrange'a, (f) wyznaczenie wielomianu Newtona, (g) analizę wpływu zaokrąglenia danych do miliona.

== Rozwiązanie

=== Podpunkt (a): Macierze Vandermonde'a

Dla każdej z czterech baz konstruujemy macierz Vandermonde'a $V$ rozmiaru $9 times 9$, gdzie element na pozycji $(i, j)$ wynosi $phi_j(t_i)$. W implementacji wykorzystano mechanizm broadcastingu biblioteki NumPy:

```python
exponent = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

V_1 = dates[:, np.newaxis] ** exponent                    # baza 1
V_2 = (dates - 1900)[:, np.newaxis] ** exponent          # baza 2
V_3 = (dates - 1940)[:, np.newaxis] ** exponent          # baza 3
V_4 = ((dates - 1940) / 40)[:, np.newaxis] ** exponent   # baza 4
```

Dla zobrazowania wynikowej struktury, poniżej przedstawiono macierz $V_4$ dla najlepiej uwarunkowanej Bazy 4 (wartości zaokrąglone do 2 miejsc po przecinku). Widać na niej wyraźnie, jak przeskalowanie węzłów czasowych (lat) do przedziału bliskiego $[-1, 1]$ zapobiega powstawaniu gigantycznych rzędów wielkości przy wyższych potęgach:

$ V_4 = mat(
  1, -1, 1, -1, 1, -1, 1, -1, 1;
  1, -0.75, 0.56, -0.42, 0.32, -0.24, 0.18, -0.13, 0.1;
  1, -0.5, 0.25, -0.12, 0.06, -0.03, 0.02, -0.01, 0;
  1, -0.25, 0.06, -0.02, 0, 0, 0, 0, 0;
  1, 0, 0, 0, 0, 0, 0, 0, 0;
  1, 0.25, 0.06, 0.02, 0, 0, 0, 0, 0;
  1, 0.5, 0.25, 0.12, 0.06, 0.03, 0.02, 0.01, 0;
  1, 0.75, 0.56, 0.42, 0.32, 0.24, 0.18, 0.13, 0.1;
  1, 1, 1, 1, 1, 1, 1, 1, 1
) $



=== Podpunkt (b): Współczynniki uwarunkowania

Współczynnik uwarunkowania $kappa(V)$ macierzy mierzy czułość rozwiązania układu równań na zaburzenia danych wejściowych. Duże $kappa$ oznacza słabe uwarunkowanie i potencjalną utratę precyzji numerycznej.

Obliczone wartości przy użyciu funkcji `numpy.linalg.cond`:

#align(center)[
  #table(
    columns: (auto, auto, auto),
    align: (left, center, center),
    stroke: 0.5pt,
    [*Baza*], [*Wzór*], [*$kappa(V)$*],
    [Baza 1], [$t^(j-1)$],                   [$approx 2.59 times 10^(26)$],
    [Baza 2], [$(t-1900)^(j-1)$],             [$approx 5.76 times 10^(15)$],
    [Baza 3], [$(t-1940)^(j-1)$],             [$approx 9.32 times 10^(12)$],
    [Baza 4], [$((t-1940)/40)^(j-1)$],        [$approx 1.61 times 10^3$],
  )
]

Najlepiej uwarunkowana jest *Baza 4*, która centruje i skaluje dane tak, że węzły interpolacji leżą w przedziale $[-1, 1]$. Baza 1 jest skrajnie źle uwarunkowana ze względu na ogromne potęgi liczb rzędu $10^3$.

#pagebreak()

=== Podpunkt (c): Schemat Hornera i wykres wielomianu

Używając najlepiej uwarunkowanej Bazy 4, rozwiązujemy układ $V_4 bold(a) = bold(y)$, aby wyznaczyć współczynniki wielomianu interpolacyjnego.

Do efektywnego obliczania wartości wielomianu zastosowano *schemat Hornera*, który minimalizuje liczbę mnożeń. Dla bazy skalowanej $(t - 1940)/40$ schemat przyjmuje postać:

$ p(t) = a_0 + s(a_1 + s(a_2 + s( dots + s dot a_8 dots))), quad s = frac(t - 1940, 40) $

```python
def horner_scheme(year, coeffs):
    scaled = (year - 1940) / 40
    res = 0
    for i in range(len(coeffs) - 1, -1, -1):
        res = res * scaled + coeffs[i]
    return res
```

Wartości wielomianu zostały obliczone w odstępach jednorocznych na przedziale $[1900, 1990]$.

#figure(image("lagrange.png", width: 90%), caption: [Wielomian interpolacyjny z węzłami])

Wyznaczone współczynniki wielomianu (dla Bazy 4) prezentują się następująco:

#align(center)[
  #table(
    columns: (auto, auto),
    align: (center, right),
    stroke: 0.5pt,
    [*Współczynnik*], [*Wartość*],
    [$a_0$], [$ 1.32164569 times 10^8$],
    [$a_1$], [$ 4.61307656 times 10^7$],
    [$a_2$], [$ 1.02716315 times 10^8$],
    [$a_3$], [$ 1.82527130 times 10^8$],
    [$a_4$], [$-3.74614715 times 10^8$],
    [$a_5$], [$-3.42668456 times 10^8$],
    [$a_6$], [$ 6.06291250 times 10^8$],
    [$a_7$], [$ 1.89175576 times 10^8$],
    [$a_8$], [$-3.15180235 times 10^8$],
  )
]

=== Podpunkt (d): Ekstrapolacja do roku 1990

Wielomian interpolacyjny wyznaczony na podstawie danych z lat 1900–1980 użyty został do ekstrapolacji populacji dla roku 1990. Prawdziwa wartość populacji USA w 1990 r. wynosiła 248 709 873.

Błąd względny ekstrapolacji definiujemy jako:

$ E_"rel" = frac(|y_"true" - y_"pred"|, y_"true") $

Otrzymane wyniki:

#align(center)[
  #table(
    columns: (auto, auto),
    align: (left, right),
    stroke: 0.5pt,
    [*Wartość przewidywana (1990)*], [82 749 141],
    [*Wartość prawdziwa (1990)*],    [248 709 873],
    [*Błąd względny*],               [$approx 66.73%$],
  )
]

Tak duży błąd ekstrapolacji jest typowym zjawiskiem dla wielomianów wysokiego stopnia — poza przedziałem interpolacji mogą one zachowywać się bardzo nieregularnie (efekt Rungego).

=== Podpunkt (e): Wielomian interpolacyjny Lagrange'a

Wielomian Lagrange'a wyznaczamy bezpośrednio z definicji wielomianów bazowych $l_j(t)$:

$ p(t) = sum_(j=0)^(n) y_j dot l_j(t), quad l_j(t) = product_(i eq.not j) frac(t - t_i, t_j - t_i) $

```python
def lagrange_interpolation(x_nodes, y_nodes, x_eval):
    y_eval = np.zeros_like(x_eval, dtype=float)
    n = len(x_nodes)
    for i in range(n):
        L_i = np.ones_like(x_eval, dtype=float)
        for j in range(n):
            if i != j:
                L_i *= (x_eval - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        y_eval += y_nodes[i] * L_i
    return y_eval
```

Metoda ta wyznacza ten sam wielomian co podejście przez macierz Vandermonde'a, lecz wymaga $O(n^2)$ operacji na punkt ewaluacji. Wyniki dla wybranych lat:

#align(center)[
  #table(
    columns: (auto, auto, auto),
    align: (center, right, right),
    stroke: 0.5pt,
    [*Rok*], [*Wartość (Lagrange)*], [*Wartość prawdziwa*],
    [1950], [151 325 798], [151 325 798],
    [1990], [82 749 141], [248 709 873],
  )
]

Zgodność w węźle interpolacji (1950) potwierdza poprawność implementacji.

=== Podpunkt (f): Wielomian interpolacyjny Newtona

Wielomian Newtona wyznaczamy przy pomocy *ilorazów różnicowych*. Tablica ilorazów różnicowych $[t_i, dots, t_{i+k}]$ definiowana jest rekurencyjnie:

$ [t_i] = y_i, quad [t_i, dots, t_{i+k}] = frac([t_{i+1}, dots, t_{i+k}] - [t_i, dots, t_{i+k-1}], t_{i+k} - t_i) $

Postać Newtona wielomianu:

$ p(t) = [t_0] + [t_0, t_1](t-t_0) + [t_0,t_1,t_2](t-t_0)(t-t_1) + dots $

```python
def divided_differences(x, y):
    n = len(y)
    coef = np.zeros([n, n])
    coef[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j] - x[i])
    return coef[0, :]

def newton_interpolation(x_nodes, coeffs, x_eval):
    n = len(coeffs)
    res = coeffs[n-1]
    for i in range(n-2, -1, -1):
        res = res * (x_eval - x_nodes[i]) + coeffs[i]
    return res
```

Predykcja dla roku 1990 wynosi 82 749 141 — identycznie jak w metodzie Lagrange'a i Vandermonde'a, co potwierdza, że wszystkie trzy metody wyznaczają ten sam wielomian interpolacyjny ósmego stopnia.

=== Podpunkt (g): Wpływ zaokrąglenia danych

Dane populacyjne zaokrąglono do najbliższego miliona. Zaokrąglone wartości populacji:

#align(center)[
  #table(
    columns: (auto, auto, auto),
    align: (center, right, right),
    stroke: 0.5pt,
    [*Rok*], [*Oryginał*], [*Zaokrąglone*],
    [1900], [76 212 168], [76 000 000],
    [1940], [132 164 569], [132 000 000],
    [1980], [226 542 199], [227 000 000],
  )
]

Zgodnie z poleceniem wyznaczono nowe współczynniki wielomianu dla najlepiej uwarunkowanej Bazy 4. Poniższa tabela prezentuje bezpośrednie porównanie współczynników przed i po zaokrągleniu danych:

#align(center)[
  #table(
    columns: (auto, auto, auto, auto),
    align: (center, right, right, right),
    stroke: 0.5pt,
    [*Współczynnik*], [*Przed zaokrągleniem (c)*], [*Po zaokrągleniu (g)*], [*Różnica*],
    [$a_0$], [$ 1.3216 times 10^8$], [$ 1.3200 times 10^8$], [$ 1.6457 times 10^5$],
    [$a_1$], [$ 4.6131 times 10^7$], [$ 4.5957 times 10^7$], [$ 1.7362 times 10^5$],
    [$a_2$], [$ 1.0272 times 10^8$], [$ 1.0014 times 10^8$], [$ 2.5750 times 10^6$],
    [$a_3$], [$ 1.8253 times 10^8$], [$ 1.8111 times 10^8$], [$ 1.4160 times 10^6$],
    [$a_4$], [$-3.7461 times 10^8$], [$-3.5676 times 10^8$], [$-1.7859 times 10^7$],
    [$a_5$], [$-3.4267 times 10^8$], [$-3.3849 times 10^8$], [$-4.1796 times 10^6$],
    [$a_6$], [$ 6.0629 times 10^8$], [$ 5.7031 times 10^8$], [$ 3.5980 times 10^7$],
    [$a_7$], [$ 1.8918 times 10^8$], [$ 1.8692 times 10^8$], [$ 2.2549 times 10^6$],
    [$a_8$], [$-3.1518 times 10^8$], [$-2.9420 times 10^8$], [$-2.0983 times 10^7$],
  )
]

*Wyjaśnienie wyniku:* Pomimo że zaokrąglenie danych o zaledwie ok. 0.1–0.5% zmienia dane wejściowe w bardzo niewielkim stopniu, różnice w skrajnych współczynnikach sięgają nawet rzędu $10^7$. Wynika to z faktu, że dla wielomianów wysokiego stopnia, nawet relatywnie niewielkie zaburzenie zostaje mocno wzmocnione przez sam układ równań (mimo zastosowania najlepiej uwarunkowanej Bazy 4, której współczynnik uwarunkowania to nadal $kappa approx 1600$).

Błąd względny ekstrapolacji dla roku 1990 obliczony na podstawie zaokrąglonych danych:
- *Przewidywana populacja:* 109 000 000
- *Błąd względny:* 56.17%

Wynik predykcji zmienił się aż o ponad 26 milionów w porównaniu do modelu opartego na oryginalnych danych (82 749 141), co ostatecznie udowadnia niestabilność i wrażliwość tego wielomianu.
== Wnioski

#v(0.7em)

Zadanie pokazało, że wybór bazy funkcji interpolacyjnych ma kluczowe znaczenie dla stabilności numerycznej. Baza skalowana i centrowana (Baza 4) jest o ponad $10^23$ razy lepiej uwarunkowana od bazy monomialnej (Baza 1), co bezpośrednio przekłada się na dokładność wyznaczonych współczynników.

Wszystkie trzy metody (Vandermonde, Lagrange, Newton) wyznaczają ten sam wielomian interpolacyjny — różnią się jedynie reprezentacją i kosztem obliczeniowym. Schemat Hornera pozwala na efektywną ewaluację wielomianu przy minimalnej liczbie operacji arytmetycznych.

Ekstrapolacja do roku 1990 daje błąd rzędu $66.73%$, co ilustruje typowy problem wielomianów wysokiego stopnia: wewnątrz przedziału interpolacji mogą zachowywać się dobrze, natomiast poza nim gwałtownie rosną lub maleją (dywergują), przez co drastycznie odbiegają od rzeczywistości. Należy stanowczo podkreślić, że tak niska dokładność predykcji nie wynika z efektu Rungego (który powoduje sztuczne oscylacje wewnątrz przedziału interpolacji przy węzłach równoodległych), lecz z naturalnej asymptotyki wielomianów. Z tego powodu globalne wielomiany wysokiego stopnia są skrajnie niestabilne i zupełnie nie nadają się do zadań ekstrapolacyjnych.

// ─── Zadanie 2 ──────────────────────────────────────────────────────────────

#pagebreak()


= Zadanie 2: Roczna produkcja cytrusów we Włoszech

== Treść zadania

Roczna produkcja cytrusów we Włoszech kształtowała się następująco:

#align(center)[
  #table(
    columns: (auto, auto),
    align: center,
    stroke: 0.5pt,
    [*Rok*], [*Produkcja [$10^5$ kg]*],
    [1965], [17 769],
    [1970], [24 001],
    [1980], [25 961],
    [1985], [34 336],
    [1990], [29 036],
    [1991], [33 417],
  )
]

Zadanie wymaga:
+ Użycia kubicznych funkcji sklejanych (splajnów 3. stopnia) różnych rodzajów, aby oszacować produkcję w latach 1962, 1977 oraz 1992. 
+ Porównania otrzymanych wyników z wartościami prawdziwymi, które wynosiły odpowiednio: 12 380, 27 403 oraz 32 059. 
+ Powtórzenia analizy, używając globalnego wielomianu interpolacyjnego Lagrange'a.

== Rozwiązanie

=== Szacowanie za pomocą kubicznych funkcji sklejanych
Kubiczne funkcje sklejane (splajny) dzielą przedział na podprzedziały i na każdym z nich definiują osobny wielomian 3. stopnia. Wymagają one podania dodatkowych warunków brzegowych na końcach dziedziny. Do analizy wykorzystano implementację biblioteki `scipy.interpolate.CubicSpline` i przetestowano trzy warianty:
- *not-a-knot* – zakłada, że trzecia pochodna jest ciągła w pierwszym i ostatnim węźle wewnętrznym.
- *natural* – wymusza zerowanie się drugiej pochodnej na brzegach przedziału, co fizycznie odpowiada "wypłaszczaniu" się krzywej na skrajach.
- *clamped* – wymusza zerowanie pierwszej pochodnej na brzegach: $S'_0 (a) = S'_0 (b) = 0$, co oznacza, że krzywa wchodzi w skrajne węzły idealnie poziomo.

=== Szacowanie za pomocą wielomianu Lagrange'a
W celu zbudowania punktu odniesienia wyznaczono tradycyjny, globalny wielomian interpolacyjny Lagrange'a (stopnia 5, ponieważ badamy 6 węzłów). Obliczono jego wartości w badanych latach w celu weryfikacji precyzji wewnątrz przedziału (interpolacja: rok 1977) oraz na zewnątrz (ekstrapolacja: lata 1962 i 1992).

#pagebreak()

=== Szacowanie za pomocą PCHIP

Metoda _Piecewise Cubic Hermite Interpolating Polynomial_ (PCHIP) konstruuje interpolację kawałkami wielomianami trzeciego stopnia, podobnie jak klasyczne splajny kubiczne. Kluczowa różnica polega na sposobie wyznaczania pochodnych w węzłach — PCHIP dobiera je *lokalnie*, na podstawie jedynie sąsiednich węzłów, według algorytmu Fritsch-Carlsona. Algorytm ten gwarantuje zachowanie *monotoniczności* danych: jeśli wartości funkcji rosną między dwoma węzłami, krzywa również rośnie w tym podprzedziale, a w węzłach będących lokalnymi ekstremami pochodna jest ustawiana na zero.

Konsekwencją lokalnego wyznaczania pochodnych jest to, że PCHIP zapewnia jedynie ciągłość pierwszej pochodnej ($C^1$), w przeciwieństwie do klasycznych splajnów kubicznych, które są $C^2$. Krzywa może być zatem mniej wizualnie "gładka", lecz wierniej odwzorowuje lokalne trendy w danych. Podobnie jak wszystkie metody interpolacyjne, PCHIP nie posiada szczególnych właściwości gwarantujących poprawną ekstrapolację poza przedział danych.

=== Zestawienie wyników i błędów
Poniższa tabela prezentuje produkcję przewidywaną przez poszczególne modele oraz ich błędy względne w stosunku do wartości prawdziwych.

#align(center)[
  #table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (center, center, left, left, left, left, left),
    stroke: 0.5pt,
    [*Rok*], [*Prawdziwa produkcja*], [*Splajn (not-a-knot)*], [*Splajn (naturalny)*], [*Splajn (clamped)*], [*PCHIP*], [*Wielomian Lagrange'a*],
    [1962], [12 380], [5 146 \ (Błąd: 58.43%)], [13 285 \ (Błąd: 7.31%)], [24 313 \ (Błąd: 96.39%)], [13 869 \ (Błąd: 12.03%)], [-77 704 \ (Błąd: 727.66%)],
    [1977], [27 403], [22 642 \ (Błąd: 17.37%)], [22 934 \ (Błąd: 16.31%)], [23 126 \ (Błąd: 15.61%)], [25 204 \ (Błąd: 8.02%)], [15 470 \ (Błąd: 43.55%)],
    [1992], [32 059], [41 894 \ (Błąd: 30.68%)], [37 798 \ (Błąd: 17.90%)], [22 166 \ (Błąd: 30.86%)], [32 663 \ (Błąd: 1.89%)], [43 191 \ (Błąd: 34.72%)],
  )
]

#figure(
  image("wykres_fixed.png", width: 90%), 
  caption: [Interpolacja i ekstrapolacja produkcji cytrusów we Włoszech przy użyciu różnych modeli. (Uwaga: oś rzędnych została celowo ograniczona dla czytelności z powodu silnej dywergencji wielomianu Lagrange'a.)]
)

== Wnioski

+ *Słabość globalnego wielomianu Lagrange'a:* Zastosowanie wielomianu Lagrange'a daje najgorsze wyniki spośród wszystkich metod. Przy ekstrapolacji w przeszłość (rok 1962) wielomian zwraca absurdalną wartość ujemną (błąd 728.62%), a nawet przy interpolacji wewnątrz przedziału (rok 1977) błąd wynosi ponad 44%. Wynika to z ogólnej właściwości wielomianów wysokiego stopnia — nie z efektu Rungego, który dotyczy oscylacji wewnątrz przedziału przy równoodległych węzłach — lecz z faktu, że wielomiany wysokiego stopnia z definicji silnie dywergują poza zakresem danych.
+ *Przewaga splajnów i PCHIP nad wielomianem globalnym:* Wszystkie metody oparte na interpolacji lokalnej (splajny oraz PCHIP) okazały się znacznie bardziej zachowawcze. W każdym z badanych lat wygenerowały mniejszy błąd niż wielomian Lagrange'a.
+ *Pułapka splajnu usztywnionego (clamped):* Splajn typu clamped narzuca zerową pierwszą pochodną na brzegach, co wymusza poziome wejście krzywej w skrajne węzły. Dla danych o wyraźnym trendzie jest to założenie nienaturalne, które prowadzi do drastycznych błędów ekstrapolacji (96.39% dla roku 1962).
+ *PCHIP — zachowanie monotoniczności kosztem gładkości:* PCHIP uzyskał najlepszy wynik dla roku 1992 (błąd 1.89%) oraz konkurencyjny dla roku 1977 (8.02%). Należy jednak zauważyć, że dobry wynik dla 1992 r. może być częściowo zbiegiem okoliczności — poza przedziałem krzywa PCHIP również silnie dywerguje, co widoczne jest na wykresie. PCHIP jest szczególnie wartościowy gdy zależy nam na zachowaniu monotoniczności wewnątrz przedziału, kosztem nieco mniejszej gładkości ($C^1$ zamiast $C^2$).
+ *Splajn naturalny jako najbardziej niezawodna metoda dla tych danych:* Spośród wszystkich metod splajn naturalny wykazał najbardziej zrównoważone wyniki — błąd 7.31% dla roku 1962 i 17.90% dla roku 1992. Zerowanie drugiej pochodnej na brzegach ogranicza zakrzywienie krzywej na skrajach przedziału, co sprzyja łagodniejszej ekstrapolacji. Należy jednak pamiętać, że jest to obserwacja specyficzna dla tego zbioru danych.