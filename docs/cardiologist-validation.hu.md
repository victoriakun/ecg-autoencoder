# Kardiológusi konzultáció — felkészítő anyag és kérdéssor

Használd ezt a dokumentumot végig a kardiológussal való találkozón. Az
első fele az, amit *te* mondasz; a második fele az, amire *ő* válaszol.
Vidd magaddal a laptopot a futó GUI demóval (208-as felvétel) **és** a
nyomtatott `clinical_proof.pdf` fájlt (16 példa-szívverés).

---

## 1. rész — Amit el kell mondanod

### 1.1 Ki vagy és mit építettél (60 másodperc)

> "Szakdolgozó hallgató vagyok, és egy felügyelet nélküli (unsupervised)
> anomália-észlelő rendszert építettem EKG-jelekre. A cél egy olyan
> szűrőeszköz, amely folyamatosan, valós időben fut egyetlen elvezetésű
> EKG-folyamon, egy másodpercnél rövidebb idő alatt jelzi a rendellenes
> szívveréseket, és minden riasztást auditálható adatbázisba ír.
> Klinikus első-szűrőnek szánom, nem a kardiológus helyettesítésére."

**Mit jelent a „felügyelet nélküli" (unsupervised) tanítás?**

> „A klasszikus (felügyelt) gépi tanulásnál minden tanítóminta mellé
> kézzel adott címke tartozik — pl. „ez a szívverés normál",
> „ez kamrai extraszisztolé". A modell ezekből a címkepárokból tanulja
> meg elkülöníteni az osztályokat. **Felügyelet nélküli** tanításnál
> nem használunk címkéket: csak egyfajta adatot — kizárólag **normál
> szívveréseket** — mutatunk a modellnek, és azt tanítjuk, hogy
> *rekonstruálja* ezeket a jeleket. A modell így megtanulja, milyen
> *az egészséges EKG mintázata*, de **sosem látott rendellenes
> verést tanítás közben**. Futásidőben bármi, ami nem illik bele a
> megtanult normál mintába, nagy rekonstrukciós hibát ad, és ezt
> jelöljük anomáliának. Előny: nem kell ritka aritmia-példákat
> gyűjteni a tanításhoz, és a modell *soha nem látott* ritkaságokra
> is reagál. Hátrány: nem tudja megmondani, hogy *milyen típusú* az
> anomália — csak azt, hogy *eltér a normáltól*."

### 1.2 Az adathalmaz — légy konkrét

**MIT-BIH Arrhythmia Database** — a világ legelterjedtebb EKG-anomália
benchmark adatbázisa, a Boston Beth Israel Kórházban és az MIT-ben vették
fel 1975–1979 között.

- **47 alany** (25 férfi, 22 nő, 23–89 év), **48 felvétel** (két alanyról
  két felvétel van)
- **Fél órás Holter-felvételek** — összesen ~24 óra adat, **~110 000
  kézzel annotált szívverés**
- **Mintavételi frekvencia: 360 Hz** — ez azt jelenti, hogy
  másodpercenként 360 mért feszültségértéket rögzítünk (egy minta
  ≈ 2,78 ms). Egy 2 másodperces ablak tehát **720 minta**. 360 Hz
  elég finom ahhoz, hogy a QRS-komplexus morfológiája (~80–120 ms) és a
  T-hullám részletei is jól láthatók legyenek, de elég ritka ahhoz,
  hogy egy Holter-felvétel órákon át elférjen szokásos tárolón.
- **Két elvezetés párhuzamosan van felvéve** — az első általában **MLII**
  (módosított II. elvezetés), a második V1, V2, V4 vagy V5. A
  dolgozatban **csak az első csatornát (MLII) használom**.
  - *Miért egy elvezetés?* Mert (1) a hordható eszközök (Apple Watch,
    KardiaMobile, patch-ek) lényegében mind egyelvezetésűek, tehát a
    klinikai alkalmazási cél is ez; (2) a rendszer mérnöki szempontból
    egyszerűbb és a valós idejű pipeline-ban közvetlenül demonstrálható.
  - *Miért éppen MLII?* Mert a II. elvezetés a szív fő elektromos
    tengelyével közel egyvonalban fut, így itt a **QRS-amplitúdó
    maximális** és az R-csúcsok a legmegbízhatóbban láthatók — ezért
    ezt használja a legtöbb ambuláns Holter és a legtöbb publikált
    MIT-BIH tanulmány.
- **Annotációk** — minden szívverést **két kardiológus függetlenül
  kézzel címkézett** (a vitás eseteket egyeztetéssel oldották fel)
  szimbólumokkal: normál `N`, kamrai extraszisztolé `V`, pitvari korai
  ütés `A`, pacemaker `/`, fúziós `F`, köteges zárványozás `L`/`R`,
  szupraventrikuláris `S` stb. — összesen 17 különböző címke. **A
  bináris feladathoz minden nem-`N` szimbólumot „anomáliának" tekintek.**
- **Előszűrés** — 0,5–40 Hz sávszűrő (alapvonal-sodródás és magasfrekvenciás
  zaj elnyomása), majd z-score normalizálás (nulla átlag, egységnyi
  szórás).

A publikus benchmark voltából adódó **fontos korlátok**, amelyeket a
szakdolgozat Diszkussziójában külön ki kell mondani: idős, főleg fehér
amerikai populáció, vezetékes Holter (nem wearable), 1970-es évekbeli
felvételi technika, *egyetlen centrumból*.

### 1.3 A modell egyszerűen elmagyarázva (matek nélkül)

> "Egy *autoenkódert* tanítottam — egy neurális hálózatot, amely
> megtanulja, hogyan tömörítsen egy 2 másodperces EKG-szakaszt 32
> számra, majd hogyan állítsa abból vissza az eredetit. Mivel csak
> normál szívveréseken tanult, a hálózat nagyon jó lesz a normál EKG
> visszaállításában, és nagyon rossz minden szokatlan jel
> rekonstruálásában. Az eredeti és a visszaállított jel közti különbség
> a **rekonstrukciós hiba** — kicsi, ha normál a beat, nagy, ha
> rendellenes."

### 1.4 Teljesítménymutatók, amiket meg kell említened

| Mérőszám | Érték | Köznyelvi jelentés |
|---|---|---|
| **ROC-AUC** | 0,972 | mennyire jól választja szét a normált a rendellenestől minden küszöbnél |
| **F1 érték** | 0,85 | a precízió és a találati arány harmonikus közepe a választott küszöbnél |
| **Szenzitivitás (recall)** | 0,91 | a rendellenes ütések 91 %-át elkapja |
| **Specificitás** | 0,96 | a normál ütések 96 %-át helyesen normálnak ítéli |
| **Precízió** | 0,83 | amikor riaszt, az esetek 83 %-ában tényleg rendellenes |
| **Küszöb (rekonstrukciós hibán)** | 0,0434 | a tanító F1-re optimalizált küszöb |
| **Riasztási késleltetés** | < 1,5 mp | 2-a-3-ból megerősítés után |

**Részletes magyarázat — fontos a kardiológusnak, hogy értse, mit számolunk**

A metrikák a tesztkészleten *szívverésenként* (2 másodperces ablak, egy
annotált szívverésre centrálva) négy számlálóból adódnak:

- **TP (igaz pozitív)** — rendellenes szívverést a modell rendellenesnek ítélt
- **TN (igaz negatív)** — normál szívverést a modell normálisnak ítélt
- **FP (hamis pozitív)** — normál szívverést a modell rendellenesnek jelölt („fals riasztás")
- **FN (hamis negatív)** — rendellenes szívverést a modell kihagyott

Ebből:

- **Szenzitivitás = recall = TP ∕ (TP + FN)** — „a valóban beteg ütések
  hány százalékát találom meg?". 0,91 azt jelenti, hogy minden 100
  rendellenes szívverésből **91-et elkapok, 9-et kihagyok**. Ez a
  *betegekre* nézve fontos szám — ha alacsony, veszélyes ritmusokat
  mulaszt el.
- **Specificitás = TN ∕ (TN + FP)** — „a valóban egészséges ütések hány
  százalékát hagyom békén?". 0,96 azt jelenti, hogy minden 100 normál
  szívverésből **96-ot helyesen normálnak ítélek, 4-re tévesen
  riasztok**. Ez a *munkafolyamatra* nézve fontos: ha alacsony, a
  kardiológust elborítják a fals riasztások.
- **Precízió = pozitív prediktív érték = TP ∕ (TP + FP)** — „amikor a
  modell riaszt, milyen gyakran van igaza?". 0,83 = **5 riasztásból kb.
  4 valódi** anomália, 1 fals.
- **F1 = a precízió és a recall harmonikus közepe** —
  `F1 = 2·P·R ∕ (P + R)`. Azért *harmonikus* közép, mert így ha akár a
  precízió, akár a recall nagyon alacsony, az F1 is alacsony lesz
  (nem lehet „kicsaljuk" egy metrikát a másik rovására). A
  választott küszöböt úgy hangoltuk, hogy az F1-et **maximalizálja**
  a tanító adatokon.
- **ROC görbe és ROC-AUC** — a küszöb értékét fokozatosan változtatva
  minden beállításhoz kiszámítjuk a szenzitivitást (y-tengely) és az
  `1 − specificitás` értékét (x-tengely), majd ezeket pontokként
  rajzoljuk. Az így kapott **ROC-görbe alatti terület (AUC)** a
  modell *küszöbtől független* rangsorolási képességét méri: **ha
  véletlenszerűen kiválasztok egy rendellenes és egy normál ütést,
  mekkora a valószínűsége, hogy a modell a rendellenesre ad nagyobb
  rekonstrukciós hibát?** 1,00 = tökéletes elválasztás,
  0,50 = vakon tippelés. A **0,972** azt mondja, hogy bármely
  véletlenszerű anomália–normál párnál ~97 %-ban magasabb hibát ad az
  anomáliára. Ez a szám a leghitelesebb összképes minőségjelző, mert
  nem függ a pontos küszöbbeállítástól.
- **Riasztási késleltetés** — mennyi idő telik el egy valódi anomália
  kezdetétől a rendszer riasztásáig. Értéke < 1,5 s, mert az 1,5 s-ig
  tartó 2-a-3-ból smoother ablak-konfirmálás a legszűkebb keresztmetszet.

Mondd ki őszintén: „Ezek a számok a MIT-BIH 15 %-os teszt-szeletére
vonatkoznak (szívverésre centrált 2 másodperces ablakok). Még nem
validáltam adatbázison kívüli betegeken, teljes 24 órás Holter-felvételen,
vagy realisztikus zajjal terhelt jelen. Pontosan ezek azok a területek,
ahol a tanácsodat kérném."

### 1.5 A 2 másodperces ablak és a valós idejű feldolgozás

Ezt a részt a klinikusok a leginkább szeretik megérteni, szóval légy
konkrét:

> "A modell soha nem néz egyetlen mintát vagy egyetlen szívverést
> önmagában. Mindig egy **2 másodperces szakaszt** dolgoz fel
> (720 minta 360 Hz-en) — jellemzően 2-3 szívverést. Minden **fél
> másodpercben** új átfedő 2 másodperces szakaszt vágunk ki, és
> végigfuttatjuk a modellen. Így másodpercenként négy
> rekonstrukciós-hiba értéket kapunk, ahol minden érték 1,5
> másodpercnyi adatban átfedi az előzőt.
>
> Riasztás csak akkor sül el, ha **a legutolsó három értékből kettő
> meghaladja a küszöböt** — ez kiszűri az egyetlen ablakra korlátozódó
> zajokat (rossz elektróda-érintkezés, mozgási műtermék).
>
> A legrosszabb esetben tehát körülbelül 1,5 másodperc a riasztási
> késleltetés: ennyi idő alatt látunk két megerősítést egy tartós
> rendellenesség kezdetét követően. A rendszer **folyamatos
> monitorozásra** van tervezve, nem egyetlen szívverés osztályozására."

**Kompatibilitás a tanítási feldolgozással — fontos részlet**

A tanításkor (és a 0,972 ROC-AUC / 0,85 F1 mérésekor) az ablakok
**R-csúcsra centráltak** voltak és a teljes jel **globális
normalizálásával** készültek. A valós idejű pipeline-ban két dolog
eltér, és ezt őszintén be kell mutatni:

1. **Normalizálás** — megoldva. A pipeline első ~30 másodpercében
   „bemelegítő" (warmup) szakaszt futtatunk, amely a stream statisztikáit
   felhasználva **ugyanabba a skálába állítja** a jelet, mint a
   tanítási globális normalizálás. Így a rekonstrukciós hibák is
   ugyanabba a tartományba esnek → **ugyanazt az F1-optimális küszöböt
   használom (0,0434), mint a batch kiértékelésnél**. (Ez egy korábbi,
   gyengébb „ablakonkénti" normalizálásból származó 0,0591 küszöböt
   váltott le.)
2. **Szívverés-centrálás** — nem megoldható valós időben, mert az
   R-csúcsokat a monitorozás *közben* látjuk. Helyette fix 0,5 mp-es
   stride-dal csúsztatjuk az ablakot. A csúsztatás miatt egy anomália
   néha „szélén van" az ablaknak, ezért a **2-a-3-ból smoother** (lásd
   fent) nélkülözhetetlen kompenzáció: biztosítja, hogy egy valódi
   rendellenes eseményt legalább két átfedő ablak is elkap.

Eredmény: ugyanazt a modellt és ugyanazt a küszöböt használjuk, mint a
batch kiértékelésben; a *skála* megegyezik, a *szívverésre centrálás*
nem. A tesztkészletes számokat a szakdolgozatban ezért **batch (ideális
feltételek között) mért** jelzővel közlöm, és külön jelzem, hogy a
valós idejű változaton a demó-felvételek (208-as és mások) is várhatóan
közeli — de szigorúan *még validálandó* — értékeket hoznak.

Mutasd meg neki a GUI-t: scrollolj végig, mutass rá a maradék-jelre
amikor átlépi a szaggatott küszöbvonalat, mutass rá a WATCHING (sárga)
→ ANOMALY (piros) állapotváltásra.

### 1.6 A példa-szívverések — `clinical_proof.pdf`

Add a kezébe a kinyomtatott PDF-et. 16 példa szívverés, négy kategóriában:

- **Igaz pozitív** — a modell jól észlelt rendellenes szívverést (4 példa)
- **Hamis negatív** — a modell **kihagyott** egy rendellenes szívverést (4 példa)
- **Hamis pozitív** — a modell **tévesen** riasztott normál szívverésen (4 példa)
- **Igaz negatív** — a modell jól minősített normál szívverést (4 példa)

> "Minden oldalon egy 2 másodperces ablak látható az eredeti EKG-vel
> (kék) és a modell rekonstrukciójával (narancs). A piros árnyékolt
> rész köztük a rekonstrukciós hiba. Egy szívverést akkor jelez a
> modell rendellenesnek, ha ez a hiba meghaladja a küszöböt
> (jelenleg **0,0434** — a tanító F1-re optimalizált érték)."

A PDF-ben szereplő 16 példa-szívverésen a modell **95,9 %
szenzitivitást** és **97,6 % specificitást** ért el. Légy őszinte,
hogy ezek picit magasabbak, mint a teszt-szelet 91 / 96 %-a — egyes
felhasznált felvételek a tanítóhalmazban is benne voltak. A PDF
*vizuális* áttekintésre szolgál, nem teljesítménymérésre.

Kérd meg, hogy a hamis pozitív és hamis negatív példáknál külön
álljatok meg — ezekből tanulsz a legtöbbet.

### 1.6b Szigorú validáció — `clinical_blind.pdf`

A `clinical_proof.pdf` *kvalitatív* áttekintés (látja a modell
döntését és reagál rá). Igazi szakértői validációhoz add neki a
**`clinical_blind.pdf`** dokumentumot is — ugyanaz a 16 szívverés
véletlenszerű sorrendben, a modell döntése és a referencia-címke
**elrejtve**. Ő maga jelöli be NORMÁL / RENDELLENES / OLVASHATATLAN
kategóriákba anélkül, hogy tudná, mit állapítottunk meg mi.

Utána összevetjük az ő címkéit:

| Összevetés | Mit mond el |
|---|---|
| Kardiológus vs MIT-BIH címkék | Maga a publikus referencia is klinikailag védhető? |
| Kardiológus vs modell | Egyetért-e a modell egy valós szakértővel? |
| Modell vs MIT-BIH | A publikált F1=0,85, ami már megvan |

A két első összevetésre Cohen-féle kappa együtthatót számolunk. A
megoldókulcs itt található: `docs/clinical_proof/blind_answer_key.csv`
— **NE mutasd meg neki, mielőtt címkézne.** Ez az a lépés, ami a
találkozót „szakértői véleményből" igazi „szakértői validációvá"
emeli — a védési bizottság ezt fogja keresni, ha rigorózus.

### 1.7 Mit szeretnél tőle

> "Három dologban kérnék segítséget: (1) klinikai validitás — azokat az
> aritmiákat észlelem, amelyek *klinikailag is fontosak*?
> (2) működési illeszkedés — hasznos lenne-e egy ilyen rendszer abban a
> munkafolyamatban, amit Te ismersz, és hová illeszkedne?
> (3) értéknövelő érvek — mibe kerül egy elmulasztott aritmia vagy egy
> téves riasztás, pénzben vagy beteg-kimenetelben, hogy a szakdolgozat
> költség-haszon elemzéséhez konkrét számokat tudjak hozni?"

---

## 2. rész — A kérdések, témák szerint

Jegyzeteld le **szó szerint** — a kardiológus pontos szóhasználata
arany a szakdolgozat Diszkusszió fejezetéhez.

### A. Klinikai validitás (mely anomáliák számítanak igazán)

- A1. A MIT-BIH 17 különböző nem-normál szívverés-típust egyetlen
  „anomália" osztályba sorol. Ezekből **mely kategóriák klinikailag a
  legfontosabbak** egy szűrőeszköznél, és melyeket lehet nyugodtan
  figyelmen kívül hagyni? (pl. VES vs PAC vs pitvarfibrilláció vs
  kamrai tachikardia)
- A2. Vannak-e olyan rendellenességek, amelyeket egy **egyelvezetéses
  EKG nem tud megbízhatóan kimutatni**, és ezeket a szakdolgozatban
  külön ki kell zárnom? (pl. STEMI, Tawara-szárblokk, P-hullám
  morfológia csak II. elvezetésben)
- A3. Klinikai biztonsági szempontból mi rosszabb: **egy rendellenes
  szívverés elmulasztása** egy hosszú felvételen, vagy **tíz normál
  szívverés** rendellenesnek jelölése? Miért? (ez határozza meg, hogy
  hogyan állítom be a küszöböt)
- A4. Egyáltalán a **szívverésenkénti észlelés** a helyes keret?
  Inkább **ritmus-epizód szintű** észlelést szeretnél (pl. „30
  másodperces pitvarfibrilláció" a „ez a szívverés rendellenes"
  helyett)?
- A5. A gyakorlatban **hogyan kezelitek a zajos vagy műtermékes
  EKG-t**? A rendszer csendben dobja el az olvashatatlan ablakokat,
  vagy jelezzen, hogy romlott a jelminőség?

### B. Teljesítmény-kompromisszumok

- B1. ROC-AUC = 0,972 — ez **klinikailag értelmes** szám, vagy egy
  ilyen eszközt csak 0,99+ felett vennél komolyan? Mihez vagy szokva
  hasonlítani?
- B2. 0,91-es szenzitivitással és 0,96-os specificitással egy 24
  órás Holter (~100 000 szívverés, ~5 % anomália arány) napi
  **~3 800 hamis riasztást** generál betegenként (95 000 normál
  × 0,04 hamis-pozitív arány). Ez kezelhető munkamennyiség, vagy
  10× alacsonyabbnak kell lennie?
- B3. Milyen **hamis-riasztási arány** miatt szűnnél meg személyesen
  egy hét alatt megbízni a rendszerben?
- B4. A másodperc alatti riasztási késleltetés — **klinikailag
  számít-e** egyáltalán, vagy „percen belül" elég lenne a járóbeteg
  esetek többségében?

### C. Munkafolyamatba illesztés

- C1. Hol illeszkedne egy ilyen eszköz a **mostani munkafolyamatba** —
  sürgősségi triázs, járóbeteg Holter-elemzés, intenzív osztályos
  folyamatos monitorozás, alapellátási szűrés, hordható eszköz?
- C2. Milyen **riasztási formátum** lenne számodra a leghasznosabb —
  azonnali képernyős villogás, műszak végi összegzés, e-mailes PDF,
  EMR-be integrált értesítés? (A szakdolgozatban képernyős +
  SQLite-napló van; mit tennél hozzá?)
- C3. Ki **reagál a riasztásra** — az ágy melletti nővér, az ügyeletes
  kardiológus, vagy maga a beteg egy hordhatón? A jó válasz ettől függ.
- C4. Ha a rendszer rendellenesnek jelöl egy szívverést, milyen
  **információ kell mellé** a döntéshez — a nyers hullámforma, a napszak,
  a beat-to-beat előzmény, a beteg korábbi alapszintje, a modell
  rekonstrukciójának overlay-e?
- C5. Mennyi ideig tűrnéd el, hogy a rendszer **kiessen**, mielőtt
  visszatérnél a kézi monitorozáshoz?

### D. Pénzügyi / költségelemzési keret

Ezek a kérdések fordítják át a CS szakdolgozatot olyasmivé, amit egy
kórházi beszerző is el fog olvasni. Még a hozzávetőleges számok is
értékesek — kérj **tartomány**-t, ha pontos szám nincs.

- D1. **Mibe kerül egy iszkémiás stroke** a magyar (vagy európai)
  egészségügyben, a teljes ellátási útvonalon (akut felvétel +
  rehabilitáció + termelékenység-veszteség)? Akár tartomány. A
  pitvarfibrilláció a megelőzhető strokeok vezető oka, így bármilyen
  korábbi felismerő eszköz stroke-megelőzést vásárol.
- D2. Egy **elmulasztott paroxizmális pitvarfibrillációs epizód**
  becsült költsége egy olyan betegnél, aki később strokeot kap —
  milyen pénzügyi láncot tulajdonítanál neki?
- D3. Mibe kerül egy **téves kardiológiai beutaló** — a konzultáció,
  az utánkövető echo vagy 24 órás Holter, a beteg munkából kiesett
  napja? Ez a költség egy zajos modell ára.
- D4. Mennyibe kerül egy szabványos **24 órás Holter-elemzés**
  (asszisztens-idő + kardiológus olvasási idő), és nagyjából mennyi
  ideig olvas a kardiológus egy ilyet? Egy automatikus triázs ezt
  csökkentené — mennyivel kellene csökkentenie ahhoz, hogy egy
  kórháznak megérje beszerezni?
- D5. **Hordható eszközök kontextusa:** a betegeid mekkora hányada
  visel már valami (Apple Watch, KardiaMobile stb.) ilyet? Az általuk
  generált riasztások hasznosak vagy zajosak számodra?
- D6. Ha egy ilyen rendszer **megszűrné az Apple Watch hamis
  riasztásait**, mielőtt egy kardiológus asztalára érnének, mennyit
  érne ez egy kardiológusnak havonta?
- D7. **Finanszírozási kódok** — ismersz olyan OEP / NEAK kódot,
  amelyik AI-asszisztált EKG-értelmezésre vonatkozik? (Akár egy
  külföldi példa is segít a telepítési érvelésben)

### E. Szabályozási és etikai kérdések

- E1. Az EU-ban való klinikai bevezetéshez **CE-MDR IIa osztályú
  szoftver** mint orvostechnikai eszköz minősítés kellene. Milyen
  dokumentumokat kérnek a kórházi beszerzők, mielőtt egy ilyet
  pilotálnának?
- E2. **GDPR / adatvédelem** — a szakdolgozatban a riasztásokat helyi
  SQLite adatbázisba mentem, fájl-szintű titkosítással a rotált
  log-okon, **nem** rögzítek páciens-azonosító adatot. Ez védhető
  kiindulás, vagy van olyan nyilvánvaló rés, amelyet egy kórházi
  adatvédelmi felelős azonnal kifogásolna?
- E3. Ki a **felelős**, ha a rendszer kihagy egy szívverést, ami
  utóbb halálos aritmia kezdete volt — a kórház, az eszköz
  forgalmazója, a felügyelő kardiológus? Akár egy mondatos vélemény
  is hasznos.
- E4. **Bias** — a MIT-BIH túlnyomórészt idősebb amerikai
  betegekből áll a 70-es és 80-as évekből. Milyen demográfiai
  csoportokra mindenképp **újra kellene tanítanom** a modellt,
  mielőtt mondjuk pediátriai vagy várandós populációra futna?
- E5. A modellnek **meg kell magyaráznia a döntéseit** (saliency
  térkép, szívverés-morfológia összevetés) ahhoz, hogy egy klinikus
  elfogadja? Vagy elég egy fekete-doboz „anomália-pontszám +
  hullámforma overlay"?

### F. Adat- és validációs hiányosságok

- F1. **MIT-BIH-en túl** — milyen adathalmazt javasolnál a következő
  külső validációs lépéshez? (PTB-XL, Chapman-Shaoxing, PhysioNet
  2017, vagy a saját anonimizált klinikai adataitok?)
- F2. Van-e **klinikailag validált alternatív algoritmus**,
  amelyhez benchmarkolnom kéne — akár egy egyszerű is, mint
  Pan-Tompkins + szabály-alapú morfológia? Mi a léc, amit át kell
  ugranom?
- F3. A tanítóhalmaz minden szívverés-típust egyformán kezel.
  Klinikailag hasznosabb lenne **több-osztályos** detektor (VES vs
  AF vs PAC vs egyéb), vagy maradjon a bináris (normál vs
  rendellenes)?
- F4. Tudnál elérhetővé tenni akár csak **5 anonimizált EKG-t**
  valós betegekből, amit szakdolgozat-védés előtt akár csak
  szemmel le tudnék ellenőrizni?

### G. Nyitott kérdések

- G1. **Hogyan kellene kinéznie** egy ilyen eszköznek ahhoz, hogy te,
  mint gyakorló kardiológus, használni akard a heti munkádban?
- G2. Mi a **legnagyobb kockázat**, amit egy ilyen rendszerrel látsz —
  és mi mérsékelné a legjobban?
- G3. Ha egyetlen irányba **terelhetnéd** a szakdolgozatomat, mi az az
  egy változtatás, amit eszközölnél?

---

## 3. rész — A találkozó után

- Írd le a válaszokat 24 órán belül, amíg a megfogalmazás friss.
- Csoportosítsd az idézeteket témák szerint — egyenesen mehetnek a
  szakdolgozat Diszkusszió fejezetébe „Klinikai szempont"
  alfejezetenként.
- A költségszámok kerüljenek egy **Költség-haszon elemzés**
  alfejezetbe — még egy hozzávetőleges tartomány is hitelesebb,
  mint semmilyen szám.
- Ha felajánlott adatot vagy kontaktot, **egy héten belül**
  e-mailben kövesd nyomon.
