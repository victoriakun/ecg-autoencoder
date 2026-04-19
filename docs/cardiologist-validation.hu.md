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
> Klinikus első-szűrőnek szánom, nem a kardiológus helyettesítésére. A
> modellt csak normál szívveréseken tanítottam, így nem tudja, hogy az
> aritmiák *milyennek látszanak* — csak azt tudja, hogy mi a *normális*,
> és mindent jelez, ami nem fér bele."

### 1.2 Az adathalmaz — légy konkrét

- **MIT-BIH Arrhythmia Database** — 48 darab fél órás felvétel, ~110 000
  annotált szívverés, 360 Hz mintavételezés, kételvezetéses ambuláns
  Holter. Szinte minden publikált EKG-anomália tanulmány ezt a publikus
  benchmark adathalmazt használja.
- **Annotációk** — minden szívverést két kardiológus kézzel címkézett
  szimbólummal (normál `N`, kamrai extraszisztolé `V`, pitvari korai
  ütés `A`, pacemaker `/`, fúziós `F` stb.). A bináris feladathoz minden
  nem-normál szimbólumot „anomáliának" tekintek.

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
| **F1 érték** | 0,85–0,87 | a precízió és a találati arány egyensúlya a választott küszöbnél |
| **Szenzitivitás** | 0,91 | a rendellenes ütések 91 %-át elkapja |
| **Specificitás** | 0,96 | a normál ütések 96 %-át helyesen normálnak ítéli |
| **Precízió** | 0,83 | amikor riaszt, az esetek 83 %-ában tényleg rendellenes |
| **Késleltetés** | < 1 mp / ablak | másodperc alatti riasztási idő |

Mondd ki őszintén: „Ezek a számok a MIT-BIH 15 %-os teszt-szeletére
vonatkoznak. Még nem validáltam adatbázison kívüli betegeken, teljes
24 órás Holter-felvételen, vagy realisztikus zajjal terhelt jelen.
Pontosan ezek azok a területek, ahol a tanácsodat kérném."

### 1.5 A 2 másodperces ablak — ezt lassan magyarázd

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
> (jelenleg 0,0591)."

Kérd meg, hogy a hamis pozitív és hamis negatív példáknál külön
álljatok meg — ezekből tanulsz a legtöbbet.

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
  **~3 800 hamis riasztást** generál betegenként. Ez kezelhető
  munkamennyiség, vagy 10× alacsonyabbnak kell lennie?
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
