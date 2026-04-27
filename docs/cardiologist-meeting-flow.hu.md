# Kardiológusi megbeszélés — lépésről lépésre

A kardiológus **klinikus**, nem AI / műszaki szakember. Minden
technikai fogalmat orvosi nyelvre kell fordítanod. Ez a dokumentum az
első kézfogástól az utolsó kérdésig végigvezet: mit **MONDJ**, mit
**MUTASS**, mit **KÉRDEZZ** minden lépésben.

Becsült teljes idő: **~60 perc**. Ha szűkös az idő, a `*OPCIONÁLIS*`
címkéjű lépések kihagyhatók.

**Amit vigyél magaddal:**

- Laptop a futó GUI-val (`demo_config_optionC.json`, 208-as felvétel)
- Kinyomtatva: `examples_by_type_raw.pdf` — **nyers mV + szabványos
  rózsaszín EKG-papír rács (1 mm = 0,04 mp × 0,1 mV)**. *Ezt használd,
  amikor klinikailag kérdezel egy ütésről.* Ezt olvassa minden nap.
- Kinyomtatva: `examples_by_type_recon.pdf` — **ugyanaz a 51 ütés
  z-scored skálán, a modell rekonstrukciójával**. *Csak akkor használd,
  amikor azt magyarázod, mit látott a modell, és miért döntött úgy.*
- Kinyomtatva: `compare_208.pdf` (per-beat idővonal 208-ra)
- Kinyomtatva: `clinical_blind.pdf` + toll (opcionális vak címkézés)

**Fontos:** a modell-PDF-ek **z-scored** tengelyei (
`examples_by_type.pdf` / `_recon.pdf`) **nem alkalmasak** klinikai
döntéshez — a z-score elveszti az abszolút mV-ot. Ha azt kérdezed
„rendellenes ez az ütés?", **mindig a nyers** PDF-et add a kezébe.

---

## 1. lépés — Az ő háttere (első 5 perc, kávé-fázis)

Vele kezdj, nem veled. Ő beszél, te hallgatsz és jegyzetelsz.

**Kérdezd (egyet egyszerre):**
- „Hol dolgozol — járóbeteg-rendelésen, osztályon, intenzíven,
  kardiológián?"
- „Milyen gyakran olvasol EKG-t? Magad olvasod a Holtereket, vagy
  asszisztens előszűri?"
- „Mely aritmiák a leggyakoribbak a betegeidnél?"
- „Milyen eszközökkel dolgozol most — 12 elvezetéses nyomat, Holter-
  szoftver, valamilyen automatika vagy AI?"

**Mit hallgass ki (jegyzetelj):**
- A *setting*-je (ez határozza meg a célkörnyezetet)
- Hány EKG-t olvas hetente
- Van-e már olyan automata eszköz, amit megbízik / nem bízik meg
- A betegpopulációja (kor, tipikus patológia)

**Miért fontos:** a válasza *mindent meghatároz* utána. Ugyanaz az
eszköz izgalmas egy háziorvosnak, aki eldönteni akarja, kit küldjön
szakrendelésre — de fölösleges egy kórházi intenzíven, ahol már
folyamatos monitoring van minden ágynál.

---

## 2. lépés — Ki vagy és mit építettél (60–90 mp)

Most te. Röviden.

**Mondd:**

> „Szakdolgozatomat írom. Egy automata EKG-anomália-észlelő rendszert
> építettem. A cél egy előszűrő eszköz — a gép jelez, ha valami nem
> tűnik normálisnak; a döntést a kardiológus hozza. **Nem** akarom
> helyettesíteni a szakember diagnózisát.
>
> Csak *normál* szívveréseken tanítottam. A modell **sosem látott
> aritmiát** tanítás közben. Futásidőben, amikor egy szívverés nem
> illeszkedik a megtanult normál mintába, anomáliát jelez. Ezt hívjuk
> **felügyelet nélküli** (unsupervised) tanításnak."

**Számokat még ne mutass.** Csak az alapkeretet: szűrés, nem
diagnosztika; csak normálokat tanított, arra jelez, ami nem illik.

---

## 3. lépés — Élő demo (5 perc)

Lássa működni, mielőtt bármi absztraktot magyaráznál.

**Mutasd:**
- Nyisd meg a GUI-t.
- Mutass rá: a 208-as felvétel kék EKG-je gördül.
- Klikk **Start**.
- Várj ~30 mp-ot. „Ez a warmup fázis, a beteg alapritmusát tanulja."
- Amikor az első **ANOMALY (piros)** sáv felbukkan, mutass rá: „Itt —
  a modell most jelzett egy rendellenes szívverést. Minden riasztás
  adatbázisba is kerül, hogy a kardiológus utólag átnézhesse."

**Kérdezd:**
- „Egy ilyen formátum — képernyős villanás + naplózott esemény —
  hasonlít arra, amit a munkafolyamatodba illesztenél, vagy valami
  mást szeretnél?"

**Mit hallgass ki:** formátum-preferenciák (pop-up vs összegzés vs
EMR-integráció), kit gondol, aki reagálna a riasztásra.

---

## 4. lépés — Az adathalmaz (3 perc)

Helyezd az eszközt az *adatába*, nem az architektúrájába.

**Mondd:**

> „A **MIT-BIH Arrhythmia Database**-en tanítottam — ez a standard
> publikus benchmark ilyen munkához. **48 félórás felvételt** tartalmaz
> **47 alanytól**, a Bostoni Beth Israel Kórházban rögzítették az
> 1970-es évek végén. Minden egyes szívverést **két kardiológus kézzel
> címkézett** — összesen ~110 000 címkézett ütés. Ez a ground truth,
> amin tanítunk és kiértékelünk."

**Légy őszinte a korlátokról:**

> „Három fontos dolog, ami ez az adathalmaz **nem**:
> 1. Régi — 1970-es évek technikája, nem modern wearable.
> 2. A populáció amerikai, főleg idős.
> 3. Egyetlen centrum — nincs kórházközi variabilitás.
>
> Szóval minden szám, amit említek, a **benchmarkon** őszinte. A
> te betegeiden még nincs validálva."

**Kérdezd:**
- „E korlátokat tudva, melyik lenne a **következő** adathalmaz, amire
  validálást kellene futtatnom? Van a klinikádnak anonimizált
  Holter-adata, amit esetleg megoszthatsz?"

---

## 5. lépés — Hogyan működik a rekonstrukció (5 perc)

Ez az egyetlen technikai fogalom, amit **muszáj** értenie. PDF-ekkel
dolgozz.

**Mondd:**

> „A modell egy *autoenkóder*. Gondold úgy, hogy egy specializált
> fénymásoló, amelyiknek csak *normál* szívveréseket mutattak be.
> Amikor kap egy normál ütést, tökéletesen másolja. Amikor kap egy
> rendellenest, a másolat *elcsúszik* — mert ilyen alakot sosem
> látott. Az eredeti és a másolat közötti **különbség** a
> **rekonstrukciós hiba**. Kis hiba = normál. Nagy hiba = valami nem
> stimmel."

**Mutasd az `examples_by_type_recon.pdf`-et:**

- Nyiss meg egy **TRUE NEGATIVE normál ütés** oldalt. Mondd: „A kék
  az eredeti, a narancs a modell rekonstrukciója. Látod, ahogy
  majdnem tökéletesen követik egymást — kis hiba."
- Nyiss meg egy **TRUE POSITIVE PVC** oldalt. Mondd: „Ugyanaz a modell,
  de itt a narancs vonal láthatóan nem tudja követni ezt a széles,
  rendellenes ütést. A kék és narancs közti hézag — a piros
  árnyékolás — a hiba, és elég nagy ahhoz, hogy átlépje a riasztási
  küszöböt."

**Kérdezd:**
- „Elfogadható-e ez mint anomália-észlelési elv? Megbíznál egy ilyen
  rendszerben — vagy elvárnád, hogy azt is elmondja, *milyen* típusú
  az anomália, ne csak azt, hogy van?"

---

## 6. lépés — Miért 2 másodperc, nem egy szívverés (5 perc)

Most az ő szakértelme kerül előtérbe. Ez az első *valódi* kérdés,
ahol az ő válasza megváltoztatja a szakdolgozatot.

**Mondd:**

> „A modell **2 másodpercnyi** EKG-t lát egyszerre — kb. 2–3 szívverés.
> Le is futtattam egy kísérletet: *egy* szívverés / ablak változat
> F1-et **0,70** kapott, szemben a 2 mp-es **0,85**-tel — jelentős
> visszaesés. Szerintem azért, mert egy ütés elveszíti az RR-intervallumot
> és a szomszédos ütésekkel való összehasonlítást, és ez a két dolog,
> amit ti használtok, amikor *kontextusban* ítéltek meg egy ütést. De
> szeretném ezt veled leellenőrizni."

**Kérdezd:**
- „Amikor *te* nézel egy EKG-csíkot és eldöntöd, hogy egy ütés
  rendellenes, inkább a **formáját** nézed, vagy a **szomszéd ütésekhez
  viszonyított időzítését**, vagy mindkettőt?"
- „2 másodperc a megfelelő mennyiségű kontextus, vagy többet szeretnél
  (5, 10 másodpercet)?"

**Mit hallgass ki:** szívverés-szintű vagy ritmus-szintű észlelést
szeretne. Ha „ritmust" mond → future-work pointer epizód-szintű
észlelésre.

---

## 7. lépés — Mely aritmiák a legfontosabbak (5 perc)

Nyitott kérdés. Az ő válasza strukturálja a Results fejezetet.

**Mondd:**

> „A MIT-BIH címkék 17 különböző nem-normál típust egyetlen *anomália*
> osztályba tömörítenek a tanításhoz. Klinikailag nyilván nem egyformák
> — egy PVC nagyon más, mint egy pitvari korai ütés."

**Kérdezd:**
- „Ezek közül — **VES (kamrai extraszisztolé), pitvari korai, fúziós
  ütés, szárblokk, pacemaker, pitvarfibrilláció** — melyiket **muszáj**
  elkapnia egy szűrőeszköznek, és melyiket hagyhatja?"
- „Ha **egyetlen** aritmiát választhatnál, amiben kimagasló legyen az
  eszköz, melyik lenne? Miért?"
- „Vannak olyan patológiák, amelyeket egyelvezetéses EKG **egyszerűen
  nem tud** megbízhatóan kimutatni, és ezeket ki kellene zárnom a
  szakdolgozatban?"

---

## 8. lépés — Mit kap el és mit mulaszt el a modell (10 perc)

A beszélgetés szíve. Számok először, utána példák.

**Mondd:**

> „A 208-as felvételen — 30 perc adat, ~3000 szívverés — a modell:
>
> - **VES (kamrai) esetén ~98,5 %-ot elkap**
> - **Fúziós ütéseknél ~30 %-ot** (fél-VES + fél-normál, morfológiailag
>   átmeneti)
> - **Pitvari korai ütéseknél 0–100 %, felvételfüggően**
>
> A minta konzisztens: kamrai ektopia megbízhatóan elkapva, fúziós
> ütések kihagyva, pitvari változó."

**Mutasd az `examples_by_type_recon.pdf`-et szekciónként:**

1. Lapozz a **V (VES)** részhez. Mutass egy TP-t. „A modell ezt
   tisztán kezeli — szerinted miért?" Hallgasd.
2. Lapozz az **F (fúziós)** részhez. Mutass egy FN-t. „A modell
   kihagyta. Ez *fúziós* címke — fél-kamrai + fél-normál. Klinikailag
   hogy néz ki egy fúziós ütés, és szerinted miért találja határesetnek
   egy csak normálokon tanított autoenkóder?"
3. Mutass egy **FP normált**. „Ezt a normál ütést tévesen jelölte a
   modell. A 2 másodperces ablakban van *bármi*, ami indokolhatja a
   modell zavarát?"

**Itt szolgál rá a pénzére.** Írd le szóról szóra, amit mond minden
kategóriához.

---

## 9. lépés — A metrikák orvosi nyelven (7 perc)

Klinikus, tehát minden metrikát úgy keretezz, mint egy szűrőtesztet,
amit ismer (a PPV/NPV laboratóriumból ismerős).

Rajzold fel a 2x2 táblázatokat papírra, és menj végig rajtuk.

### 1. doboz: 2×2 konfúziós mátrix

|                                | Valóban rendellenes   | Valóban normál        |
|---|---|---|
| Modell azt mondja: RENDELLENES | **Igaz pozitív (TP)** | **Hamis pozitív (FP)** — téves riasztás |
| Modell azt mondja: NORMÁL      | **Hamis negatív (FN)** — kihagyott | **Igaz negatív (TN)** — jól elengedte |

### A négy szám, amit visszavisz magával

| Metrika | Köznyelv |
|---|---|
| **Szenzitivitás** = TP / (TP + FN) | „A valóban rendellenes ütések hány %-át kapom el?" **Nálunk: 91 %.** Ez azt jelenti: 100-ból 9 kimarad. |
| **Specificitás** = TN / (TN + FP) | „A valóban normál ütések hány %-át hagyom helyesen békén?" **Nálunk: 96 %.** Ez azt jelenti: 100-ból 4 normál téves riasztást okoz. |
| **Precízió (PPV)** = TP / (TP + FP) | „Amikor azt mondja rendellenes, az esetek hány %-ában tényleg az?" **Nálunk: 83 %.** 100 riasztásból 17 téves. |
| **F1** | A szenzitivitás és precízió harmonikus közepe — **kiegyensúlyozza az elkapást és a túlriasztást.** A küszöböt úgy választottuk, hogy maximalizálja: 0,85. |
| **ROC-AUC** (0,97) | Képzelj el egy véletlenszerűen kiválasztott normál és egy rendellenes ütést. Mennyi a valószínűsége, hogy a modell a rendellenesnek nagyobb rekonstrukciós hibát ad? **97 %.** Ez a modell küszöbfüggetlen minősége. |

**Hasznos analógiák**, amiket már ismer:

- D-dimer PE-re: ~95 % szenzitivitás, ~45 % specificitás → kevés
  kimaradó, sok téves riasztás (rule-out teszt).
- Troponin MI-re standard cut-off-nál: ~95 % szenzitivitás, ~80 %
  specificitás → hasonló műveleti pont.
- Mammográfia szűrés: ~85 % szenzitivitás, ~90 % specificitás — a
  legközelebbi analógia.

**Kérdezd:**
- „Ezek ismeretében a 91 % / 96 % klinikailag elegendő egy szűrőtesztnek,
  vagy a léc alatt van?"
- „Egy 24 órás Holteren ~100 000 ütéssel ez nagyjából
  **~3800 téves riasztás / beteg / nap**-ot jelent. Elfogadható
  munkateher, vagy **10×-szer kevesebbnek** kell lennie?"

---

## 10. lépés — Hamis negatív vs hamis pozitív csere (5 perc)

Ez az az egy kérdés, ami eldönti a küszöböt.

**Mondd:**

> „Van egy gomb, amit elforgathatunk. Ha csökkentem a riasztási
> küszöböt, az eszköz több rendellenest elkap, **de** több normálra is
> riaszt. Ha emelem, kevesebb téves, **de** több kihagyás. Nincs
> ingyenebéd."

**Kérdezd:**
- „Melyik oldalra térjen el, a *te* betegpopulációdban? Elmulasztott
  valódi VES, vagy túlhívott normál?"
- „A válasz változik-e járóbeteg (kis kockázat, drága téves riasztás)
  és intenzív (nagy kockázat, kimaradás végzetes) között?"

---

## 11. lépés — Vak validáció: mikor és egyáltalán (opcionális, 10 perc)

A Cohen-kappáról kérdezted — egyszerűsítsük le.

### Mi az a Cohen-féle kappa egyszerűen

> „A kappa egy szám 0 és 1 között, ami azt méri, **mennyire értenek
> egyet két olvasók ugyanazon EKG-kon, a véletlenszerű egyetértést
> levonva**. Ha két kardiológus egy tisztán normál adathalmazban 90 %-
> ban azt mondja „normál", első ránézésre egyetértenek — de ennek egy
> része csak azért történik, mert a normálok dominálnak. A kappa levonja
> ezt az alapvonalat. 0 = véletlen szintű egyetértés, 1 = tökéletes.
> Az EKG-irodalomban a kardiológusok közti tipikus kappa 0,6–0,8."

### Két módszer a vak lépésre

**A verzió — teljes vak validáció (legjobb, de ~15 perc az ő idejéből).**

- Add a kezébe a `clinical_blind.pdf`-et + tollat.
- Kérd meg: jelölje be mind a 16 szívverést **NORMÁL / RENDELLENES /
  OLVASHATATLAN** kategóriába, anélkül hogy bármi mást nézne.
- Gyűjtsd be a lapot.
- Utána számold ki a kappát az ő címkéi és (a) a MIT-BIH referencia,
  (b) a modell között. Egy bekezdést írj a szakdolgozatba a háromoldalú
  egyetértésről.

**B verzió — csak kvalitatív áttekintés (ha szűkös az idő).**

- Skippeld a vak lapot.
- Menj végig vele az `examples_by_type_recon.pdf`-en oldalanként (mint a
  8. lépésben).
- Jegyezd le a **szóbeli reakcióját** minden nézeteltérés esetén.
- Ez nem kappa-szintű validáció, de **szakértői klinikai review** —
  tökéletesen védhető a szakdolgozatban mint **kvalitatív értékelés**.

### Elég 16 szívverés a kappához?

Szigorúan véve nem — 16 ütésből a kappa statisztikai megbízhatósága
gyenge (széles konfidencia-intervallum). De egy BSc/MSc szakdolgozathoz
mutatja a **módszert és a szándékot** — ez számít. Ha hajlandó, ~50
ütés (`examples_by_type_raw.pdf`) szorosabb kappát adna.

**Ajánlásom:** A verziót ajánld fel. Ha visszautasítja vagy szűkös az
ideje, menj B verzióra. Mindkettő védhető.

---

## 12. lépés — Költség-haszon (5 perc, itt ő az orákulum)

Valószínűleg nem egészség-gazdász, de a költségeket első kézből látja.

**Kérdezd puhán, nem kikérdezésként:**
- „Nagyságrendben: mennyibe kerül egy iszkémiás stroke az
  egészségügynek, akut + rehab összesen?"
- „Mennyibe kerül egy szabvány 24 órás Holter-elemzés a klinikádon?
  Ebből mennyi a kardiológus olvasási ideje?"
- „Egy téves kardiológiai beutaló — a konzultáció, az utánkövető echo,
  a beteg munkából kiesett napja — összesen mennyi lenne durván?"

Bármely szám, akár nagyságrendi tartomány, bekerül a szakdolgozat
Költség-haszon alfejezetébe. „X klinika kardiológusa szerint…"
idézhető.

---

## 13. lépés — Lezárás: egy nyitott kérdés (2 perc)

Az ő őszinte iránymutatásával zárj.

**Kérdezd:**
- „Ha **egyetlen tervezési döntést** átírhatnál a projektben, mi lenne
  az?"
- „Mi a **legnagyobb kockázat**, amit egy ilyen eszköz valós beteghez
  kerülésénél látsz — és mi mérsékelné a legjobban?"

Írd le a pontos szavait. Ez a két sor gyakran egyenesen a szakdolgozat
Diszkusszió fejezetének a nyitásává válik.

---

## A találkozó után (24 órán belül)

- [ ] Írd le szó szerint a válaszokat egy jegyzetfájlba.
- [ ] Küldj köszönő e-mailt bármilyen megegyezett nyomon követéssel
      (adat, kontakt).
- [ ] Ha bejelölte a vak PDF-et, szkenneld be még aznap, és fuss
      kappa-számolást, amíg a kontextus friss.
- [ ] Vázolj 1–2 bekezdést a Diszkusszióhoz, ami őt idézi.

---

## Egy-soros emlékeztető a zsebedben

> *„Ő szívekhez ért. Én a kódhoz értek. A megbeszélés akkor működik,
> ha mindketten ahhoz szólunk hozzá, amihez értünk, és a másik részről
> a másikat kérdezzük."*
