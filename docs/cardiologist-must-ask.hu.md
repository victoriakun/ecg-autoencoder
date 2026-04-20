# Kötelezően megkérdezendő — kardiológusi konzultáció

Egyoldalas emlékeztető. A teljes kérdéssor a
`cardiologist-validation.hu.md`-ben; ezek azok, **amelyek nélkül nem
szabad kijönnöd** a találkozóról.

## Klinikai validitás (nem alku)

- [ ] **Mely szívverés-típusok számítanak klinikailag** — a MIT-BIH
      nem-normál szimbólumai közül (V, F, A, L, R, /, f, S, J…) mely
      típusokat kell egy szűrőeszköznek *feltétlenül* észlelnie, és
      melyek hagyhatók nyugodtan figyelmen kívül? Kérj rangsort.
- [ ] **Hamis negatív vs hamis pozitív csere** — rosszabb egyetlen
      VES-t elmulasztani, mint tíz normál ütést rendellenesnek
      jelölni, vagy fordítva? Ez határozza meg a küszöbbeállítást.
- [ ] **Szívverés-szintű vs epizód-szintű keret** — „ez a szívverés
      rendellenes" vagy „ez a 30-s szakasz pitvarfibrilláció" legyen-e a
      kimenet? Ha epizód, át kell gondolni az ablakozási konvenciót.
- [ ] **Egy-elvezetés korlátai** — nevezze meg az összes aritmiát /
      pathológiát, amelyet egyelvezetéses EKG **nem tud megbízhatóan
      kimutatni** (STEMI lokalizáció, egyes szár-blokkok, P-hullám
      morfológia más elvezetésekben…) — így a szakdolgozat
      Limitations-e kimondhatja.

## Teljesítmény, ami valóban elfogadható

- [ ] **Elfogadható riasztások napi Holterre vetítve** — ő a vevő. A
      jelenlegi szenz./spec. mellett ~3800 hamis riasztás/nap/beteg; ez
      használható, vagy 10× kevesebbnek kell lennie, mielőtt bevezetné?
- [ ] **ROC-AUC 0,97 kontextus** — klinikailag értelmes szám, vagy a
      személyes léce 0,99+? Mihez hasonlítja fejben?
- [ ] **Riasztási késleltetési keret** — a másodperc alatt fontos a
      munkafolyamatában, vagy „percen belül" is elég járóbetegnél?

## Munkafolyamatba illeszkedés

- [ ] **Hová illeszkedik** — sürgősségi triázs, járóbeteg Holter
      kiértékelés, intenzív folyamatos monitor, wearable, alapellátási
      szűrés? *Egy* elsődleges felhasználás, nem lista.
- [ ] **Riasztási formátum, amit ténylegesen használna** — képernyős
      villogás, műszak végi összegzés, EMR-integráció, emailes PDF?
      Milyen a *jelenlegi* eszközpark?
- [ ] **Ki reagál a riasztásra** — nővér, ügyeletes kardiológus,
      maga a beteg egy hordhatón? A jó válasz teljesen más designot
      kíván.

## Pénzügyi keret (költség-haszon a szakdolgozathoz)

- [ ] **Egy iszkémiás stroke költsége** a magyar rendszerben (akut
      + rehabilitáció + termelékenység-veszteség). Tartomány is jó.
- [ ] **Egy elmulasztott paroxizmális pitvarfibrillációs epizód
      költsége** egy olyan betegnél, aki később strokeot kap.
- [ ] **Egy szabványos 24-órás Holter-elemzés költsége** — asszisztens-
      idő + kardiológus olvasási idő. Ebből látszik, mennyit spórolna
      egy automata triázs.
- [ ] **Egy téves kardiológiai beutaló költsége** (normál szívverést
      jelöl rendellenesnek → konzultáció + echo + munkanap kiesés). Ez
      a zajos modell ára.

## Adat- és validációs hiányosságok

- [ ] **Melyik külső adathalmaz** legyen a *következő* validációs cél —
      PTB-XL, Chapman-Shaoxing, PhysioNet 2017, vagy a saját
      anonimizált klinikai adatai?
- [ ] **Benchmark algoritmus**, amit meg kell vernem — akár egy
      egyszerű is, mint Pan-Tompkins + szabály-alapú morfológia. Mi a
      léc, amit át kell ugranom?
- [ ] **5 anonimizált EKG** vizuális ellenőrzésre a védés előtt —
      meg tudja osztani?

## Amit validációként magában a találkozón kell csinálni

- [ ] Add a kezébe a **`clinical_blind.pdf`** dokumentumot *az első
      PDF-ek előtt*. Címkézze fel a 16 szívverést NORMÁL / RENDELLENES
      / OLVASHATATLAN kategóriákba anélkül, hogy a modell döntését
      látná. Kérd el a lapot. Utána számoljunk Cohen-féle kappa-t az ő
      címkéi, a MIT-BIH referencia és a modell között — ez az a szigor,
      amit a védési bizottság keres.

## Egy „mit változtatnál" nyitott kérdés

- [ ] **Legnagyobb kockázat**, amit egy ilyen rendszerrel lát — és mi
      mérsékelné a legjobban. Egy mondat is arany.
- [ ] Ha **egyetlen tervezési döntést** átírhatna a projektben, mi lenne az?

---

**24 órán belül a találkozó után:** írd le a válaszokat szó szerint (a
megfogalmazása egy-az-egyben mehet a Diszkusszió fejezetbe). A
költségszámok kerüljenek egy Költség-haszon alfejezetbe. Ha adatot /
kontaktot ajánlott, **egy héten belül** email-ben kövesd.
