# Cardiologist meeting — step-by-step flow

The cardiologist is a clinician, **not** an AI / tech person. Every
technical concept has to land in medical language. This document walks
the meeting from the first handshake to the last question, with exactly
what to **SAY**, what to **SHOW**, and what to **ASK** at each stage.

Estimated total time: ~60 minutes. If she's short on time, steps
marked `*OPTIONAL*` can be dropped.

Files to have on the laptop and printed:

- GUI running with `demo_config_optionC.json` (record 208)
- Printed `examples_by_type_raw.pdf` — **raw mV with standard pink ECG-
  paper grid (1 mm = 0.04 s × 0.1 mV)**. *Use this whenever you ask
  her to clinically judge a beat.* This is the format she reads every
  day.
- Printed `examples_by_type_recon.pdf` — **same 51 beats but z-scored
  with the model's reconstruction overlaid**. *Use this only when
  explaining what the model saw / why it decided what it did.*
- Printed `compare_208.pdf` (per-beat timeline, record 208)
- Printed `clinical_blind.pdf` + a pen (for optional blind labelling)

**Important:** the **z-scored** axes in the model PDFs (
`examples_by_type.pdf` / `_recon.pdf`) are useless for clinical
judgment — z-score throws away absolute mV. Always hand her the
**raw** PDF first when asking "is this beat abnormal?"

---

## Step 1 — Her background (first 5 minutes, coffee-stage)

Start with her, not with you. She talks, you listen and note.

**Ask (one at a time):**
- "Where do you work — outpatient clinic, hospital ward, ICU, cardiology
  department?"
- "How often is ECG part of your day? Do you personally read Holter
  recordings, or does a technician pre-screen them for you?"
- "Which arrhythmias do you see most in your patients?"
- "What tools do you currently use — 12-lead printouts, Holter software,
  any AI or automated reading?"

**What you're listening for (write it down):**
- Her *setting* (defines the target deployment environment)
- How many ECGs she actually reads per week
- Whether there's already an automated tool she trusts or distrusts
- Her patient population (age, typical pathology)

**Why this matters:** her background frames *every answer* that follows.
The same tool is exciting for a GP screening out whom to refer, but
useless for a hospital ICU where every bed already has continuous
telemetry.

---

## Step 2 — Who you are and what you've built (60–90 seconds)

Now your turn. Keep it short.

**Say:**

> "I'm a thesis student. I built an automated ECG anomaly-detection
> system. The goal is a first-pass screening tool — the computer looks
> at the signal and raises its hand whenever something doesn't look
> normal; a human cardiologist then decides what to do. I'm **not**
> trying to replace the cardiologist's diagnosis.
>
> I trained it only on *normal* beats. It never saw arrhythmias during
> training. At runtime, when a beat doesn't match the learned 'normal'
> pattern, it flags an anomaly. This is called *unsupervised learning*."

**Don't yet show numbers.** Just establish the frame: screening, not
diagnosing; learned only normal, flags what doesn't fit.

---

## Step 3 — Live demo (5 minutes)

She should see it working before any abstract explanation.

**Show:**
- Open the GUI on your laptop.
- Point at record 208 scrolling in blue.
- Click **Start**.
- Wait ~30 seconds. Say "this is the warm-up, it's learning the
  patient's baseline."
- When the first **ANOMALY (red)** banner fires, point to it: "There —
  the model just flagged an abnormal beat. Every alert is also written
  to a database so the cardiologist can review them afterwards."

**Ask:**
- "Does this kind of alert — an on-screen flash plus a logged event —
  look like something you'd want in your workflow, or would you want
  something different?"

**Listen for:** format preferences (pop-up vs summary vs EMR
integration), who she thinks would *react* to the alert.

---

## Step 4 — The dataset (3 minutes)

Now ground the tool in its *data*, not its architecture.

**Say:**

> "I trained on the **MIT-BIH Arrhythmia Database** — the standard
> public benchmark for this kind of work. It contains **48 half-hour
> recordings from 47 subjects**, recorded at Boston's Beth Israel
> Hospital in the late 1970s. Every single heartbeat was **labelled by
> hand by two cardiologists** — about 110 000 labelled beats in total.
> That's the ground truth we train and evaluate against."

**Be upfront about limitations:**

> "Three important things this dataset is **not**:
> 1. It's older — 1970s equipment, not modern wearables.
> 2. The population is American, mostly older patients.
> 3. It's a single centre — no cross-hospital variation.
>
> So any number I quote is honest **on this benchmark**. It hasn't
> yet been validated on your patients."

**Ask:**
- "Given those limitations, what would be the most convincing **next
  dataset** to validate on? Do you have access to any anonymised
  Holter data from your clinic?"

---

## Step 5 — How the reconstruction works (5 minutes)

This is the one technical idea she **has** to understand. Use the PDFs.

**Say:**

> "The model is an *autoencoder*. Think of it as a very specialised
> photocopier that has only ever been shown pictures of normal heart
> beats. When I give it a normal beat, it copies it perfectly. When I
> give it an abnormal beat, its copy is *wrong* — because it has never
> seen that shape. The **difference** between the original and its
> copy is the **reconstruction error**. Small error = normal. Big
> error = something off."

**Show `examples_by_type_recon.pdf`:**

- Open a page with a clean **TRUE NEGATIVE normal beat**. Say:
  "Blue is the original ECG, orange is what the model drew. Notice
  how they track each other almost perfectly — small error."
- Open a page with a **TRUE POSITIVE PVC**. Say: "Same model, but here
  the orange line clearly struggles to match the wide abnormal beat.
  That gap between blue and orange — the red shading — is the error,
  and it's big enough to cross our alert threshold."

**Ask:**
- "Does that concept make sense as a way to detect anomalies? Would
  you trust a system built this way — or would you expect it to
  explain *what kind* of anomaly, not just that there is one?"

---

## Step 6 — Why 2 seconds, not one beat (5 minutes)

Now *her* expertise becomes central. This is the first real question
where her answer changes the thesis.

**Say:**

> "The model looks at **2 seconds** of ECG at a time — roughly 2–3
> consecutive beats. I could have used *one* beat per window, but I
> ran the experiment: the single-beat version scored F1 **0.70**
> instead of **0.85** — a big drop. I believe that's because a single
> beat loses the RR-interval and the neighbour-beat comparison, which
> are the two things cardiologists use to judge a beat in context.
> But I want to check that reasoning with you."

**Ask:**
- "When *you* look at an ECG strip and decide a beat is abnormal, are
  you mostly looking at the beat's **shape**, or at its **timing
  relative to neighbours**, or both?"
- "Is 2 seconds the right amount of context, or would you want more
  (5, 10 seconds)?"

**Listen for:** whether she prefers beat-level or rhythm-level
detection. If she says "rhythm" → future-work pointer to episode-level
detection.

---

## Step 7 — Which arrhythmias matter most (5 minutes)

Open question. Her answer structures the Results chapter.

**Say:**

> "The MIT-BIH labels group 17 different non-normal beat types into
> one *anomaly* class for training. Clinically they're obviously not
> equal — a PVC is very different from an atrial premature beat."

**Ask:**
- "Of these categories — **PVC (ventricular ectopic), atrial premature,
  fusion beats, bundle-branch block, paced beats, atrial fibrillation**
  — which **must** a screening tool catch, and which can it miss?"
- "If you had to pick **one** arrhythmia for the tool to be excellent
  at, which would it be? Why?"
- "Are there pathologies a single-lead ECG simply **cannot** detect
  reliably that I should rule out of scope?"

---

## Step 8 — What the model catches and what it misses (10 minutes)

The heart of the discussion. Numbers first, then show examples.

**Say:**

> "On record 208 — 30 minutes of data with ~3000 beats — the model
> catches:
>
> - **98.5 % of PVCs** (nearly all ventricular ectopy)
> - **~30 % of fusion beats** (half-PVC half-normal — morphologically
>   ambiguous)
> - **0–100 % of atrial premature beats depending on record**
>
> The pattern is consistent: ventricular ectopy caught reliably,
> fusion beats missed, atrial beats variable."

**Show `examples_by_type_recon.pdf` by section:**

1. Flip to **V (PVC)** pages. Show a TP. "The model handles this
   cleanly — why do you think it does?" Listen.
2. Flip to **F (Fusion)** pages. Show a FN. "The model missed this.
   It's labelled *fusion* — a half-ventricular half-normal beat.
   Clinically, what does a fusion beat look like to you, and why do
   you think an autoencoder trained on normals would find it
   borderline?"
3. Flip to an FP normal. "This is a normal beat the model flagged
   incorrectly. Looking at the 2-second window, is there *anything* in
   it that might justify the model's confusion?"

**This is where she earns her keep.** Write verbatim what she says
about each category.

---

## Step 9 — Metrics explained in medical language (7 minutes)

She's a clinician, so frame every metric like a screening test she
already knows (she will know these from NPPV / PPV in lab medicine).

Write these two-by-two boxes on paper and walk through them.

### Box 1: the 2×2 confusion matrix

|                       | Beat is really abnormal (truth) | Beat is really normal (truth) |
|---|---|---|
| Model says ABNORMAL   | **True positive (TP)** — correct catch | **False positive (FP)** — false alarm |
| Model says NORMAL     | **False negative (FN)** — missed abnormal | **True negative (TN)** — correctly cleared |

### The four numbers she should carry away

| Metric | Plain English |
|---|---|
| **Sensitivity** = TP / (TP + FN) | "Of all the truly abnormal beats, what fraction did I catch?" (same as recall). **Ours: 91 %.** Means 9 out of 100 abnormal beats are missed. |
| **Specificity** = TN / (TN + FP) | "Of all the truly normal beats, what fraction did I correctly leave alone?" **Ours: 96 %.** Means 4 out of 100 normal beats trigger a false alarm. |
| **Precision** (PPV) = TP / (TP + FP) | "When my test says abnormal, what fraction is actually abnormal?" **Ours: 83 %.** Means 17 out of 100 alerts are false alarms. |
| **F1-score** | Harmonic mean of sensitivity and precision — **balances catching vs over-flagging.** We chose the threshold to maximise this: 0.85. |
| **ROC-AUC** (0.97) | Imagine picking one normal beat and one abnormal beat at random. What's the probability the model assigns the abnormal one a higher reconstruction error? **97 %.** This is the threshold-independent quality of the model. |

**Useful analogies** she'll already know:

- D-dimer for PE: ~95 % sensitivity, ~45 % specificity → very few
  missed, lots of false alarms (rule-out test).
- Troponin for MI at standard cut-off: ~95 % sensitivity, ~80 %
  specificity → similar operating point to ours.
- Mammography screening: ~85 % sensitivity, ~90 % specificity — closest
  analogy to what our tool would do.

**Ask:**
- "Given those comparators, does 91 % / 96 % sound clinically
  adequate for a screening tool, or is it below the bar?"
- "On a 24-hour Holter with ~100 000 beats, this implies roughly
  **~3 800 false alarms per patient per day**. Is that workload
  acceptable, or would it have to be **10× less**?"

---

## Step 10 — The false-negative vs false-positive trade-off (5 minutes)

This is the single question that sets the threshold.

**Say:**

> "There's a dial we can turn. If I lower the alert threshold, the
> tool catches more abnormal beats **but** flags more normal ones as
> well. If I raise it, fewer false alarms **but** more missed
> abnormals. There's no free lunch — one goes up when the other goes
> down."

**Ask:**
- "Which side do you want this tool to err on, for the patient
  population you actually see? Missing a real PVC, or over-calling a
  normal one?"
- "Does your answer change depending on whether the patient is
  outpatient (low risk, false alarms are expensive) versus ICU (high
  risk, a miss can be fatal)?"

---

## Step 11 — Blind validation: when and whether to do it (optional, 10 minutes)

You asked about Cohen's kappa — let's make this simple.

### What Cohen's kappa actually is

> "Kappa is a number between 0 and 1 that measures **how much two
> readers agree on the same ECGs, adjusted for the agreement you'd
> get by pure chance**. If two cardiologists both say 'normal' on 90 %
> of a clean dataset, they look like they agree — but some of that
> agreement happens just because normals dominate. Kappa subtracts
> that baseline. 0 = chance-level agreement, 1 = perfect. In the ECG
> literature, 0.6–0.8 is typical inter-cardiologist agreement."

### Two ways to run the blind step

**Option A — full blind validation (best, but needs ~15 min of her time).**

- Hand her `clinical_blind.pdf` + pen.
- Ask her to mark each of 16 beats as **NORMAL / ABNORMAL / UNREADABLE**
  without looking at anything else.
- Collect the sheet.
- Afterwards, compute kappa between her labels and (a) the MIT-BIH
  reference, (b) the model. Write one paragraph in the thesis about
  the three-way agreement.

**Option B — qualitative review only (if she's short on time).**

- Skip the blind sheet.
- Just walk her through `examples_by_type_recon.pdf` page by page
  (as in Step 8).
- Note her **verbal reaction** on each disagreement case.
- This isn't kappa-grade validation but is still *expert clinical
  review* — perfectly defensible in the thesis as a **qualitative
  assessment** subsection.

### Is 16 beats enough for kappa?

Strictly, no — statistical confidence on kappa from 16 beats is weak
(wide confidence interval). But for a BSc/MSc thesis it shows **method
and intent** — that's what counts. If she's willing, ~50 beats (use
`examples_by_type_raw.pdf` instead) would give a tighter kappa.

**My recommendation:** offer Option A. If she declines or is short on
time, fall back to Option B. Either way, it's defensible.

---

## Step 12 — Cost-benefit with concrete anchor numbers (7 minutes)

Don't ask her cold for cost figures — she's a clinician, not a health
economist. Bring her **published anchor numbers** (below) and ask
whether they match what she sees. That way she only has to *react*,
not *estimate*.

### Anchor numbers I researched

These are literature-based estimates from peer-reviewed and government
sources. **Hungarian figures may differ** — that's exactly why we
ask her.

**Ischaemic stroke — direct medical cost first 12 months**

| Region | First-year direct cost | Source |
|---|---|---|
| United States | **$15 000 – $35 000** (median ~$20 000) | AHA *Heart Disease & Stroke Statistics* 2023 |
| Western Europe (UK / DE / NL) | **€11 000 – €23 000** | Luengo-Fernandez et al., *Eur. J. Neurol.* 2020 |
| Hungary (likely lower band) | **~€8 000 – €15 000** estimated | Extrapolated from CEE cost-of-illness data; ask her to verify |

**Lifetime cost** (acute + rehab + lost productivity) is **2-4×**
the first-year cost. A US ischaemic stroke is estimated at **~$140 000
lifetime** (AHA).

**AF and stroke**: atrial fibrillation **multiplies stroke risk 5×**,
and AF-related strokes are on average **more severe and more costly
than non-AF strokes** (~30-50 % higher first-year cost; Andersson et
al., *Eur. Heart J.* 2013). One paroxysmal AF episode caught early
enough to start anticoagulation prevents a stroke in roughly **1 in
33 patient-years** (CHADS-VASc 2-3 cohort, NNT data).

**24-hour Holter analysis cost**

| Region | Cost per recording | Cardiologist time |
|---|---|---|
| United States (CMS code 93225-7) | **$170 – $250** | 20-40 min |
| United Kingdom (NHS reference cost) | **£65 – £115** | 25-35 min |
| Hungary (NEAK OENO code 12060/12064 estimate) | **~10 000 – 25 000 HUF** | 30-60 min (ask her) |

**False-positive cardiology referral** — the cost cascade:

- Cardiology consultation: **€50-150** (varies by country/system)
- Echo if ordered: **€100-250**
- Patient's lost workday: **€80-200** (avg gross salary / day)
- Total per false-positive referral: **roughly €230-600** end-to-end

### What to actually say to her

> "I want to give the thesis a cost-benefit chapter, but I'd rather
> use *your* numbers than mine. I read that an ischaemic stroke in
> Hungary costs roughly **€8 000-15 000** in the first year of
> treatment, that a typical 24-h Holter takes a cardiologist
> **30-60 minutes** to read, and that a false-positive referral
> probably ends up costing **€230-600** end-to-end. Do those numbers
> sound right to you, or are they off — and if so, by how much?"

### Three concrete questions

- **Stroke cost** — "Is €8 000-15 000 first-year a reasonable anchor
  for an ischaemic stroke in Hungary, or way off?"
- **Holter time** — "How long does it actually take *you* to read a
  24-h Holter, and how many do you read per week? If a tool reduced
  that by, say, 70 %, would your department procure it?"
- **AF chain** — "If catching one paroxysmal AF episode early prevents
  one stroke per 33 patient-years, does that sound clinically
  realistic to you, and is preventing strokes the *primary* value of
  this kind of tool, or are there other use-cases I'm missing?"

### What you write down

Whatever she says, in her exact words. "According to Dr [name],
practising cardiologist at [hospital], a typical Holter analysis takes
~45 minutes and a missed AF leading to stroke costs roughly XYZ HUF
end-to-end" is **citable** in the thesis Cost-Benefit chapter.

### Sources I'm citing (be transparent if she asks)

- American Heart Association, *Heart Disease and Stroke Statistics —
  2023 Update*
- Luengo-Fernandez R. et al. "Economic burden of stroke across
  Europe" *Eur. J. Neurol.* 2020
- Andersson T. et al. "All-cause mortality in 272 186 patients
  hospitalized with incident atrial fibrillation 1995–2008" *Eur.
  Heart J.* 2013
- Kim H. et al. "Estimating the cost of cardiac care services in
  developing countries" *PLoS One* 2017 (for CEE extrapolation)
- NEAK (Hungarian Health Insurance Fund) OENO codes for ECG/Holter

If she points out any of these are out of date or inaccurate for
Hungary specifically, **note the correction** — that itself is
citable.

---

## Step 13 — Closing: one open question (2 minutes)

End with her honest steer.

**Ask:**
- "If you could **change one design decision** in this project, which
  would it be?"
- "What's the single **biggest risk** you see in a tool like this
  reaching real patients, and what would mitigate it?"

Write the exact words she uses. These two lines often become the
opening of the thesis Discussion chapter.

---

## After the meeting (within 24 hours)

- [ ] Transcribe her answers into a notes file, verbatim.
- [ ] Email her a thank-you with any agreed follow-up (data, contacts).
- [ ] If she labelled the blind PDF, scan the sheet and run the kappa
      calculation the same day while the context is fresh.
- [ ] Draft a 1–2 paragraph Discussion stub quoting her.

---

## One-line reminder to keep in your pocket

> *"She's an expert on hearts. I'm an expert on this code. The meeting
> works when each of us sticks to what we know and asks the other for
> the parts we don't."*

---

# Appendix A — MIT-BIH beat-level annotation symbols

These are the 16 beat symbols that appear in the `.atr` annotation
files of the MIT-BIH Arrhythmia Database. The model treats `N`, `L`,
`R`, `e`, `j` as NORMAL and **everything else** as ANOMALY.

| Symbol | Meaning | Treated as |
|---|---|---|
| `N` | Normal sinus beat | Normal |
| `L` | Left bundle branch block beat | Normal* |
| `R` | Right bundle branch block beat | Normal* |
| `e` | Atrial escape beat | Normal* |
| `j` | Nodal (junctional) escape beat | Normal* |
| `A` | Atrial premature beat (APC / PAC) | Anomaly |
| `a` | Aberrated atrial premature beat | Anomaly |
| `J` | Nodal (junctional) premature beat | Anomaly |
| `S` | Supraventricular premature beat | Anomaly |
| `V` | Premature ventricular contraction (PVC) | Anomaly |
| `F` | Fusion of ventricular and normal beat | Anomaly |
| `!` | Ventricular flutter wave | Anomaly |
| `E` | Ventricular escape beat | Anomaly |
| `/` | Paced beat | Anomaly |
| `f` | Fusion of paced and normal beat | Anomaly |
| `Q` | Unclassifiable beat | Anomaly |

*Note*: bundle-branch blocks and escape beats are sometimes labelled
"normal" and sometimes "anomaly" in the literature. This project
follows the convention of `dataset.py` (NORMAL_SYMBOLS = N, L, R, e, j)
to match the published 0.85 F1 baseline. **This is itself a thesis
discussion point** — the cardiologist may rightly argue an LBBB
should be flagged.

# Appendix B — MIT-BIH rhythm-level annotations

These appear as auxiliary notes in the `.atr` file marking the *start*
of a rhythm episode. They aren't used in the per-beat training but are
clinically important.

| Code | Rhythm |
|---|---|
| `(N` | Normal sinus rhythm |
| `(AFIB` | Atrial fibrillation |
| `(AFL` | Atrial flutter |
| `(AB` | Atrial bigeminy |
| `(B` | Ventricular bigeminy |
| `(T` | Ventricular trigeminy |
| `(VT` | Ventricular tachycardia |
| `(VFL` | Ventricular flutter |
| `(SVTA` | Supraventricular tachyarrhythmia |
| `(PREX` | Pre-excitation (Wolff-Parkinson-White) |
| `(BII` | Second-degree heart block |
| `(IVR` | Idioventricular rhythm |
| `(NOD` | Nodal (AV-junctional) rhythm |
| `(P` | Paced rhythm |
| `(PAF` | Paroxysmal atrial fibrillation |
| `(SBR` | Sinus bradycardia |
| `(SVT` | Supraventricular tachycardia |

# Appendix C — Broader cardiac arrhythmia catalogue (clinical reference)

What a cardiologist actually sees in practice — not all are in
MIT-BIH. Useful so you can recognise terminology she might use.

**Sinus disorders**
- Sinus tachycardia (HR > 100 bpm)
- Sinus bradycardia (HR < 60 bpm)
- Sinus arrhythmia (respiratory variation)
- Sinus arrest / pause
- Sick sinus syndrome (tachy-brady)

**Supraventricular**
- Atrial premature contractions (APC / PAC)
- Atrial tachycardia (focal, multifocal)
- Atrial flutter (typical, atypical)
- Atrial fibrillation (paroxysmal, persistent, permanent)
- AV nodal re-entrant tachycardia (AVNRT)
- AV re-entrant tachycardia (AVRT, e.g. WPW)
- Junctional rhythm / tachycardia

**Ventricular**
- Premature ventricular contractions (PVC, monomorphic, multifocal)
- Bigeminy / trigeminy / quadrigeminy
- Couplets, triplets, salvos
- Non-sustained ventricular tachycardia (NSVT, < 30 s)
- Sustained VT (monomorphic, polymorphic)
- Torsades de pointes
- Ventricular flutter
- Ventricular fibrillation (VF) — life-threatening
- Idioventricular rhythm
- Ventricular escape

**Conduction blocks**
- 1st-degree AV block (long PR)
- 2nd-degree AV block, Mobitz I (Wenckebach) and Mobitz II
- 3rd-degree (complete) AV block
- Left bundle branch block (LBBB)
- Right bundle branch block (RBBB)
- Left anterior fascicular block (LAFB)
- Left posterior fascicular block (LPFB)
- Bifascicular / trifascicular block

**Pre-excitation / channelopathies**
- Wolff-Parkinson-White syndrome (WPW)
- Long QT syndrome (congenital, drug-induced)
- Short QT syndrome
- Brugada syndrome
- Early repolarisation syndrome
- Catecholaminergic polymorphic VT (CPVT)
- Arrhythmogenic right ventricular cardiomyopathy (ARVC)

**Ischaemia / structural — not arrhythmia per se but read on ECG**
- ST-elevation MI (STEMI) — territory-specific patterns
- Non-ST-elevation MI (NSTEMI) — T inversion, ST depression
- Pericarditis (diffuse concave ST elevation)
- Pulmonary embolism (S1Q3T3, RBBB pattern)
- Hyperkalaemia (peaked T waves, wide QRS)
- Hypokalaemia (U waves, prolonged QT)
- Hypothermia (Osborn / J waves)
- Digitalis effect

**What this tool *cannot* see (single-lead, 2-second window)**
- Most ST-segment changes (STEMI / NSTEMI localisation needs 12-lead)
- Subtle conduction-axis changes
- P-wave morphology beyond what lead II shows
- QT-interval pathology (T-wave end is usually outside the 2-s window)
- Anything requiring multiple leads (e.g. left axis deviation
  classification)
