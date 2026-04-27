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

## Step 12 — Cost-benefit (5 minutes, she's the oracle here)

She's likely not a health economist, but she has seen costs first-hand.

**Ask softly, not as a test:**
- "Rough order of magnitude: what does an ischaemic stroke cost the
  health system, acute + rehab combined?"
- "What does a standard 24-h Holter analysis cost at your institution?
  How much of it is cardiologist read time?"
- "A false-positive cardiology referral — the consult, the follow-up
  echo, the patient's lost workday — what would that add up to
  roughly?"

Any number, even a rough range, goes into the thesis Cost-Benefit
subsection. "According to a practising cardiologist at [clinic]…" is
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
