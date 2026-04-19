# Cardiologist validation meeting — briefing & question set

Use this document end-to-end in your meeting with the cardiologist. The first
half is what *you* explain; the second half is what *they* answer. Bring a
laptop with the GUI demo running on record 208.

---

## Part 1 — What to tell the cardiologist

### 1.1 Who you are and what you built (60 seconds)

> "I'm a thesis student building an unsupervised anomaly-detection system
> for ECG signals. The goal is a screening tool that runs continuously on
> a single-lead ECG stream, flags abnormal beats in under one second, and
> writes every alert to an auditable database. It's intended as a
> first-pass filter for a clinician, not a replacement for one."

**What does "unsupervised" mean?**

> "In classical (supervised) machine learning, every training example
> comes with a hand-applied label — e.g. 'this beat is normal',
> 'this is a PVC'. The model learns to separate classes from those
> labelled pairs. In **unsupervised** learning we use no labels at all:
> we show the model **only one kind of data — normal beats only** — and
> teach it to *reconstruct* those signals. The network therefore learns
> what a *healthy ECG pattern* looks like, but **never sees a single
> abnormal beat during training**. At runtime, anything that doesn't
> fit the learned normal pattern produces a large reconstruction error,
> which we call an anomaly. Upside: no need to collect rare-arrhythmia
> examples for training, and the system reacts to rarities it has
> *never seen before*. Downside: it can't tell you *what type* of
> anomaly it is — only that it *differs from normal*."

### 1.2 The dataset — be specific

**MIT-BIH Arrhythmia Database** — the most widely used public
ECG-anomaly benchmark, recorded at Boston's Beth Israel Hospital and
MIT between 1975 and 1979.

- **47 subjects** (25 male, 22 female, ages 23–89), **48 recordings**
  (two subjects have two recordings each)
- **Half-hour Holter recordings** — ~24 hours total, **~110 000
  hand-annotated beats**
- **Sampling rate: 360 Hz** — meaning 360 voltage samples per second
  are recorded (one sample ≈ 2.78 ms). A 2-second window is therefore
  **720 samples**. 360 Hz is fine enough to resolve the QRS
  morphology (~80–120 ms) and T-wave detail, yet low enough to fit a
  24-hour recording on standard storage.
- **Two leads recorded in parallel** — the first is typically **MLII**
  (modified lead II); the second is V1, V2, V4, or V5. In this
  thesis I **use only the first channel (MLII)**.
  - *Why single-lead?* Because (1) wearables (Apple Watch,
    KardiaMobile, ECG patches) are essentially all single-lead, so
    the clinical deployment target is single-lead; (2) it is simpler
    to engineer and to demonstrate in a real-time pipeline.
  - *Why MLII specifically?* Because lead II runs roughly parallel to
    the heart's main electrical axis, so **QRS amplitude is maximal
    and R-peaks are the most reliably visible** — this is why most
    ambulatory Holters and most published MIT-BIH studies use it.
- **Annotations** — every beat was **hand-labelled independently by
  two cardiologists** (disagreements resolved by consensus) with
  symbols: normal `N`, ventricular ectopic `V`, atrial premature `A`,
  paced `/`, fusion `F`, bundle-branch block `L`/`R`,
  supraventricular `S`, etc. — 17 distinct symbols in total. **For
  the binary problem I treat any non-`N` symbol as "anomaly".**
- **Pre-filtering** — 0.5–40 Hz band-pass (to reject baseline drift
  and high-frequency noise), then z-score normalization (zero mean,
  unit variance).

**Important limitations** inherent to this public benchmark — to be
disclosed explicitly in the thesis Discussion: elderly, predominantly
white American population; wired Holter (not wearable); 1970s-era
recording technology; data from a *single centre*.

### 1.3 The model in plain language (no math)

> "I trained an *autoencoder* — a neural network that learns to compress
> a 2-second ECG snippet down to a 32-number summary and then rebuild it.
> Because I only train it on normal beats, the network gets very good at
> reconstructing normal-looking ECG and very bad at reconstructing
> anything unusual. The difference between the original signal and what
> the model rebuilds is the **reconstruction error** — small for normal,
> large for abnormal."

### 1.4 Performance numbers to put on the table

| Metric | Value | Plain meaning |
|---|---|---|
| **ROC-AUC** | 0.972 | how well it separates normal vs abnormal across all thresholds |
| **F1 score** | 0.85 | harmonic mean of precision and recall at the chosen threshold |
| **Sensitivity (recall)** | 0.91 | catches 91 % of abnormal beats |
| **Specificity** | 0.96 | correctly clears 96 % of normal beats |
| **Precision** | 0.83 | when it flags an alert, 83 % of the time it's right |
| **Threshold on reconstruction error** | 0.0434 | F1-optimal value from training |
| **Detection latency** | < 1.5 s | after 2-of-3 window confirmation |

**Detailed explanation — the cardiologist must understand what these
numbers actually count**

All metrics are computed on the test set from four counts over beats
(one 2-second window centred on each annotated beat):

- **TP (true positive)** — abnormal beat, model flagged it as abnormal
- **TN (true negative)** — normal beat, model accepted it as normal
- **FP (false positive)** — normal beat, model wrongly flagged it ("false alarm")
- **FN (false negative)** — abnormal beat, model missed it

Then:

- **Sensitivity = recall = TP ∕ (TP + FN)** — "of the truly abnormal
  beats, what fraction did I catch?". 0.91 means that for every
  100 abnormal beats, **I catch 91 and miss 9**. This is the number
  that matters *for the patient* — low sensitivity means dangerous
  rhythms are missed.
- **Specificity = TN ∕ (TN + FP)** — "of the truly normal beats, what
  fraction did I leave alone?". 0.96 means that for every 100 normal
  beats, **I correctly accept 96 and wrongly alarm on 4**. This is
  the number that matters *for the workflow* — low specificity means
  the cardiologist is drowned in false alarms.
- **Precision (positive predictive value) = TP ∕ (TP + FP)** — "when
  the model fires an alert, how often is it right?". 0.83 means
  **roughly 4 of every 5 alerts are real**, 1 is false.
- **F1 = harmonic mean of precision and recall** —
  `F1 = 2·P·R ∕ (P + R)`. It's the *harmonic* mean because if either
  precision or recall is very low the F1 also drops (you can't "cheat"
  by trading one against the other). The chosen threshold was tuned
  to **maximise F1** on the training data.
- **ROC curve and ROC-AUC** — by sweeping the threshold across its
  full range we plot sensitivity (y-axis) against `1 − specificity`
  (x-axis). The area under that curve (**AUC**) measures the model's
  *threshold-independent* ranking ability: **if I pick one abnormal
  and one normal beat at random, what is the probability that the
  model assigns a higher reconstruction error to the abnormal one?**
  1.00 = perfect separation, 0.50 = random guessing. **0.972** means
  for any random abnormal–normal pair, the abnormal gets the higher
  error ~97 % of the time. This is the most credible overall quality
  number because it doesn't depend on threshold choice.
- **Detection latency** — elapsed time from the onset of a true
  anomaly to the moment the system fires. It's < 1.5 s because the
  2-of-3 smoother confirmation step is the tightest bottleneck.

State this honestly: "These numbers are on a held-out 15 % test split
of MIT-BIH (beat-centred 2-second windows). I have not yet validated
on patients outside that database, on 24-hour Holter recordings end to
end, or on signals with realistic noise. Those are the gaps I'd like
your help thinking through."

### 1.5 The 2-second window and real-time processing

This is the part most clinicians latch onto, so be concrete:

> "The model never looks at a single sample or a single beat in
> isolation. It always processes a **2-second slice** of the ECG (720
> samples at 360 Hz) — typically 2-3 heartbeats. Every **half second** a
> new overlapping 2-second slice is cut and pushed through the model. So
> we get four reconstruction-error scores per second, with each
> consecutive score sharing 1.5 seconds of data with the previous one.
>
> An alert only fires when **two of the last three scores** are above the
> threshold — this prevents single-window noise spikes (electrode
> contact, motion artefact) from triggering a false alarm.
>
> So the worst-case detection latency is about 1.5 seconds: the time to
> see two confirmations after a sustained abnormal pattern starts. The
> system is designed for **continuous monitoring**, not for single-beat
> classification."

**Compatibility with the training pipeline — important detail**

At training time (and when the 0.972 ROC-AUC / 0.85 F1 numbers were
measured), windows were **R-peak-centred** and used **global
normalization** computed over the full signal. Two things differ in the
real-time pipeline, and this should be stated honestly:

1. **Normalization — solved.** The pipeline runs a ~30-second "warmup"
   phase that uses the stream's own statistics to bring the signal
   **into the same scale** as batch global normalization. That puts
   the reconstruction errors into the same numerical range → **I reuse
   the original batch F1-optimal threshold of 0.0434** as the
   operating point. (This replaces an earlier, weaker per-window
   normalization that forced a different 0.0591 threshold.)
2. **Beat-centring — not solvable in real time**, because R-peaks are
   only known *as monitoring happens*. Instead, the window is slid at
   a fixed 0.5 s stride. That sliding means an anomaly can sometimes
   fall near the edge of a window, which is exactly why the
   **2-of-3 smoother** (above) is an indispensable compensation: it
   guarantees that a true abnormal event is captured by at least two
   overlapping windows before an alert fires.

Bottom line: the same model and the same threshold are used in real
time as in the batch evaluation; the *scale* matches, the *beat
centring* does not. The test-split numbers are therefore reported in
the thesis as **batch (ideal-condition) measurements**, with a clear
note that the real-time variant on demo recordings (208 and others) is
expected to produce close — but still *to be validated* — numbers.

Show them the GUI: scroll through, point at the residual line crossing
the dashed threshold, point at the WATCHING (yellow) → ANOMALY (red)
state transition.

### 1.6 The example beats — `clinical_proof.pdf`

Hand them the printed PDF. 16 example beats in four categories:

- **True positive** — model correctly flagged abnormal beat (4 examples)
- **False negative** — model **missed** an abnormal beat (4 examples)
- **False positive** — model **wrongly** alerted on a normal beat (4 examples)
- **True negative** — model correctly cleared a normal beat (4 examples)

> "Each page shows a 2-second window with the original ECG (blue) and
> the model's reconstruction (orange). The shaded red region is the
> reconstruction error. The model flags a beat when that error exceeds
> the threshold (currently **0.0434** — the F1-optimal value from
> training)."

On the 16 beats currently in the PDF the model achieved
**sensitivity 95.9 %** and **specificity 97.6 %**. Be honest that
these are slightly higher than the held-out test split (91 / 96 %)
because some of these records were used during training — the PDF
is for *visual* review, not for measuring performance.

Ask her to specifically pause on the **false positives** and **false
negatives** — those are the most informative pages.

### 1.6b Rigorous validation — `clinical_blind.pdf`

The `clinical_proof.pdf` is *qualitative* review (she sees the
model's decision and reacts). For real expert validation hand her
also **`clinical_blind.pdf`** — the same 16 beats in randomized
order with the model's decision and ground-truth label hidden. She
marks each beat herself as NORMAL / ABNORMAL / UNREADABLE on the
worksheet without knowing what we found.

Then we score her labels against:

| Comparison | What it tells you |
|---|---|
| Cardiologist vs MIT-BIH labels | Is the published reference itself clinically defensible? |
| Cardiologist vs model | Does the model agree with a real expert? |
| Model vs MIT-BIH | The published F1=0.85 we already have |

Compute Cohen's kappa for the first two comparisons. The answer key
is at `docs/clinical_proof/blind_answer_key.csv` — **do NOT show it
to her before she labels.** This step is what turns the meeting from
"expert opinion" into "expert validation" — it's the element your
defence committee will look for if they're rigorous.

### 1.7 What you want from them

> "I have three things I'd like your input on: (1) clinical validity —
> are the arrhythmias I'm catching the *clinically important* ones?
> (2) operational fit — would a system like this actually be useful in
> the workflow you know, and where would it slot in? (3) the value
> argument — what does a missed arrhythmia or a false alarm actually
> cost, in money or in patient outcome, so I can frame the cost-benefit
> for the thesis?"

---

## Part 2 — Questions to ask, grouped by topic

Take notes verbatim — the cardiologist's exact phrasing is gold for
your thesis Discussion chapter.

### A. Clinical validity (which anomalies actually matter)

- A1. The MIT-BIH labels lump 17 different non-normal beat types into one
  "anomaly" class. Of those, **which categories are most clinically
  important to detect — and which can be safely ignored** in a screening
  tool? (e.g. PVC vs PAC vs atrial fibrillation vs ventricular tachycardia)
- A2. Are there any abnormalities **a single-lead ECG cannot reliably
  detect** that I should explicitly disclaim in the thesis? (e.g. STEMI,
  bundle branch blocks, P-wave morphology in lead II only)
- A3. From a clinical safety perspective, which is worse: **missing one
  abnormal beat** in a long stream, or **flagging ten normal beats** as
  abnormal? Why? (this drives how I tune the threshold)
- A4. Is **per-beat detection** even the right frame? Would you instead
  want **per-rhythm-episode** detection ("AF for 30 seconds" rather than
  "this beat is abnormal")?
- A5. How do you, in practice, deal with **noisy or artefacted ECG**?
  Should the system silently drop unreadable windows or alert that signal
  quality is degraded?

### B. Performance trade-offs

- B1. ROC-AUC of 0.972 — is that **clinically meaningful**, or do you
  expect 0.99+ before you'd trust a tool? What benchmark are you used to
  comparing against?
- B2. With sensitivity 0.91 and specificity 0.96, in a 24-hour Holter
  with ~100 000 beats and ~5 % anomaly rate, the system would produce
  about **3 800 false positives per day per patient** (95 000 normal
  × 0.04 false-positive rate). Is that workload acceptable, or does
  it need to be 10× lower?
- B3. What **false-alarm rate** would make you personally stop trusting
  the system within a week of using it?
- B4. Sub-second detection latency — does it actually matter clinically,
  or is "within the same minute" good enough for most outpatient use
  cases?

### C. Workflow integration

- C1. Where does a tool like this **fit in your existing workflow** —
  emergency department triage, outpatient Holter review, ICU continuous
  monitoring, primary-care screening, wearable consumer device?
- C2. What **format of alert** is most useful to you — instant on-screen
  flash, end-of-shift summary, emailed PDF, integration into the
  hospital EMR? (For my thesis I have on-screen + SQLite log; what
  would you add?)
- C3. Who **acts on the alert** — the bedside nurse, the on-call
  cardiologist, the patient themselves on a wearable? The right answer
  depends on this.
- C4. If the system flags a beat as abnormal, what **information do you
  need alongside the alert** to decide what to do — the raw waveform,
  the time of day, beat-to-beat history, prior baseline, the model's
  reconstruction overlay?
- C5. How long would you tolerate a **system being down** before you'd
  switch back to manual monitoring?

### D. Financial / cost-of-care framing

These are the questions that turn a CS thesis into something a
hospital procurement officer will read. Even rough numbers are gold —
push for a *range* if they don't have an exact figure.

- D1. **What does an ischaemic stroke cost** in your healthcare system,
  end-to-end (acute admission + rehabilitation + lost productivity)?
  Even a rough range. Atrial fibrillation is the leading preventable
  cause, so any tool that catches it earlier is buying stroke
  prevention.
- D2. Estimated cost of **one missed paroxysmal AF episode** in a
  patient who later strokes — what's the financial chain you'd
  attribute to it?
- D3. What does **a false-positive cardiology referral** cost — the
  consultation, the follow-up echo or 24-h Holter, the patient's lost
  workday? This is the cost a noisy model imposes.
- D4. How much does a **standard 24-hour Holter analysis** cost (tech
  time + cardiologist read time), and roughly how long does the
  cardiologist spend reading one? An automated triage tool would
  reduce this — by how much would it have to reduce it to be worth
  procuring?
- D5. **Wearable-device market context:** what fraction of your
  patients already wear something (Apple Watch, KardiaMobile, etc.)?
  Are the alerts those devices generate useful or noise to you?
- D6. If a system like mine could **screen the false-positive
  Apple-Watch alerts** before they reach a cardiologist's desk, what
  would that be worth per cardiologist per month?
- D7. **Reimbursement codes** — do you know the billing code in your
  country for AI-assisted ECG interpretation? (NHS, EU, US Medicare
  all differ; even one example helps frame the deployment story)

### E. Regulatory and ethical

- E1. To deploy this in a clinical setting in the EU, I'd need **CE-MDR
  Class IIa** approval (software as medical device). What documents do
  you typically see hospital procurement requesting before they'll
  pilot a tool like this?
- E2. **GDPR / data-protection** — for the thesis I log alerts in a
  local SQLite database with file-level encryption on rotated logs and
  no PHI. Is that defensible as a starting point, or are there obvious
  gaps a hospital data-protection officer would flag immediately?
- E3. Who is **liable** if the system misses a beat that turns out to
  be the start of a fatal arrhythmia — the hospital, the device
  vendor, the supervising cardiologist? Even a one-sentence opinion
  is useful.
- E4. **Bias** — MIT-BIH is mostly older American patients from the
  1970s-80s. What demographics would I most need to retrain on before
  this could safely run on, say, a paediatric or pregnant population?
- E5. Should the model **explain its decisions** (saliency map, beat
  morphology comparison) before a clinician would accept it? Or is a
  black-box "anomaly score + waveform overlay" enough?

### F. Data and validation gaps

- F1. **Beyond MIT-BIH** — what dataset would you point me to as the
  next step for external validation? (PTB-XL, Chapman-Shaoxing,
  PhysioNet 2017, your own anonymised clinic data?)
- F2. Is there a **clinically validated alternative algorithm** I
  should benchmark against — even a simple one like Pan-Tompkins +
  rule-based morphology? What's the bar I should clear?
- F3. The training set treats every beat-type the same. Would it be
  more clinically useful to **build a multi-class** detector (PVC vs
  AF vs PAC vs other) or stick with binary (normal vs abnormal)?
- F4. Do you have access to **even 5 anonymised ECGs** from real
  patients I could test against, even just visually, as a sanity check
  before the thesis defence?

### G. Open-ended

- G1. **What does a tool like this need to look like** for you, as a
  practising cardiologist, to want to use it in your week?
- G2. What's the **single biggest risk** you see with a system like
  this — and what would mitigate it most?
- G3. If you had to **steer my thesis** in one direction, what's the
  single change you'd make to the project?

---

## Part 3 — After the meeting

- Write up the answers within 24 hours while phrasing is fresh.
- Group quotes thematically — they go straight into the thesis
  Discussion chapter as "Clinical perspective" subsections.
- Any cost numbers go into a Cost-Benefit Analysis subsection — even
  a rough range is more credible than no number at all.
- If they offered to share data or contacts, follow up by email
  within the week.
