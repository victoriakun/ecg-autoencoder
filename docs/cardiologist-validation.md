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
> first-pass filter for a clinician, not a replacement for one. I trained
> the model only on normal beats, so it doesn't know what arrhythmias
> *look* like — it just knows what *normal* looks like and flags anything
> that doesn't fit."

### 1.2 The dataset — be specific

- **MIT-BIH Arrhythmia Database** — 48 half-hour recordings, ~110 000
  annotated beats, 360 Hz sampling, two-lead ambulatory Holter. Public
  benchmark used for almost every published ECG anomaly study.
- **Annotations** — every beat is hand-labelled by two cardiologists with
  a symbol (normal `N`, ventricular ectopic `V`, atrial premature `A`,
  paced `/`, fusion `F`, etc.). I treat any non-normal symbol as
  "anomaly" for the binary problem.

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
| **F1 score** | 0.85–0.87 | balance of precision and recall at the chosen threshold |
| **Sensitivity** (recall) | 0.91 | catches 91 % of abnormal beats |
| **Specificity** | 0.96 | correctly clears 96 % of normal beats |
| **Precision** | 0.83 | when it flags an alert, 83 % of the time it's right |
| **Detection latency** | < 1 second per window | sub-second time-to-flag |

State this honestly: "These numbers are on a held-out 15 % test split of
MIT-BIH. I have not yet validated on patients outside that database, on
24-hour Holter recordings end to end, or on signals with realistic noise.
Those are the gaps I'd like your help thinking through."

### 1.5 The 2-second window — explain this slowly

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

Show them the GUI: scroll through, point at the residual line crossing
the dashed threshold, point at the WATCHING (yellow) → ANOMALY (red)
state transition.

### 1.6 What you want from them

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
  about **3 800 false positives per day per patient**. Is that
  workload acceptable, or does it need to be 10× lower?
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
