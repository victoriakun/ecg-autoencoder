# Must-ask checklist for the cardiologist meeting

Single-page cheat sheet. The full question set is in
`cardiologist-validation.md`; this is the minimum you cannot leave the
meeting without answers to.

## Clinical validity (the non-negotiable block)

- [ ] **Which beat types matter clinically** — which non-normal symbols in
      MIT-BIH (V, F, A, L, R, /, f, S, J…) are the ones a screening tool
      *must* catch, and which can it safely ignore? Get a prioritized list.
- [ ] **False-negative vs false-positive trade-off** — is missing one PVC
      worse than flagging ten normal beats, or the opposite? This directly
      sets the threshold.
- [ ] **Per-beat vs per-episode frame** — should the tool output
      "this beat is abnormal" or "this 30-s segment is AF"? If the answer
      is "episode", the current windowing convention needs revisiting.
- [ ] **Single-lead limitation** — name every arrhythmia/pathology that
      a single-lead ECG *cannot* reliably detect. (STEMI localisation,
      some bundle-branch patterns, P-wave morphology in other leads…) —
      so the thesis Limitations section lists them explicitly.

## Performance that's actually acceptable

- [ ] **Acceptable alerts per 24 h Holter** — she's the customer. The
      current numbers imply ~3800 false alerts/day/patient; is that
      usable or does it need to be 10× lower before she'd adopt it?
- [ ] **ROC-AUC 0.97 context** — is that clinically meaningful, or is
      her personal bar 0.99+? What does she compare against mentally?
- [ ] **Detection latency budget** — does sub-second matter in her
      workflow, or is "within a minute" fine for outpatient use?

## Workflow fit

- [ ] **Where this slots in** — emergency, outpatient Holter review,
      ICU continuous monitoring, wearables, primary-care screening?
      One primary use case, not a list.
- [ ] **Alert format she'd actually use** — on-screen flash, end-of-shift
      summary, EMR integration, email PDF? What does the *existing*
      tooling look like?
- [ ] **Who reacts to the alert** — nurse, on-call cardiologist,
      patient themselves on a wearable? The right answer changes the
      design significantly.

## Financial framing (cost-benefit for the thesis)

- [ ] **Cost of one ischaemic stroke** in her health system (acute +
      rehab + lost productivity). A range is fine.
- [ ] **Cost of one missed paroxysmal AF episode** that later causes a
      stroke (the chain you'd attribute to the miss).
- [ ] **Cost of a standard 24-h Holter analysis** — tech time +
      cardiologist read time. Tells you how much an automated triage
      would save.
- [ ] **Cost of one false-positive referral** (a normal beat flagged →
      consult, echo, patient time off work). This is the price tag of
      a noisy model.

## Data & validation gaps to close

- [ ] **Which external dataset** should be the *next* validation target
      — PTB-XL, Chapman-Shaoxing, PhysioNet 2017, or her own clinic's
      anonymised data?
- [ ] **Benchmark algorithm to beat** — even a simple one like
      Pan-Tompkins + rule-based morphology. What's the bar she'd expect
      a new tool to clear?
- [ ] **Five anonymised ECGs** for a visual sanity check before the
      defence — is that something she can share?

## Validation step she should actually do in the room

- [ ] Hand her **`clinical_blind.pdf`** *before* the other PDFs. Let
      her label the 16 beats NORMAL / ABNORMAL / UNREADABLE without
      seeing the model's decision. Collect the sheet. Then compute
      Cohen's kappa between her labels, the MIT-BIH reference, and the
      model — that's the rigorous validation the defence committee
      will look for.

## One "what would you change" catch-all

- [ ] **Single biggest risk** she sees with a system like this, and
      what would mitigate it most. One sentence is gold.
- [ ] If she could **steer one design decision**, what would it be?

---

**After the meeting, within 24 h:** write answers verbatim (her phrasing
goes straight into the Discussion chapter). Group cost numbers into a
Cost-Benefit subsection. Any offered data / contacts → email follow-up
within the week.
