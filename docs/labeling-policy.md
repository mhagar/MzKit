# Peak labeling policy

Labels train the learned segmenter (`core/labeling/peak_model/`) that decides
which signals seed ensembles and which get excluded from recruitment. Consistency
matters more than any single call, so when unsure, follow these rules.

## Classes

| Mark | Key | Meaning | Engine effect |
|---|---|---|---|
| **peak** | Q | A chromatographic peak you'd want an ensemble built around | can seed; its span is the seed's boundaries |
| **plateau** | T | Persistent, smeared, streaking or whine-like background | never seeds, never recruited |
| *(unmarked)* | — | Baseline and minor blips (e.g. 3–4 point triangles) | recruitable, never seeds |

## Rules

1. **Space means "I reviewed the whole lane".** Anything unmarked is trained as
   *not a peak*, so a missed peak teaches the model to skip it. Zoom through busy
   lanes (A/D pan, W/S zoom); use Shift+Space to skip a lane you can't judge.
2. **Peak extent:** from where the signal leaves baseline to where it returns,
   or to the valley if it touches another peak.
3. **Touching / coeluting peaks:** one peak span each, sharing the valley as the
   edge (selection edges snap to existing spans).
4. **Bump on a streak:** a peak if it's tall enough, otherwise part of the plateau.
   Your eye is the definition here; the model learns where your line is.
5. **Peak-shaped but much wider than typical (more than ~5× the usual peak width):** plateau.
6. **Small but clearly peak-shaped and seed-worthy:** peak. Tiny blips that
   would only ever ride along as cofeatures: leave unmarked.
7. **Single-scan spikes:** leave unmarked.
