Video transcript demo — speaker attribution pipeline
=====================================================

OPEN THE DEMO
  Double-click  transcript_demo_share.html  — it opens in any browser,
  works offline, nothing to install. Use the tabs at the top to switch
  between the two processed videos:

    Reuters   Epstein accusers package (6:18) — machine attribution,
              flagged "needs review" (several soundbites have no
              on-screen or spoken identification; the review gate holds
              them as Unidentified rather than guessing)
    Insurer   Insurer TV interview with Tuio CEO Juan Garcia (7:26) —
              both speakers fully named; ambiguous dual-name graphic was
              flagged by the system and resolved in human review

  Toggles under the player switch the on-video subtitles and the
  speaker-name strap on/off. Click any transcript line to jump there.

WHAT YOU'RE LOOKING AT
  video/HLS stream -> WhisperX transcription -> pyannote speaker
  diarization -> on-screen name reading (OpenCV detection + Claude
  vision) -> Claude speaker attribution -> captions + review gate.

  Names are only assigned on hard evidence (self-identification, host
  hand-off, shotlist match, on-screen graphic); anything uncertain stays
  "Unidentified" and the video is flagged NEEDS REVIEW for an editor.
  Misattribution is the failure mode the system is built to avoid.

  The page also contains the pipeline explainer, the AWS production
  proposal, and the per-video time/cost table (scroll down).

RAW PIPELINE OUTPUTS (in out/)
  <id>.vtt          WebVTT captions with speaker voice tags
  <id>.json         structured record: speakers, confidence, evidence
  <id>.labeled.txt  human-readable transcript for review
