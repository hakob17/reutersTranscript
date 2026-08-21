Reuters video transcript demo — speaker attribution pipeline
=============================================================

OPEN THE DEMO
  Double-click  epstein_778738_share.html  — it opens in any browser,
  works offline, nothing to install. Video, transcript and the project
  brief (pipeline explainer, AWS proposal, per-video cost) are all
  embedded in that single file.

WHAT YOU'RE LOOKING AT
  A 6-minute Reuters package processed by the pipeline:
  video -> WhisperX transcription -> pyannote speaker diarization ->
  on-screen name reading (OpenCV + Claude vision) -> Claude speaker
  attribution -> captions + review gate.

  Speaker names shown were resolved automatically with cited evidence;
  anything uncertain stays "Unidentified" and the package is flagged
  NEEDS REVIEW for an editor - misattribution is the failure mode the
  system is built to avoid.

RAW PIPELINE OUTPUTS (in out/)
  epstein_778738.vtt          WebVTT captions with speaker voice tags
  epstein_778738.json         structured record: speakers, confidence,
                              evidence per name, all segments
  epstein_778738.labeled.txt  human-readable transcript for review
