# Project 4 — Image / Text Recognition (OCR)

**Decode Labs — AI Internship, Optional Mastery Phase**
Built by Muhammad Sharjeel Sarwar, November 2026.

## What this is

Project 4 is the optional Mastery Phase of the Decode Labs internship: implement a
basic image or text recognition task using a pre-trained model or library, and
display the output clearly. This project takes **Path 1 — OCR**: read text out of
an image with Google's Tesseract engine (via `pytesseract`), with the PDF's
pre-processing pipeline in front of it.

The test image is my own **Vultr Agent Rush Certificate of Completion** (LabLab,
Nov 3–8, 2026) — a real document, and a deliberately hard one: light text on a
dark blue gradient background.

## Pipeline

1. **Load** the image with OpenCV.
2. **Grayscale** — collapse 3 colour channels into 1 intensity channel. Colour
   carries no information for reading text.
3. **Gaussian Blur (5×5)** — smooth out tiny specks and noise so the next step
   doesn't mistake a speck for a letter.
4. **Adaptive Threshold (Otsu)** — force every pixel to pure black or white.
   Otsu finds the cutoff automatically. This is the PDF's "forcing the binary
   decision" step.
5. **Invert** (`cv2.bitwise_not`) — the certificate has light text on a dark
   background; Tesseract expects dark text on light background, so flip it.
6. **OCR** with Tesseract, `--psm 6` (one uniform block of text).
7. **Confidence check** — Tesseract gives a confidence score per word. The PDF's
   rule: **80% is the absolute minimum**. Words below 80% are dropped.
8. **Visual confirmation** — a green box is drawn around every detected word
   (80%+ only), with its confidence written on top.

## Results (real run, Google Colab, 14 Nov 2026)

- OpenCV 4.14.0, Tesseract 5.3.4
- **Mean confidence: 89.1%**
- **Words above 80%: 30 of 33**
- **Verdict: PASS** — above the 80% minimum

Text read from the certificate:

```
Vultr: Agent Rush Hackathon
AWARDED TO
Sharjeel Sarwar
. FOR OUTSTANDING PERFORMANCE, ATTENDANCE,
SUCCESSFULLY COMPLETING & SUBMITTING A
SOLUTION BASED ON VULTR, ANTHROPIC CLAUDE, F
ANTIGRAVITY IN THE VULTR: AGENT RUSH
HACKATHON i
```

## Validation (the PDF's 4 checks)

| Check | Result |
|---|---|
| Library Integration | ✅ pytesseract + OpenCV installed and running, no errors |
| Pre-Processing Integrity | ✅ Grayscale + Gaussian blur + Otsu thresholding executed and shown side-by-side |
| Accuracy Benchmarking | ✅ Mean confidence 89.1% (≥ 80% minimum), 30/33 words above 80% |
| Visual Confirmation | ✅ Green boxes + confidence labels drawn on the detected words |

## Honest limitations

- The big **"CERTIFICATE OF COMPLETION"** header was not detected — PSM 6
  assumes one uniform block of text and sometimes skips a large title. Trying
  `--psm 3` (fully automatic) is the suggested fix.
- The small glitches in the output (a stray `.`, `F` where ANTIGRAVITY split
  across lines, `HACKATHON i`) are the 3 words that fell below 80% — all from
  the small body text and the gradient background.
- The bottom of the certificate (date / signature area) is lost in the
  thresholded image: one global Otsu threshold can't separate text from a
  gradient background everywhere. A normal page (dark text on white paper) is
  the easy case; a dark certificate with a gradient is the hard case.

## How to run

Open `Decodelabs_project_4.ipynb` in Google Colab and run the cells top to
bottom. The notebook installs Tesseract itself (cell 1), then asks you to
upload an image with printed text. Replace the certificate with any image of
your own to try it on something else.

## Files

- `Decodelabs_project_4.ipynb` — the full notebook, with outputs (including the
  side-by-side pre-processing figure and the boxed detection figure) embedded.
