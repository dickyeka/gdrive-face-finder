# DESIGN.md: Google Drive Face Finder

## 1. Identity & Design Read
- **Product:** Local computer-vision photo indexing workbench (`YuNet` 5-point face detector + `InsightFace ArcFace` 512-d embedding matcher) for scanning shared Google Drive sports and event albums.
- **Audience:** Athletes, event participants, and photographers sorting large folders of action shots.
- **Design Read:** Local computer-vision photo indexing workbench for athletes and photographers, in a high-contrast darkroom/lightroom technical studio style, dial `ENERGY 2 / RHYTHM 2 / MOTION 1`.

## 2. Liveliness Dials
- **ENERGY: 2 (Balanced)** — Clear typographic hierarchy and crisp photo cards without decorative gradients or marketing fluff.
- **RHYTHM: 2 (Consistent with structural variation)** — Asymmetric 2-column workbench control panel at the top, followed by distinct Primary and Secondary photo grids and a compact tabular index at the bottom.
- **MOTION: 1 (Hover and active states only)** — No perpetual pulsing dots or looping animations; transitions occur only on hover, focus, or active scan progress.

## 3. Color Palette (2 Core Neutral Tiers + 1 Accent + 1 Semantic State)
- **Dark Theme (Default Darkroom Mode):**
  - Page Canvas (`--bg-page`): `#0b0f19` (reduces glare so full-color sports photos stand out with maximum perceptual contrast).
  - Panel Surface (`--bg-panel`): `#111827` with `#1f2937` (`1px`) border.
  - Primary Text (`--text-primary`): `#f8fafc` (18.1:1 contrast on `#0b0f19`).
  - Secondary Text (`--text-secondary`): `#cbd5e1` (12.6:1 contrast on `#0b0f19`, >= 4.5:1 WCAG AA everywhere).
- **Light Theme (Daylight Studio Mode):**
  - Page Canvas (`--bg-page`): `#f8fafc`.
  - Panel Surface (`--bg-panel`): `#ffffff` with `#cbd5e1` (`1px`) border.
  - Primary Text (`--text-primary`): `#0f172a` (17.4:1 contrast on `#f8fafc`).
  - Secondary Text (`--text-secondary`): `#334155` (10.3:1 contrast on `#f8fafc`, >= 4.5:1 WCAG AA everywhere).
- **Primary Accent (`--accent`):** `#0369a1` (`sky-700`, 5.93:1 contrast with `#ffffff`) — Used for the primary scan CTA, active filter selection, and focus ring.
- **Semantic Verification State (`--match-primary`):** `#15803d` (Light) / `#22c55e` (Dark) — Marks verified primary face matches (`Sim >= 0.42`).

## 4. Typography
- **UI Sans (`Plus Jakarta Sans`):** Used for all headings, form labels, buttons, and table labels in natural sentence case for fast scanning.
- **Technical Mono (`JetBrains Mono`):** Restricted strictly to numeric biometric metrics (`Sim: 0.9144`, `IPD: 28.4px`), file names (`DSC_0966.jpg`), and URLs so numbers align vertically in tables and overlays.

## 5. One-Line Reasons for Major Visual Decisions (R-31)
- **Why this color palette?** High-contrast neutral slate surfaces (`#0b0f19` darkroom / `#f8fafc` daylight) keep color cast off the photographs, while a single sky-blue accent (`#0284c7`) directs the eye to the primary scan action.
- **Why this layout?** A compact 2-column workbench header puts reference selection and Drive folder input side-by-side above the fold so results (Primary grid, Secondary grid, and raw link table) occupy the main viewport without marketing filler sections.
- **Why this typography?** `Plus Jakarta Sans` in sentence case provides legible UI controls, while `JetBrains Mono` is reserved for similarity decimals (`0.9144`), pixel coordinates, and filenames so numeric data aligns cleanly.
- **Why this spacing and radius scale?** Two fixed border-radius tokens (`8px` for controls, `12px` for panels/cards) and `16px`/`24px` section gaps distinguish interactive controls from photo containers.
- **Why use cards for photos and a table for links?** Photo cards with `4:3` aspect ratio show the spatial 5-point facial landmark boxes on the image, while the bottom table enables rapid batch scanning of filenames, scores, and raw download links.
- **Why these icons?** Every icon maps to a concrete hardware or file action (camera lens for webcam capture, upload tray for file selection, external link for Google Drive view, download arrow for raw file download).
