---
name: SignalScope
description: Media forensics for AI-generated imagery with faithful, localized explanations
colors:
  primary: "#f59e0b"
  primary-bright: "#fbbf24"
  ai-crimson: "#f43f5e"
  real-emerald: "#10b981"
  unsure-topaz: "#eab308"
  bench-obsidian: "#07090e"
  panel-titanium: "#101522"
  panel-elevated: "#161e30"
  ink-white: "#f8fafc"
  ink-silver: "#cbd5e1"
  ink-slate: "#64748b"
typography:
  display:
    fontFamily: "'Instrument Serif', Georgia, serif"
    fontSize: "36px"
    fontWeight: 400
    lineHeight: 1.1
  body:
    fontFamily: "'Plus Jakarta Sans', system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "'IBM Plex Mono', monospace"
    fontSize: "10.5px"
    fontWeight: 500
    letterSpacing: "0.18em"
rounded:
  sm: "6px"
  md: "10px"
  lg: "14px"
  xl: "20px"
spacing:
  sm: "8px"
  md: "14px"
  lg: "24px"
  xl: "36px"
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "#0c0904"
    rounded: "{rounded.md}"
    padding: "8px 15px"
---

# Design System: SignalScope

## Overview

**Creative North Star: "The Cyber-Forensic Intelligence Terminal"**

SignalScope is engineered to feel like an authoritative, high-tier intelligence workstation: deep obsidian void surfaces, hairline titanium enclosures, precision amber instrument accents, and luminescent forensic readouts.

Rather than generic dark gray blocks or washed-out beige panels, the interface balances maximum contrast, spatial rhythm, and haptic depth. It evokes the atmosphere of an advanced spectrometry cleanroom where every visual artifact and confidence score is presented with scientific rigor.

**Key Characteristics:**
- **Obsidian & Titanium Palette:** Deep void tones (`#07090e`) with subtle chromatic ambient lighting.
- **Double-Bezel Architecture:** Layered enclosures with hairline rim highlights (`rgba(255,255,255,0.08)`) and inset top bevels.
- **Precision Signal Language:** Electric amber (`#f59e0b`) as the primary instrument color; bioluminescent emerald (`#10b981`) for confirmed authenticity; vibrant crimson (`#f43f5e`) for synthetic anomalies.
- **Typography Trifecta:** `Instrument Serif` for editorial gravity, `Plus Jakarta Sans` for clean interface clarity, and `IBM Plex Mono` for tabular forensics.

## Colors

The palette is anchored in an obsidian void with calibrated semantic status tokens that glow with subtle luminescence.

### Primary
- **Electric Amber** (`#f59e0b` / `#fbbf24`): The primary instrument and focal accent. Used for primary CTAs, active navigation beacons, reticle brackets, and focal telemetry needles.

### Secondary
- **Bioluminescent Emerald** (`#10b981` / `#34d399`): Indicates authentic camera capture, high confidence validation, and live service health.
- **Crimson Hazard** (`#f43f5e` / `#fb7185`): Flags synthetic signatures, adversarial anomalies, and critical failure states.
- **Solar Topaz** (`#eab308`): Denotes uncertain threshold ranges and cautionary observations.

### Neutral
- **Deep Obsidian Void** (`#07090e`): Canvas background with ambient radial orbs and high-precision micro-grid.
- **Titanium Panel** (`#101522` / `#161e30`): Elevated module surfaces with subtle glassmorphic backdrop filters.
- **Titanium White Ink** (`#f8fafc`): Crisp display and headline typography.
- **Silver Slate** (`#cbd5e1`): Secondary labels and body explanations.
- **Muted Slate** (`#64748b`): Technical metadata and reference boundaries.

## Typography

**Display Font:** `Instrument Serif` (fallback: Georgia, serif)
**Body Font:** `Plus Jakarta Sans` (fallback: `IBM Plex Sans`, system-ui)
**Mono Font:** `IBM Plex Mono` (fallback: Consolas, monospace)

### Hierarchy
- **Display** (400, 36px, line-height 1.1): Page titles and primary verdicts.
- **Headline** (600, 13.5px, uppercase/tracking 0.02em): Module headers and card titles.
- **Body** (400, 14px, line-height 1.55, max 64ch): Plain-language explanations and instructions.
- **Label / Eyebrow** (500, 10.5px, tracking 0.18em, uppercase): Technical categories and module markers.

## Elevation & Depth

Surfaces rely on dark glassmorphism, inner hairline highlights, and diffuse ambient shadows rather than stark drop-shadows.

### Shadow Vocabulary
- **Panel Base** (`0 0 0 1px rgba(255,255,255,0.07), 0 4px 16px -2px rgba(0,0,0,0.5), 0 12px 32px -4px rgba(0,0,0,0.6)`): Standard structural cards and data panels.
- **Elevated Dossier** (`0 0 0 1px rgba(255,255,255,0.05), 0 16px 40px -10px rgba(0,0,0,0.7), inset 0 1px 0 rgba(255,255,255,0.12)`): Archival verdict plaque.

## Components

### Buttons
- **Primary:** Gradient amber fill (`#fbbf24` to `#f59e0b`), dark titanium text (`#0c0904`), crisp inset top highlight, tactile press feedback (`scale(0.98)`).
- **Secondary / Ghost:** Titanium glass background, hairline border, gold hover glow with subtle `translateY(-1px)`.

### Cards & Panels
- **Doppelrand Shell:** Nested enclosures with outer hairline rim and inner recessed content.
- **Dropzone:** Tactical corner reticles with hover pulse and ambient radial spotlight.

### Telemetry Gauge
- **Dial:** 234° sweep arc with luminous drop shadow glow, threshold marker, and crisp numeric readout.

## Do's and Don'ts

### Do:
- **Do** preserve the dark obsidian void as the global background across all views.
- **Do** keep tabular data in `IBM Plex Mono` with `font-variant-numeric: tabular-nums`.
- **Do** use semantic glows (`var(--amber-dim)`, `var(--ai-dim)`, `var(--real-dim)`) sparingly to highlight verified status.

### Don't:
- **Don't** use pure `#000000` or muddy yellowish-gray darks.
- **Don't** introduce generic un-tinted drop shadows.
- **Don't** make sudden light-mode card insertions in the middle of a dark view.
