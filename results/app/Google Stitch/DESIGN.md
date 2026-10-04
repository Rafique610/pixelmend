---
name: PixelMend Neural Research Console
colors:
  surface: '#0b1326'
  surface-dim: '#0b1326'
  surface-bright: '#31394d'
  surface-container-lowest: '#060e20'
  surface-container-low: '#131b2e'
  surface-container: '#171f33'
  surface-container-high: '#222a3d'
  surface-container-highest: '#2d3449'
  on-surface: '#dae2fd'
  on-surface-variant: '#cbc3d7'
  inverse-surface: '#dae2fd'
  inverse-on-surface: '#283044'
  outline: '#958ea0'
  outline-variant: '#494454'
  surface-tint: '#d0bcff'
  primary: '#d0bcff'
  on-primary: '#3c0091'
  primary-container: '#a078ff'
  on-primary-container: '#340080'
  inverse-primary: '#6d3bd7'
  secondary: '#4cd7f6'
  on-secondary: '#003640'
  secondary-container: '#03b5d3'
  on-secondary-container: '#00424e'
  tertiary: '#4edea3'
  on-tertiary: '#003824'
  tertiary-container: '#00a572'
  on-tertiary-container: '#00311f'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#e9ddff'
  primary-fixed-dim: '#d0bcff'
  on-primary-fixed: '#23005c'
  on-primary-fixed-variant: '#5516be'
  secondary-fixed: '#acedff'
  secondary-fixed-dim: '#4cd7f6'
  on-secondary-fixed: '#001f26'
  on-secondary-fixed-variant: '#004e5c'
  tertiary-fixed: '#6ffbbe'
  tertiary-fixed-dim: '#4edea3'
  on-tertiary-fixed: '#002113'
  on-tertiary-fixed-variant: '#005236'
  background: '#0b1326'
  on-background: '#dae2fd'
  surface-variant: '#2d3449'
typography:
  display-lg:
    fontFamily: Space Grotesk
    fontSize: 36px
    fontWeight: '700'
    lineHeight: 44px
    letterSpacing: -0.03em
  headline-lg:
    fontFamily: Space Grotesk
    fontSize: 28px
    fontWeight: '600'
    lineHeight: 36px
    letterSpacing: -0.02em
  headline-lg-mobile:
    fontFamily: Space Grotesk
    fontSize: 22px
    fontWeight: '600'
    lineHeight: 30px
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Space Grotesk
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 24px
    letterSpacing: -0.01em
  body-lg:
    fontFamily: Geist
    fontSize: 15px
    fontWeight: '400'
    lineHeight: 22px
  body-md:
    fontFamily: Geist
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
  body-sm:
    fontFamily: Geist
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
  metric-display:
    fontFamily: JetBrains Mono
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: -0.02em
  metric-value:
    fontFamily: JetBrains Mono
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 18px
  label-mono:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 14px
    letterSpacing: 0.04em
  label-ui:
    fontFamily: Geist
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.01em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter: 0.75rem
  margin: 1.25rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.75rem
  space-lg: 1.25rem
  space-xl: 2rem
---

## Brand & Style

This design system targets machine learning researchers, computer vision engineers, and imaging scientists operating at the edge of neural image restoration (super-resolution, artifact removal, denoising, inpainting). The brand personality is rigorous, computationally potent, surgically precise, and unapologetically technical. It avoids consumer-grade simplification in favor of information density, signal integrity, and low cognitive latency.

The visual style unites **Technical Minimalism** with **Hyper-Functional Glassmorphism**. Dark charcoal and slate foundations prevent visual exhaustion during extended monitoring runs, while focused chromatic bursts—electric violet for generative processes, neon cyan for tensor throughput, and emerald for validation health—illuminate critical thresholds. The interface balances high-density data instrumentation with fine-milled glass surface overlays, luminous hairline strokes, and glowing telemetry indicators.

## Colors

The palette is engineered specifically for OLED/HDR displays and low-light laboratory workflows:

- **Primary (`#8B5CF6` Electric Violet):** Reserved for generative inference actions, model compilation triggers, latent space explorations, and primary state callouts.
- **Secondary (`#06B6D4` Neon Cyan):** Dedicated to dynamic data streams, tensor bandwidth telemetry, loss curve plotting, and active pipeline routing.
- **Tertiary (`#10B981` Emerald Green):** Signals model convergence, validation breakthroughs, deterministic parity, and healthy node clusters.
- **Accent Warning (`#F59E0B` Amber):** Applied to gradient explosions, GPU thermal throttling, and epoch stagnation.
- **Accent Critical (`#EF4444` Crimson):** Marks CUDA out-of-memory panics, gradient divergence, and tensor dimension mismatches.
- **Neutral Foundations:**
  - Base Canvas: `#090D16` (Deep Obsidian)
  - Surface Tier 1 (Containers/Panels): `#0F172A` (Midnight Slate)
  - Surface Tier 2 (Elevated Nodes): `#1E293B` (Charcoal Slate)
  - Surface Overlay (Glassmorphism): `rgba(15, 23, 42, 0.72)` with hairline `rgba(148, 163, 184, 0.12)`
  - Text High-Emphasis: `#F8FAFC`
  - Text Mid-Emphasis: `#94A3B8`
  - Text Muted/Subtle: `#475569`

## Typography

The typographic hierarchy enforces a disciplined separation between human-readable administrative UI and raw computational output:

- **Display & Section Titles (`Space Grotesk`):** Provides sharp, geometric, and scientific character to screen headers, modal titles, and high-level architectural viewports.
- **Application Body (`Geist`):** Delivers neutral, hyper-legible structural navigation, form inputs, tooltips, and contextual notes without competing with dense metric arrays.
- **Instrumentation & Telemetry (`JetBrains Mono`):** Every numerical readout, epoch counter, tensor shape `[B, C, H, W]`, learning rate decimal, PSNR/SSIM calculation, and code trace utilizes monospaced styling with tabular numerals enabled (`font-variant-numeric: tabular-nums`). This guarantees absolute column stability during rapid real-time telemetry streaming.

## Layout & Spacing

This design system uses a high-density, 12-column fluid grid system engineered for multi-monitor workstation setups (1440px to 4K displays) down to diagnostic laptop screens.

- **Desktop (1280px+):** 12 columns with compact `0.75rem` (12px) gutters and `1.25rem` (20px) outer margins. Side rails (epoch queues, hyperparameter controls) are docked in fixed-width 280px–360px panes, while the central canvas hosts variable multi-column split views for input/output image restoration comparisons and loss curves.
- **Tablet / Responsive Breakpoint (768px - 1279px):** 8 columns with 12px gutters; telemetry drawers convert into stacked collapsibles.
- **Mobile (Diagnostic Fallback, < 768px):** 4 columns with `0.5rem` gutters and `0.75rem` margins. Side panels collapse into off-canvas sliding overlays.
- **Rhythm Philosophy:** Dense, compact vertical spacing (`space-xs` through `space-md`) ensures maximum metric visibility above the fold, eliminating gratuitous scrolling for critical runtime controls.

## Elevation & Depth

Visual hierarchy does not rely on heavy drop shadows, which muddy dark interfaces. Instead, it utilizes **Tonal Stratification, Luminous Hairlines, and Glassmorphism**:

1. **Base Floor (L0):** Solid deep charcoal `#090D16` canvas with subtle faint grid lines (`rgba(148, 163, 184, 0.03)`).
2. **Structural Panels (L1):** Slate `#0F172A` containers bounded by `1px` crisp borders tinted `rgba(148, 163, 184, 0.08)`. No blur, high rendering performance.
3. **Floating Glass Overlays & Hover Cards (L2):** Translucent backdrop `rgba(15, 23, 42, 0.75)` backed by `backdrop-filter: blur(16px)` and an inner highlight ring `inset 0 1px 0 0 rgba(255, 255, 255, 0.08)`.
4. **Active & Diagnostic Nodes (L3):** Targeted perimeter glows using the accent spectrum:
   - Primary Focus: `0 0 16px -2px rgba(139, 92, 246, 0.35)`
   - Neon Telemetry Pulse: `0 0 14px -2px rgba(6, 182, 212, 0.35)`
   - Healthy Convergence: `0 0 12px -2px rgba(16, 185, 129, 0.35)`
5. **Modals & Tensor Inspector Dialogs (L4):** Heavy blur `backdrop-filter: blur(24px)` over an ambient drop shadow: `0 20px 40px -10px rgba(0, 0, 0, 0.7)`.

## Shapes

The design system enforces a disciplined, mechanical aesthetic with **Soft (`roundedness: 1`)** geometry. Radii are intentionally tight to preserve data grid alignment and communicate laboratory instrumentation:

- **Base Components (Inputs, Metric Badges, Buttons):** `4px` (`0.25rem`) corner radius.
- **Panels, Chart Canvases, Image Viewports:** `8px` (`0.5rem`, `rounded-lg`) corner radius.
- **Floating Modals and HUD Drawers:** `12px` (`0.75rem`, `rounded-xl`) corner radius.
- **Status Indicators & Health Pills:** Full capsule pill styling (`9999px`) reserved specifically for live operational indicators (e.g., `CUDA: ONLINE`, `FP16 EVAL`), creating an immediate functional contrast against rectangular data matrices.

## Components

### Buttons & Trigger Controls
- **Primary Inference Action:** High-intensity electric violet background (`#8B5CF6`), high-contrast white text, `4px` radius, subtle top-edge bevel highlight. Hover triggers a radiant violet bloom (`0 0 16px rgba(139, 92, 246, 0.45)`).
- **Secondary Ghost Action:** Transparent fill, `1px` border `rgba(148, 163, 184, 0.2)`, `JetBrains Mono` or `Geist` text. Active hover fills with `rgba(255, 255, 255, 0.04)`.
- **Destructive (Abort Run):** Crimson outline with an electric red hover aura.

### Status Pills & Telemetry Chips
- Capsule-shaped pills with an ultra-compact footprint (`padding: 2px 8px`).
- Built with a glowing core: a `6px` circular LED indicator displaying an ambient pulse animation (` tertiary` emerald for synced runs, `secondary` cyan for processing batches), paired with uppercase monospaced labels (`label-mono`).

### Data Panels & Metric Cards
- Grounded in `#0F172A` with `1px` solid borders (`rgba(148, 163, 184, 0.08)`).
- Card headers feature an inline badge showing tensor dimensions or metric units (e.g., `dB`, `SSIM`, `ms/it`).
- Primary metrics are rendered in 24px `JetBrains Mono` with trailing change indicators (`+0.042 PSNR`).

### Glassmorphism Overlays (Inference HUD)
- Positioned directly over split-screen before/after image restoration viewports.
- Background: `rgba(15, 23, 42, 0.65)` with `12px` backdrop blur, crisp `1px` border (`rgba(255, 255, 255, 0.08)`).
- Houses live zoom coordinates, pixel delta heatmaps, and floating inference latency counters without obstructing visual analysis.

### Interactive Image Splitter (Before / After Comparison)
- Dual canvas viewport with a 1px vertical neon cyan divider line terminating in a diamond-shaped slider thumb.
- Draggable with zero-latency response; overlays display monospaced metadata pills ("DEGRADED: BPF 1.2" vs "RESTORED: LATENT DIFFUSION").

### Input Fields & Parameter Sliders
- Inputs feature dark slate backgrounds (`#070A11`), hairline borders, and `JetBrains Mono` text for float values (e.g., `0.0001` learning rate).
- Sliders use a neon cyan track with a glowing violet thumb, displaying numeric tooltips during drag adjustments.

### Checkboxes & Segmented Toggles
- Custom square `14px` check containers with a `2px` radius. Active state produces a solid violet fill with an SVG check mark.
- Multi-state toggles (e.g., `Bicubic | ESRGAN | LatentMend`) share a recessed `#070A11` background, using a sliding elevated `#1E293B` segment with glowing border cues.