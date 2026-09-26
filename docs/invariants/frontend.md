# Frontend Invariants (React + Tailwind CSS)

## Stack

- **React** (Vite-based) with TypeScript.
- **Tailwind CSS** for all styling — no CSS modules, no styled-components.
- **No additional UI framework** unless explicitly approved.

## Folder Structure

```
src/app/frontend/
  src/
    pages/
      UniversalRestoration.tsx    # Task 1 workspace
      HardRoutedRestoration.tsx   # Task 2 workspace
      SoftMoERestoration.tsx      # Task 3 workspace
      FaceToSketch.tsx            # Task 4 workspace
    components/
      layout/                    # Navbar, sidebar, workspace shell
      shared/                    # ImageUploader, ImagePanel, ResultCard
      task1/                     # Task-specific components
      task2/
      task3/
      task4/
    hooks/                       # Custom hooks (useInference, useImageUpload)
    services/
      api.ts                     # Axios/fetch wrapper for backend calls
    types/                       # TypeScript interfaces
    App.tsx
    main.tsx
  public/
  index.html
  tailwind.config.js
  vite.config.ts
  package.json
```

## UI Requirements (from Assignment)

- **Single application** with four workspaces (tabs or sidebar navigation).
- Workspace names match the assignment exactly:
  1. Universal Restoration
  2. Hard-Routed Restoration
  3. Soft Mixture-of-Experts Restoration
  4. Face-to-Sketch Generator
- **Google Stitch** design required — design the UI in Stitch first,
  include screenshots in the report, then implement in React.
- Each workspace must show: image upload, result display, inference time,
  and task-specific metadata.

## Component Conventions

- One component per file, max ~150 lines.
- Props via TypeScript interfaces, not `any`.
- API calls in `services/api.ts`, not inside components.
- Loading states and error states for every async operation.

## Responsive Layout

- Must work on desktop (evaluator's browser). Mobile-responsive is optional
  but appreciated.
