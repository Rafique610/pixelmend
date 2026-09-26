# Application Implementation Plan: React, FastAPI, and Docker Compose

This document defines the 13-step technical implementation plan for the application layer of the Generative AI assignment. The application integrates four distinct deep learning tasks into a single coherent web system: Universal Restoration (Task 1), Hard-Routed Restoration (Task 2), Soft Mixture-of-Experts Restoration (Task 3), and Face-to-Sketch Synthesis (Task 4).

---

## Step 1: Google Stitch UI Design (Research Step)

### Decision & Why It Matters
The layout and navigation paradigm govern how evaluators interact with the four deep learning workspaces. A disjointed or cluttered interface degrades user experience and obscures key ML metrics (routing probabilities, gating weights, inference latency breakdowns, and residual error maps). Furthermore, designing the application in Google Stitch prior to frontend coding is a mandatory assignment constraint; visual design artifacts and evidence of alternative evaluations must be documented for inclusion in the IEEE final report.

### Alternatives to Research

| Option | Architecture Pattern | Pros | Cons |
|---|---|---|---|
| **Option 1: Sidebar Navigation** | Fixed persistent left sidebar with icons and labels for each of the 4 workspaces; right-hand main content canvas. | Standard dashboard pattern; instantaneous workspace switching; ample vertical space for global application health and model status indicators. | Consumes horizontal screen width; requires responsive collapsible drawer behavior on smaller viewports. |
| **Option 2: Top Tab Navigation** | Horizontal tab bar spanning the upper header; full-width content viewport directly beneath. | Familiar, uncluttered mental model; preserves full horizontal width for side-by-side image comparison panels (input vs. restored). | Limited horizontal header space for rich workspace metadata or status badges; less scalable if sub-views expand. |
| **Option 3: Card Landing Hub with Workspace Views** | Central dashboard displaying 4 interactive cards with task summaries and status; clicking a card navigates into a focused workspace view with a back link. | Visually striking first impression; clear separation of task context; high suitability for presentation/demonstration walkthroughs. | Introduces an extra navigation click when frequently switching between restoration models to compare outputs; breaks side-by-side workflow. |

#### Design Evaluation Matrix

| Criterion | Sidebar Navigation (Option 1) | Top Tab Navigation (Option 2) | Card Landing Hub (Option 3) |
|---|---|---|---|
| **Space Efficiency** | Moderate (reserves 240px left) | High (uses ~48px top bar) | High inside workspace, low on hub |
| **Workspace Switching Speed** | Instant (1 click) | Instant (1 click) | Multi-step (2 clicks via hub) |
| **Side-by-Side Image Room** | Good on ≥1280px displays | Excellent across all displays | Excellent inside workspace |
| **Metadata & Status Room** | High (sidebar footer holds badges) | Moderate (header chips only) | High on card, low in workspace |
| **Mobile / Small Screen Ergonomics** | Requires drawer collapse | Scrolls horizontally or wraps | Stacks naturally |
| **Google Stitch Prototyping Speed** | Fast | Very Fast | Moderate |

### Recommended Approach
*(To be selected and documented during implementation following Google Stitch prototyping and screenshot capture.)*

### Research Notes
*(To be populated during implementation with Google Stitch export URLs, layout screenshots, and design tokens.)*

---

## Step 2: FastAPI Backend Skeleton

### Scope
Establish the backend application skeleton using FastAPI, Pydantic Settings, and ONNX Runtime. The backend serves inference requests, validates inputs, manages ONNX model sessions in memory, and exposes health monitoring.

### What to Build
- Application entrypoint in `src/app/backend/main.py` configuring CORS middleware, lifespan event handlers for model loading, and router mounting.
- Configuration module in `src/app/backend/config.py` using `pydantic_settings.BaseSettings` with the `GENAI_` environment prefix.
- Health check router in `src/app/backend/routers/health.py` exposing `GET /health`.
- ONNX Runtime session manager service in `src/app/backend/services/inference.py`.
- Image validation and tensor preprocessing service in `src/app/backend/services/preprocessing.py`.
- Shared response and error schemas in `src/app/backend/schemas/common.py`.

### Key Details
- **Configuration & Invariants**: All settings (model paths, host, port, CORS origins, max upload size) are managed via `Settings` in `src/app/backend/config.py` with `GENAI_` prefix per backend invariants. No direct `os.environ` access.
- **Model Lifespan Management**: All ONNX sessions are instantiated once during FastAPI application startup via `@asynccontextmanager` lifespan, preventing per-request disk I/O overhead.
- **ONNX Execution Providers**: Use `onnxruntime.InferenceSession` configured with `CPUExecutionProvider` by default, falling back gracefully if `CUDAExecutionProvider` is unavailable.
- **Image Preprocessing Service**:
  - Validates image uploads against MIME types (`image/jpeg`, `image/png`) and file size limits (maximum 10MB).
  - Decodes image bytes into RGB format, resizes to $128 \times 128$ using bilinear interpolation.
  - Normalizes pixel values from $[0, 255]$ uint8 to $[0.0, 1.0]$ float32 tensor of shape `[1, 3, 128, 128]` (NCHW).
- **Postprocessing & Base64 Encoding**:
  - Clamps output float32 tensor to $[0.0, 1.0]$, scales to $[0, 255]$ uint8, encodes to PNG, and converts to a base64 Data URL string (`data:image/png;base64,...`).

### Verification
- Run backend with `uv run uvicorn src.app.backend.main:app --port 8000`.
- Verify `GET http://localhost:8000/health` returns HTTP 200 with `{ "status": "ok", "models_loaded": { ... } }`.
- Verify OpenAPI interactive documentation is accessible at `http://localhost:8000/docs`.

### Files Changed
- `src/app/backend/main.py`
- `src/app/backend/config.py`
- `src/app/backend/routers/__init__.py`
- `src/app/backend/routers/health.py`
- `src/app/backend/schemas/__init__.py`
- `src/app/backend/schemas/common.py`
- `src/app/backend/services/__init__.py`
- `src/app/backend/services/inference.py`
- `src/app/backend/services/preprocessing.py`

---

## Step 3: Backend — Universal Restoration Workspace

### Scope
Implement the inference endpoint for Task 1 (Universal Autoencoder), supporting direct upload of corrupted images as well as synthetic on-the-fly server-side corruption for clean images.

### What to Build
- Restoration endpoint `POST /api/v1/restore/universal` in `src/app/backend/routers/task1.py`.
- Request and response schemas in `src/app/backend/schemas/task1.py`.
- Synthetic corruption utility service in `src/app/backend/services/corruption.py`.
- Residual error map generator computing pixel-level absolute differences.

### Key Details
- **Input Handling**: Accepts `multipart/form-data` containing an image file, alongside optional form parameters:
  - `apply_corruption`: boolean flag indicating whether server should corrupt a clean input.
  - `corruption_type`: enum (`salt_and_pepper`, `gaussian_blur`, `rectangular_occlusion`).
  - `severity`: integer or float matching the assignment test benchmark levels:
    - *Salt-and-pepper noise*: $p \in \{0.03, 0.08, 0.15\}$.
    - *Gaussian blur*: $(\text{kernel}, \sigma) \in \{(3, 0.7), (5, 1.5), (7, 2.5)\}$.
    - *Rectangular occlusion*: $1, 2, \text{or } 3\text{ boxes}$ covering $\sim 10\%, \sim 20\%, \sim 35\%$ area.
- **Inference Pipeline**:
  - Image is converted to tensor `[1, 3, 128, 128]`.
  - If `apply_corruption` is true, apply specified synthetic corruption to generate the corrupted input.
  - Execute `task1_universal_ae.onnx` session.
  - Measure inference execution time in milliseconds using `time.perf_counter_ns()`.
- **Error Map Calculation**: Compute absolute difference $|I_{\text{restored}} - I_{\text{corrupted}}|$ (or $|I_{\text{restored}} - I_{\text{clean}}|$ if clean reference exists), apply a colormap (e.g., Jet or Turbo) or normalized grayscale intensity, and encode as base64 PNG.
- **Response Schema (`UniversalRestoreResponse`)**:
  - `original_image`: base64 data URL of the input image.
  - `restored_image`: base64 data URL of restored output.
  - `error_map`: base64 data URL of residual error heatmap.
  - `corruption_applied`: dictionary detailing applied corruption type and severity (null if direct corrupted upload).
  - `inference_time_ms`: float inference duration in milliseconds.

### Verification
- Send a test pet image to `POST /api/v1/restore/universal` via `curl` or OpenAPI docs with `apply_corruption=true`, `corruption_type=salt_and_pepper`, `severity=0.08`.
- Confirm response contains valid base64 images for input, output, and error map, with execution latency < 100ms on CPU.

### Files Changed
- `src/app/backend/routers/task1.py`
- `src/app/backend/schemas/task1.py`
- `src/app/backend/services/corruption.py`
- `src/app/backend/main.py` (mount task1 router)

---

## Step 4: Backend — Hard-Routed Restoration Workspace

### Scope
Implement the sequential two-stage inference endpoint for Task 2: a 4-way convolutional classifier predicts the corruption category and routes the image exclusively to the corresponding specialist autoencoder (or identity bypass).

### What to Build
- Hard-routed endpoint `POST /api/v1/restore/hard-routed` in `src/app/backend/routers/task2.py`.
- Request and response schemas in `src/app/backend/schemas/task2.py`.
- Routing logic orchestrating classifier session and specialist autoencoder sessions in `src/app/backend/services/inference.py`.

### Key Details
- **Pipeline Architecture**:
  1. Input image is preprocessed to `[1, 3, 128, 128]`.
  2. **Stage 1 (Classification)**: Image passes through `task2_classifier.onnx`. Outputs 4 class logits corresponding to `[clean, salt_and_pepper, gaussian_blur, rectangular_occlusion]`.
  3. Softmax is applied to extract probabilities $P = [p_{\text{clean}}, p_{\text{salt}}, p_{\text{blur}}, p_{\text{occlusion}}]$.
  4. Decision rule: $\hat{c} = \arg\max(P)$.
  5. **Stage 2 (Specialist Routing)**:
     - If $\hat{c} = \text{clean}$: Identity bypass (no restoration needed, output equals input, specialist latency is $0.0\,\text{ms}$).
     - If $\hat{c} = \text{salt\_and\_pepper}$: Route to `task2_specialist_salt.onnx`.
     - If $\hat{c} = \text{gaussian\_blur}$: Route to `task2_specialist_blur.onnx`.
     - If $\hat{c} = \text{rectangular\_occlusion}$: Route to `task2_specialist_occlusion.onnx`.
- **Response Schema (`HardRoutedRestoreResponse`)**:
  - `original_image`: base64 data URL.
  - `restored_image`: base64 data URL.
  - `class_probabilities`: map containing floating-point probabilities for each class: `clean`, `salt_and_pepper`, `gaussian_blur`, `rectangular_occlusion`.
  - `predicted_class`: string of highest probability class.
  - `selected_expert`: name of the routed specialist or identity bypass.
  - `inference_time`: structured breakdown object containing `classifier_ms`, `specialist_ms`, and `total_ms`.

### Verification
- Test with known clean image: verify predicted class is `clean`, selected expert is `Identity Bypass`, and specialist latency is $0.0\,\text{ms}$.
- Test with blurred image: verify classifier outputs highest probability for `gaussian_blur`, specialist called is `task2_specialist_blur.onnx`, and restored image is deblurred.

### Files Changed
- `src/app/backend/routers/task2.py`
- `src/app/backend/schemas/task2.py`
- `src/app/backend/main.py` (mount task2 router)

---

## Step 5: Backend — Soft MoE Restoration Workspace

### Scope
Implement the inference endpoint for Task 3 (Soft Mixture-of-Experts Restoration), serving a model where all 4 experts (identity bypass + 3 trained specialists) contribute to the output weighted by a softmax gating network.

### What to Build
- Soft MoE endpoint `POST /api/v1/restore/soft-moe` in `src/app/backend/routers/task3.py`.
- Request and response schemas in `src/app/backend/schemas/task3.py`.
- MoE ONNX inference execution and weight extraction in `src/app/backend/services/inference.py`.

### Key Details
- **MoE Formulation**:
  The soft mixture model combines expert outputs via gating coefficients:
  $$\hat{x} = \sum_{i=1}^4 w_i E_i(x)$$
  where $w = \text{softmax}(G(x)) \in \mathbb{R}^4$, $\sum_{i=1}^4 w_i = 1.0$, and $E_1(x) = x$ represents the identity bypass.
- **ONNX Model Graph**: The exported model `task3_moe.onnx` produces two outputs:
  - `restored_image`: tensor of shape `[1, 3, 128, 128]`.
  - `routing_weights`: tensor of shape `[1, 4]` containing $[w_{\text{clean}}, w_{\text{salt}}, w_{\text{blur}}, w_{\text{occlusion}}]$.
- **Response Schema (`SoftMoERestoreResponse`)**:
  - `original_image`: base64 data URL.
  - `restored_image`: base64 data URL.
  - `routing_weights`: dictionary mapping expert identifiers (`identity_clean`, `expert_salt`, `expert_blur`, `expert_occlusion`) to gating weights (floats summing to 1.0).
  - `dominant_expert`: identifier of the expert with the highest gating weight.
  - `inference_time_ms`: float total model latency in milliseconds.

### Verification
- Upload test images with single and mixed corruptions.
- Validate that the returned routing weights sum to $1.0 \pm 10^{-5}$.
- Verify that `dominant_expert` accurately reflects the largest weight coefficient.

### Files Changed
- `src/app/backend/routers/task3.py`
- `src/app/backend/schemas/task3.py`
- `src/app/backend/main.py` (mount task3 router)

---

## Step 6: Backend — Face-to-Sketch Workspace

### Scope
Implement the synthesis inference endpoint for Task 4 (FS2K Conditional GAN), generating facial sketches conditioned on user-selected styles (Style 1, 2, or 3).

### What to Build
- Synthesis endpoint `POST /api/v1/sketch/generate` in `src/app/backend/routers/task4.py`.
- Request parameters and response schemas in `src/app/backend/schemas/task4.py`.
- Conditioning tensor formatting and generator inference in `src/app/backend/services/inference.py`.

### Key Details
- **Input Parameters**:
  - `image`: uploaded face photo (`multipart/form-data`).
  - `style`: integer or enum $\in \{1, 2, 3\}$, designating the FS2K target sketch style.
- **Conditioning Mechanism**:
  - Preprocess face photo to `[1, 3, 128, 128]` float32.
  - Format style conditioning input according to the generator's exported ONNX signature (e.g., style index integer scalar/tensor `[1]` or one-hot vector `[1, 3]`).
- **Inference**:
  - Run `task4_cgan_generator.onnx` using the photo tensor and style conditioning tensor.
  - Output sketch tensor `[1, 3, 128, 128]` (or `[1, 1, 128, 128]` depending on grayscale/RGB format) is scaled back to uint8.
  - Convert to PNG and encode as base64 data URL.
  - The discriminator network is not loaded in backend inference (training only).
- **Response Schema (`SketchGenerateResponse`)**:
  - `original_image`: base64 data URL of the input face photo.
  - `sketch_image`: base64 data URL of synthesized sketch.
  - `selected_style`: integer style index (1, 2, or 3).
  - `style_description`: human-readable label (e.g., "FS2K Style 1 - Pencil Sketch").
  - `inference_time_ms`: float generator execution latency.

### Verification
- Submit a face photo with `style=1`, `style=2`, and `style=3`.
- Confirm synthesis produces distinct sketch visual traits for each style and completes without tensor dimension mismatch.

### Files Changed
- `src/app/backend/routers/task4.py`
- `src/app/backend/schemas/task4.py`
- `src/app/backend/main.py` (mount task4 router)

---

## Step 7: React Frontend Skeleton

### Scope
Scaffold the frontend single-page application using React, Vite, TypeScript, and Tailwind CSS. Implement the application layout shell, client-side routing, shared UI components, and API service layer.

### What to Build
- Project scaffolding (`package.json`, `vite.config.ts`, `tailwind.config.js`, `tsconfig.json`).
- Application shell in `src/components/layout/Shell.tsx` and navbar in `src/components/layout/Navbar.tsx` implementing the layout decided in Step 1.
- Client-side routing with `react-router-dom` in `src/App.tsx`.
- Centralized API service layer in `src/services/api.ts`.
- Reusable UI component library in `src/components/shared/`:
  - `ImageUploader.tsx`: Drag-and-drop file upload with preview, size/type checking.
  - `ImagePanel.tsx`: Side-by-side comparison container with zoom and pan.
  - `LoadingSpinner.tsx`: Animated loading states and progress skeletons.
  - `ErrorAlert.tsx`: Dismissible error messages with retry triggers.
  - `MetricBadge.tsx`: Visual badge for inference times and status indicators.

### Key Details
- **Routing Configuration**:
  - `/` redirects to `/universal`.
  - `/universal` -> Universal Restoration Workspace.
  - `/hard-routed` -> Hard-Routed Restoration Workspace.
  - `/soft-moe` -> Soft Mixture-of-Experts Workspace.
  - `/face-to-sketch` -> Face-to-Sketch Generator Workspace.
- **Frontend Invariants**: Strict compliance with `docs/invariants/frontend.md`:
  - Tailwind CSS for all styling (no CSS modules or styled-components).
  - Maximum ~150 lines per component file; strict TypeScript typing with zero `any`.
  - All network calls encapsulated within `src/services/api.ts` with typed Axios/Fetch methods.
- **Backend Health Polling**: App shell checks `GET /health` on mount, displaying a green/amber status badge indicating whether ONNX models are ready.

### Verification
- Start frontend with `pnpm dev`.
- Confirm clean build without TypeScript or lint errors.
- Verify smooth client-side routing between all four workspace routes.

### Files Changed
- `src/app/frontend/package.json`
- `src/app/frontend/vite.config.ts`
- `src/app/frontend/tailwind.config.js`
- `src/app/frontend/tsconfig.json`
- `src/app/frontend/src/index.css`
- `src/app/frontend/src/App.tsx`
- `src/app/frontend/src/main.tsx`
- `src/app/frontend/src/types/index.ts`
- `src/app/frontend/src/services/api.ts`
- `src/app/frontend/src/components/layout/Shell.tsx`
- `src/app/frontend/src/components/layout/Navbar.tsx`
- `src/app/frontend/src/components/shared/ImageUploader.tsx`
- `src/app/frontend/src/components/shared/ImagePanel.tsx`
- `src/app/frontend/src/components/shared/LoadingSpinner.tsx`
- `src/app/frontend/src/components/shared/ErrorAlert.tsx`
- `src/app/frontend/src/components/shared/MetricBadge.tsx`

---

## Step 8: Frontend — Universal Restoration Page

### Scope
Construct the user interface for Workspace 1 (`/universal`), providing image upload, clean sample selection with synthetic corruption controls, restoration invocation, and visual inspection of reconstructed images and error heatmaps.

### What to Build
- Workspace page component in `src/app/frontend/src/pages/UniversalRestoration.tsx`.
- Synthetic corruption controls in `src/app/frontend/src/components/task1/CorruptionControls.tsx`.
- Error map visualization container in `src/app/frontend/src/components/task1/ErrorMapViewer.tsx`.
- Preset clean sample gallery in `src/app/frontend/src/components/task1/SampleGallery.tsx`.

### Key Details
- **Dual Input Modes**:
  1. *Upload Corrupted Image*: Drag-and-drop or select an existing corrupted photo.
  2. *Synthetic Corruption Studio*: Select a bundled clean pet image, choose corruption type (Salt-and-Pepper, Gaussian Blur, Rectangular Occlusion), and select severity via a slider or step selector corresponding to assignment validation points ($0.03, 0.08, 0.15$ for salt; $(3, 0.7), (5, 1.5), (7, 2.5)$ for blur; $1, 2, 3$ boxes for occlusion).
- **Execution & Visual Presentation**:
  - "Restore Image" action button triggers `api.restoreUniversal(formData)`.
  - Side-by-side view: Corrupted Input (left) vs Reconstructed Output (right).
  - Below image pair: Absolute difference error map with color intensity legend.
  - Header badge displaying `Inference: XX ms`.

### Verification
- Select a clean sample, set Gaussian blur severity $(5, 1.5)$, click Restore.
- Verify corrupted preview, reconstructed output, and residual heatmap appear with accurate latency metrics.

### Files Changed
- `src/app/frontend/src/pages/UniversalRestoration.tsx`
- `src/app/frontend/src/components/task1/CorruptionControls.tsx`
- `src/app/frontend/src/components/task1/ErrorMapViewer.tsx`
- `src/app/frontend/src/components/task1/SampleGallery.tsx`

---

## Step 9: Frontend — Hard-Routed Restoration Page

### Scope
Construct the user interface for Workspace 2 (`/hard-routed`), displaying the 4-class classifier prediction probabilities, highlighted routing selection, specialist autoencoder restoration output, and decomposed execution latency.

### What to Build
- Workspace page component in `src/app/frontend/src/pages/HardRoutedRestoration.tsx`.
- Classifier probabilities bar chart in `src/app/frontend/src/components/task2/ClassifierProbabilities.tsx`.
- Routing decision banner in `src/app/frontend/src/components/task2/RoutingCard.tsx`.
- Latency breakdown badge in `src/app/frontend/src/components/task2/LatencyBreakdown.tsx`.

### Key Details
- **Workflow & UI Elements**:
  - Upload input image via `ImageUploader`.
  - "Analyze & Restore" button triggers `api.restoreHardRouted(formData)`.
  - **Classification Probabilities Card**: Four horizontal progress bars displaying percentage probabilities with color coding:
    - Clean: Emerald Green
    - Salt-and-Pepper: Amber Orange
    - Gaussian Blur: Sky Blue
    - Rectangular Occlusion: Rose Red
  - **Routing Decision Callout**: Prominent card highlighting:
    - Predicted Corruption: e.g., `Gaussian Blur (94.2%)`
    - Routed Specialist: e.g., `task2_specialist_blur.onnx` or `Identity Bypass`
  - **Latency Decomposition**: Visual chip displaying:
    - Classifier Time: $t_{\text{cls}}\,\text{ms}$
    - Specialist Time: $t_{\text{spec}}\,\text{ms}$
    - Total Time: $t_{\text{total}}\,\text{ms}$
  - Side-by-side input vs reconstructed output viewer.

### Verification
- Upload salt-and-pepper corrupted image.
- Verify classifier progress bar for salt-and-pepper shows dominant percentage, specialist card indicates `Salt Specialist`, and latency displays both classifier and specialist timings.

### Files Changed
- `src/app/frontend/src/pages/HardRoutedRestoration.tsx`
- `src/app/frontend/src/components/task2/ClassifierProbabilities.tsx`
- `src/app/frontend/src/components/task2/RoutingCard.tsx`
- `src/app/frontend/src/components/task2/LatencyBreakdown.tsx`

---

## Step 10: Frontend — Soft MoE Restoration Page

### Scope
Construct the user interface for Workspace 3 (`/soft-moe`), highlighting continuous gating coefficients across all 4 experts, visual indicators for dominant contributing experts, reconstructed output, and inference metrics.

### What to Build
- Workspace page component in `src/app/frontend/src/pages/SoftMoERestoration.tsx`.
- Gating weights visualizer in `src/app/frontend/src/components/task3/GatingWeights.tsx`.
- Expert contribution breakdown card in `src/app/frontend/src/components/task3/ExpertContribution.tsx`.

### Key Details
- **Workflow & UI Elements**:
  - Image upload area with preview.
  - "Run Soft MoE Restoration" action button triggering `api.restoreSoftMoE(formData)`.
  - **Gating Weights Panel**:
    - Displays 4 weight gauges / horizontal bars: Identity (Clean), Salt Expert, Blur Expert, Occlusion Expert.
    - Each gauge displays exact percentage ($w_i \times 100\%$).
    - Dominant expert ($w_{\max}$) highlighted with a glowing accent border and badge.
  - **Expert Blending Explanation**: Small visual schematic illustrating how output is the convex combination $\hat{x} = \sum w_i E_i(x)$.
  - Side-by-side comparison of input and soft-reconstructed result with inference timing badge.

### Verification
- Upload test images with single and mixed corruptions.
- Verify all 4 weight bars render correctly, percentages sum to 100%, and dominant expert is highlighted.

### Files Changed
- `src/app/frontend/src/pages/SoftMoERestoration.tsx`
- `src/app/frontend/src/components/task3/GatingWeights.tsx`
- `src/app/frontend/src/components/task3/ExpertContribution.tsx`

---

## Step 11: Frontend — Face-to-Sketch Page

### Scope
Construct the user interface for Workspace 4 (`/face-to-sketch`), enabling face photo synthesis into sketches with live webcam snapshot support, 3-way style switching, side-by-side comparison, and sketch download capability.

### What to Build
- Workspace page component in `src/app/frontend/src/pages/FaceToSketch.tsx`.
- Live webcam snapshot capture component in `src/app/frontend/src/components/task4/WebcamCapture.tsx`.
- Style selector interactive card group in `src/app/frontend/src/components/task4/StyleSelector.tsx`.
- Download action button and preview panel in `src/app/frontend/src/components/task4/SketchViewer.tsx`.

### Key Details
- **Dual Input Modes**:
  1. *Photo File Upload*: Standard drag-and-drop image selection.
  2. *Live Webcam Capture*: Uses browser `navigator.mediaDevices.getUserMedia` video stream with snapshot freeze and accept/retake controls.
- **Style Selection Controls**:
  - Interactive radio cards for the 3 FS2K styles (Style 1, Style 2, Style 3).
  - Each card displays sample sketch texture thumbnail and descriptive style title.
- **Execution & Export**:
  - "Generate Sketch" button calls `api.generateSketch(formData)`.
  - Side-by-side display: Original Face (left) vs Synthesized Sketch (right).
  - Download Button: Triggers direct client download of generated sketch as PNG (`sketch_style_{id}_{timestamp}.png`).
  - Inference time badge.

### Verification
- Test file upload and webcam snapshot capture.
- Switch between Style 1, 2, and 3, verify generated sketch updates, and test file download.

### Files Changed
- `src/app/frontend/src/pages/FaceToSketch.tsx`
- `src/app/frontend/src/components/task4/WebcamCapture.tsx`
- `src/app/frontend/src/components/task4/StyleSelector.tsx`
- `src/app/frontend/src/components/task4/SketchViewer.tsx`

---

## Step 12: Docker Compose Deployment

### Scope
Create production-grade containerization for both backend and frontend, and configure multi-container orchestration with Docker Compose for a single-command startup.

### What to Build
- Two-stage backend Dockerfile in `Dockerfile.backend`.
- Two-stage frontend Dockerfile in `Dockerfile.frontend`.
- Nginx reverse proxy configuration in `nginx.conf`.
- Multi-container orchestration specification in `docker-compose.yml`.
- Container ignore file in `.dockerignore`.

### Key Details
- **Backend Dockerfile (`Dockerfile.backend`)**:
  - *Stage 1 (Builder)*: Base `python:3.11-slim`, install `uv`, copy `pyproject.toml` and `uv.lock`, run `uv sync --frozen --no-dev`.
  - *Stage 2 (Runtime)*: Base `python:3.11-slim`, copy virtual environment from builder, copy `src/app/backend`, non-root user setup, expose port 8000.
  - Entrypoint: `uvicorn src.app.backend.main:app --host 0.0.0.0 --port 8000`.
- **Frontend Dockerfile (`Dockerfile.frontend`)**:
  - *Stage 1 (Builder)*: Base `node:20-alpine`, enable `pnpm`, copy `package.json` and `pnpm-lock.yaml`, run `pnpm install --frozen-lockfile`, copy frontend source, run `pnpm build`.
  - *Stage 2 (Runtime)*: Base `nginx:alpine`, copy built static distribution from builder to `/usr/share/nginx/html`, copy `nginx.conf`, expose port 80.
- **Nginx Configuration (`nginx.conf`)**:
  - Proxies `/api/` requests to `http://backend:8000/api/` and `/health` to `http://backend:8000/health`.
  - Configures `try_files $uri $uri/ /index.html` for client-side React routing.
  - Eliminates browser CORS issues in containerized production.
- **Docker Compose (`docker-compose.yml`)**:
  - `backend` service: exposes port 8000, mounts `./models/onnx:/app/models/onnx:ro` as a volume so models do not need image rebuilding, defines health check on `http://localhost:8000/health` (interval 10s, timeout 5s, retries 3).
  - `frontend` service: exposes port 80, depends on `backend` with `condition: service_healthy`.

### Verification
- Run `docker compose up --build`.
- Verify backend passes healthcheck and frontend container starts.
- Open `http://localhost` in browser and confirm all four workspaces function end-to-end.

### Files Changed
- `Dockerfile.backend`
- `Dockerfile.frontend`
- `nginx.conf`
- `docker-compose.yml`
- `.dockerignore`

---

## Step 13: Integration Testing & Polish

### Scope
Conduct full end-to-end testing across all four workspaces inside Docker, validate error handling and edge cases, polish UI responsiveness, and document evaluator execution steps in the project README.

### What to Build
- Automated smoke test script or test matrix verifying all four workspace endpoints.
- Robust error boundary handling for model-missing states, oversized files, and invalid formats.
- Complete evaluator instructions in `README.md`.

### Key Details
- **End-to-End Test Matrix**:
  1. *Universal Restoration*: Upload corrupted image; test synthetic corruption slider; verify restored image and error map render.
  2. *Hard-Routed Restoration*: Test images for all 4 corruption classes; verify classification probabilities, specialist routing, and timing breakdown.
  3. *Soft MoE Restoration*: Upload sample image; verify 4 routing weights sum to 1.0; check dominant expert highlighting.
  4. *Face-to-Sketch*: Upload photo; test webcam snapshot; generate sketches for Styles 1, 2, and 3; test PNG download button.
- **Edge Case & Resilience Testing**:
  - Non-image upload: Returns HTTP 400 with user-friendly alert in frontend.
  - Oversized file (>10MB): Returns HTTP 413 Payload Too Large.
  - Missing ONNX model: Returns HTTP 503 Service Unavailable with clear actionable error message instead of 500 server crash.
- **Evaluator Workflow Documentation**:
  Provide an unambiguous walkthrough in `README.md`:
  1. Clone repository: `git clone <repo-url>`
  2. Download ONNX models into `models/onnx/` via provided script or download link.
  3. Run single command: `docker compose up --build`.
  4. Open `http://localhost` in browser.

### Verification
- Perform a clean clone into a separate test directory, follow `README.md` instructions, execute `docker compose up --build`, and verify all workflows succeed.

### Files Changed
- `README.md`
- `src/app/backend/routers/health.py`
- `src/app/backend/services/inference.py`
- `src/app/frontend/src/services/api.ts`
- `src/app/frontend/src/components/shared/ErrorAlert.tsx`
