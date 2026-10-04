/** Centralized API service layer for PixelMend backend endpoints */

import type {
  HealthResponse,
  UniversalRestoreResponse,
  HardRoutedRestoreResponse,
  SoftMoERestoreResponse,
  SketchGenerateResponse,
} from '../types';

const API_BASE = '';

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorDetail = `Request failed with status ${response.status}`;
    try {
      const errorJson = await response.json();
      if (typeof errorJson.detail === 'string') {
        errorDetail = errorJson.detail;
      } else if (Array.isArray(errorJson.detail)) {
        errorDetail = errorJson.detail.map((d: { msg?: string }) => d.msg).filter(Boolean).join('; ') || errorDetail;
      }
    } catch {
      // Fallback to status text
    }
    throw new Error(errorDetail);
  }
  return response.json();
}

export const api = {
  /** Fetch backend system health and ONNX models loading status */
  async getHealth(): Promise<HealthResponse> {
    const res = await fetch(`${API_BASE}/health`);
    return handleResponse<HealthResponse>(res);
  },

  /** Task 1: Universal Autoencoder Restoration */
  async restoreUniversal(formData: FormData): Promise<UniversalRestoreResponse> {
    const res = await fetch(`${API_BASE}/api/v1/restore/universal`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse<UniversalRestoreResponse>(res);
  },

  /** Task 2: Hard-Routed Specialist Restoration */
  async restoreHardRouted(formData: FormData): Promise<HardRoutedRestoreResponse> {
    const res = await fetch(`${API_BASE}/api/v1/restore/hard-routed`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse<HardRoutedRestoreResponse>(res);
  },

  /** Task 3: Soft Mixture-of-Experts Differentiable Restoration */
  async restoreSoftMoE(formData: FormData): Promise<SoftMoERestoreResponse> {
    const res = await fetch(`${API_BASE}/api/v1/restore/soft-moe`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse<SoftMoERestoreResponse>(res);
  },

  /** Task 4: Style-Conditioned Face-to-Sketch Synthesis */
  async generateSketch(formData: FormData): Promise<SketchGenerateResponse> {
    const res = await fetch(`${API_BASE}/api/v1/sketch/generate`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse<SketchGenerateResponse>(res);
  },
};
