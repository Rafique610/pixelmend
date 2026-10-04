/** TypeScript interfaces for PixelMend API contracts and application state */

export interface HealthResponse {
  status: string;
  provider: string;
  models_loaded: Record<string, boolean>;
}

export interface CorruptionApplied {
  type: string;
  severity?: number;
  p?: number;
  kernel_size?: number;
  sigma?: number;
  num_boxes?: number;
  actual_coverage?: number;
}

export interface UniversalRestoreResponse {
  original_image: string;
  corrupted_image: string;
  restored_image: string;
  error_map: string;
  clean_error_map: string | null;
  corruption_applied: CorruptionApplied | null;
  inference_time_ms: number;
}

export interface LatencyBreakdown {
  classifier_ms: number;
  specialist_ms: number;
  total_ms: number;
}

export interface HardRoutedRestoreResponse {
  original_image: string;
  corrupted_image: string;
  restored_image: string;
  error_map: string;
  class_probabilities: Record<string, number>;
  predicted_class: string;
  confidence: number;
  selected_expert: string;
  inference_time: LatencyBreakdown;
  corruption_applied: CorruptionApplied | null;
}

export interface SoftMoERestoreResponse {
  original_image: string;
  corrupted_image: string;
  restored_image: string;
  error_map: string;
  clean_error_map: string | null;
  routing_weights: Record<string, number>;
  dominant_expert: string;
  dominant_weight: number;
  entropy: number;
  inference_time_ms: number;
  corruption_applied: CorruptionApplied | null;
}

export interface SketchGenerateResponse {
  original_image: string;
  sketch_image: string;
  selected_style: number;
  style_description: string;
  inference_time_ms: number;
}

export type CorruptionType = 'clean' | 'salt_and_pepper' | 'gaussian_blur' | 'rectangular_occlusion';
