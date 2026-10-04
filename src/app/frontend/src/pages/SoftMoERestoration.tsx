import React, { useState } from 'react';
import { Network, Wand2 } from 'lucide-react';
import { ImageUploader } from '../components/shared/ImageUploader';
import { ImagePanel } from '../components/shared/ImagePanel';
import { MetricBadge } from '../components/shared/MetricBadge';
import { LoadingSpinner } from '../components/shared/LoadingSpinner';
import { ErrorAlert } from '../components/shared/ErrorAlert';
import { CorruptionControls } from '../components/task1/CorruptionControls';
import { SampleGallery } from '../components/task1/SampleGallery';
import { ErrorMapViewer } from '../components/task1/ErrorMapViewer';
import { GatingWeights } from '../components/task3/GatingWeights';
import { ExpertContribution } from '../components/task3/ExpertContribution';
import { api } from '../services/api';
import type { CorruptionType, SoftMoERestoreResponse } from '../types';

export const SoftMoERestoration: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [applyCorruption, setApplyCorruption] = useState(false);
  const [corruptionType, setCorruptionType] = useState<CorruptionType>('gaussian_blur');
  const [severity, setSeverity] = useState(2);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SoftMoERestoreResponse | null>(null);

  const handleSelectSample = (sampleFile: File, url: string) => {
    setFile(sampleFile);
    setPreviewUrl(url);
    setResult(null);
    setError(null);
  };

  const handleClear = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  const handleRestore = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('apply_corruption', String(applyCorruption));
      formData.append('corruption_type', corruptionType);
      formData.append('severity', String(severity));
      const res = await api.restoreSoftMoE(formData);
      setResult(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Soft MoE restoration failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-purple-950/60 border border-purple-800/60 text-purple-400">
              <Network className="w-5 h-5" />
            </div>
            <h2 className="text-lg font-bold text-slate-100">Soft Mixture-of-Experts Workspace</h2>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Task 3: Differentiable gating network blending identity bypass and 3 specialists via temperature-scaled softmax weights.
          </p>
        </div>
        {result && (
          <div className="flex items-center gap-2">
            <MetricBadge label="Inference" value={result.inference_time_ms} unit="ms" variant="purple" />
            <MetricBadge label="Model" value="task3_soft_moe.onnx" variant="slate" icon="cpu" />
          </div>
        )}
      </div>

      {error && <ErrorAlert message={error} onDismiss={() => setError(null)} onRetry={handleRestore} />}

      {/* Grid: Left Controls, Right Results */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-5 space-y-4">
          <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 space-y-4">
            <ImageUploader
              onImageSelected={(f) => {
                setFile(f);
                setPreviewUrl(URL.createObjectURL(f));
                setResult(null);
              }}
              onClear={handleClear}
              previewUrl={previewUrl}
            />
            <SampleGallery onSelectSample={handleSelectSample} selectedSample={previewUrl} />
            <CorruptionControls
              applyCorruption={applyCorruption}
              onToggleApply={setApplyCorruption}
              corruptionType={corruptionType}
              onChangeType={setCorruptionType}
              severity={severity}
              onChangeSeverity={setSeverity}
            />
            <button
              onClick={handleRestore}
              disabled={!file || loading}
              className={`w-full py-2.5 px-4 rounded-xl font-semibold text-xs flex items-center justify-center gap-2 transition-all shadow-md ${
                !file || loading
                  ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                  : 'bg-purple-600 hover:bg-purple-500 text-white shadow-purple-600/20 active:scale-[0.99]'
              }`}
            >
              <Wand2 className="w-4 h-4" />
              {loading ? 'Blending Experts...' : 'Run Soft MoE Restoration'}
            </button>
          </div>
        </div>

        <div className="lg:col-span-7 space-y-4">
          {loading ? (
            <div className="p-12 rounded-xl border border-slate-800 bg-slate-900/40">
              <LoadingSpinner label="Evaluating continuous gating network and blending experts..." variant="purple" size="lg" />
            </div>
          ) : result ? (
            <div className="space-y-4">
              <GatingWeights
                routingWeights={result.routing_weights}
                dominantExpert={result.dominant_expert}
                dominantWeight={result.dominant_weight}
                entropy={result.entropy}
              />
              <ExpertContribution
                dominantExpert={result.dominant_expert}
                dominantWeight={result.dominant_weight}
              />
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <ImagePanel
                  title="Input Image"
                  subtitle={result.corruption_applied ? `Corrupted: ${result.corruption_applied.type}` : 'Direct Upload'}
                  imageUrl={result.corrupted_image}
                  badge="Input"
                  downloadFilename="pixelmend-moe-input.png"
                />
                <ImagePanel
                  title="Blended MoE Output"
                  subtitle="Convex Combination x̂ = ∑ wᵢ · Eᵢ(x)"
                  imageUrl={result.restored_image}
                  badge="Blended"
                  downloadFilename="pixelmend-moe-restored.png"
                />
              </div>
              <ErrorMapViewer errorMap={result.error_map} cleanErrorMap={result.clean_error_map} />
            </div>
          ) : (
            <div className="p-12 rounded-xl border border-dashed border-slate-800 bg-slate-900/20 text-center text-slate-500">
              <Network className="w-8 h-8 mx-auto opacity-30 mb-2 text-purple-400" />
              <p className="text-xs">Upload an image or choose a sample to inspect continuous MoE gating distributions and blended restoration.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SoftMoERestoration;
