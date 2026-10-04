import React, { useState } from 'react';
import { GitFork, Wand2 } from 'lucide-react';
import { ImageUploader } from '../components/shared/ImageUploader';
import { ImagePanel } from '../components/shared/ImagePanel';
import { MetricBadge } from '../components/shared/MetricBadge';
import { LoadingSpinner } from '../components/shared/LoadingSpinner';
import { ErrorAlert } from '../components/shared/ErrorAlert';
import { CorruptionControls } from '../components/task1/CorruptionControls';
import { SampleGallery } from '../components/task1/SampleGallery';
import { ErrorMapViewer } from '../components/task1/ErrorMapViewer';
import { ClassifierProbabilities } from '../components/task2/ClassifierProbabilities';
import { RoutingCard } from '../components/task2/RoutingCard';
import { LatencyBreakdown } from '../components/task2/LatencyBreakdown';
import { api } from '../services/api';
import type { CorruptionType, HardRoutedRestoreResponse } from '../types';

export const HardRoutedRestoration: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [applyCorruption, setApplyCorruption] = useState(false);
  const [corruptionType, setCorruptionType] = useState<CorruptionType>('gaussian_blur');
  const [severity, setSeverity] = useState(2);
  const [forceExpert, setForceExpert] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<HardRoutedRestoreResponse | null>(null);

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

  const handleRestore = async (override?: string | null) => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('apply_corruption', String(applyCorruption));
      formData.append('corruption_type', corruptionType);
      formData.append('severity', String(severity));
      const activeOverride = override !== undefined ? override : forceExpert;
      if (activeOverride) {
        formData.append('force_expert', activeOverride);
      }
      const res = await api.restoreHardRouted(formData);
      setResult(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Hard-routed restoration failed');
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
            <div className="p-2 rounded-lg bg-amber-950/60 border border-amber-800/60 text-amber-400">
              <GitFork className="w-5 h-5" />
            </div>
            <h2 className="text-lg font-bold text-slate-100">Hard-Routed Restoration Workspace</h2>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Task 2: Sequential pipeline routing clean to identity bypass and corruptions to dedicated specialist autoencoders.
          </p>
        </div>
        {result && (
          <div className="flex items-center gap-2">
            <MetricBadge label="Total Time" value={result.inference_time.total_ms} unit="ms" variant="amber" />
            <MetricBadge label="Specialist" value={result.selected_expert} variant="slate" icon="cpu" />
          </div>
        )}
      </div>

      {error && <ErrorAlert message={error} onDismiss={() => setError(null)} onRetry={() => handleRestore()} />}

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
              onClick={() => handleRestore()}
              disabled={!file || loading}
              className={`w-full py-2.5 px-4 rounded-xl font-semibold text-xs flex items-center justify-center gap-2 transition-all shadow-md ${
                !file || loading
                  ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                  : 'bg-amber-500 hover:bg-amber-400 text-slate-950 shadow-amber-500/20 active:scale-[0.99]'
              }`}
            >
              <Wand2 className="w-4 h-4" />
              {loading ? 'Analyzing & Routing...' : 'Analyze & Restore'}
            </button>
          </div>
        </div>

        <div className="lg:col-span-7 space-y-4">
          {loading ? (
            <div className="p-12 rounded-xl border border-slate-800 bg-slate-900/40">
              <LoadingSpinner label="Classifying corruption and routing to specialist..." variant="amber" size="lg" />
            </div>
          ) : result ? (
            <div className="space-y-4">
              <RoutingCard
                predictedClass={result.predicted_class}
                confidence={result.confidence}
                selectedExpert={result.selected_expert}
                forceExpert={forceExpert}
                onForceExpertChange={(val) => {
                  setForceExpert(val);
                  handleRestore(val);
                }}
              />
              <ClassifierProbabilities
                probabilities={result.class_probabilities}
                predictedClass={result.predicted_class}
              />
              <LatencyBreakdown latency={result.inference_time} />
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <ImagePanel
                  title="Input Image"
                  subtitle={`Detected: ${result.predicted_class}`}
                  imageUrl={result.corrupted_image}
                  badge="Input"
                  downloadFilename="pixelmend-routed-input.png"
                />
                <ImagePanel
                  title="Restored Output"
                  subtitle={result.selected_expert}
                  imageUrl={result.restored_image}
                  badge="Output"
                  downloadFilename="pixelmend-routed-output.png"
                />
              </div>
              <ErrorMapViewer errorMap={result.error_map} />
            </div>
          ) : (
            <div className="p-12 rounded-xl border border-dashed border-slate-800 bg-slate-900/20 text-center text-slate-500">
              <GitFork className="w-8 h-8 mx-auto opacity-30 mb-2 text-amber-400" />
              <p className="text-xs">Upload an image or choose a sample to inspect Stage 1 classification and Stage 2 specialist routing.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default HardRoutedRestoration;
