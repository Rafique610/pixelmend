import React, { useRef, useState, useEffect } from 'react';
import { Camera, CameraOff, Check } from 'lucide-react';

interface WebcamCaptureProps {
  onCapture: (file: File, previewUrl: string) => void;
  onClose?: () => void;
}

export const WebcamCapture: React.FC<WebcamCaptureProps> = ({ onCapture, onClose }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [isActive, setIsActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [snapshotUrl, setSnapshotUrl] = useState<string | null>(null);

  const startCamera = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 512 }, height: { ideal: 512 }, facingMode: 'user' },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setIsActive(true);
    } catch (err) {
      setError('Unable to access camera. Please check browser permissions.');
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setIsActive(false);
  };

  useEffect(() => {
    return () => stopCamera();
  }, []);

  const takeSnapshot = () => {
    if (!videoRef.current) return;
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 128;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(videoRef.current, 0, 0, 128, 128);
    canvas.toBlob((blob) => {
      if (!blob) return;
      const file = new File([blob], 'webcam_portrait.jpg', { type: 'image/jpeg' });
      const url = URL.createObjectURL(blob);
      setSnapshotUrl(url);
      stopCamera();
      onCapture(file, url);
    }, 'image/jpeg', 0.95);
  };

  return (
    <div className="p-4 rounded-xl border border-slate-700 bg-slate-900/80 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-xs font-semibold text-slate-200">
          <Camera className="w-4 h-4 text-pink-400" />
          <span>Live Webcam Snapshot</span>
        </div>
        {onClose && (
          <button
            type="button"
            onClick={() => {
              stopCamera();
              onClose();
            }}
            className="text-xs text-slate-400 hover:text-slate-200"
          >
            Cancel
          </button>
        )}
      </div>

      {error && <p className="text-xs text-red-400">{error}</p>}

      <div className="relative aspect-square max-w-[240px] mx-auto rounded-lg overflow-hidden bg-slate-950 flex items-center justify-center border border-slate-800">
        <video
          ref={videoRef}
          playsInline
          muted
          className={`w-full h-full object-cover ${isActive ? 'block' : 'hidden'}`}
        />
        {!isActive && !snapshotUrl && (
          <div className="text-center p-4 text-slate-500">
            <CameraOff className="w-8 h-8 mx-auto mb-1 opacity-40" />
            <p className="text-[11px]">Camera inactive</p>
          </div>
        )}
        {snapshotUrl && !isActive && (
          <img src={snapshotUrl} alt="Snapshot Preview" className="w-full h-full object-cover" />
        )}
      </div>

      <div className="flex justify-center gap-2">
        {!isActive ? (
          <button
            type="button"
            onClick={startCamera}
            className="px-3 py-1.5 rounded-lg bg-pink-600 hover:bg-pink-500 text-white text-xs font-medium flex items-center gap-1.5 transition-colors"
          >
            <Camera className="w-3.5 h-3.5" />
            {snapshotUrl ? 'Retake Photo' : 'Start Camera'}
          </button>
        ) : (
          <button
            type="button"
            onClick={takeSnapshot}
            className="px-4 py-1.5 rounded-lg bg-pink-500 hover:bg-pink-400 text-slate-950 text-xs font-bold flex items-center gap-1.5 transition-colors shadow-md"
          >
            <Check className="w-3.5 h-3.5" />
            Capture Snapshot
          </button>
        )}
      </div>
    </div>
  );
};
