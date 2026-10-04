import React from 'react';
import { AlertCircle, X, RefreshCw } from 'lucide-react';

interface ErrorAlertProps {
  message: string;
  onDismiss?: () => void;
  onRetry?: () => void;
}

export const ErrorAlert: React.FC<ErrorAlertProps> = ({
  message,
  onDismiss,
  onRetry,
}) => {
  return (
    <div className="flex items-start justify-between gap-3 p-4 rounded-xl bg-red-950/40 border border-red-800/60 text-red-200">
      <div className="flex items-start gap-3">
        <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
        <div>
          <h4 className="text-sm font-semibold text-red-300">Inference Error</h4>
          <p className="text-xs text-red-200/90 mt-0.5 leading-relaxed">{message}</p>
        </div>
      </div>
      <div className="flex items-center gap-2 shrink-0">
        {onRetry && (
          <button
            onClick={onRetry}
            className="flex items-center gap-1 px-2.5 py-1 text-xs font-medium rounded-lg bg-red-900/60 hover:bg-red-800/80 text-red-200 transition-colors"
          >
            <RefreshCw className="w-3 h-3" />
            Retry
          </button>
        )}
        {onDismiss && (
          <button
            onClick={onDismiss}
            className="p-1 rounded-lg hover:bg-red-900/50 text-red-400 hover:text-red-200 transition-colors"
            aria-label="Dismiss error"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  );
};
