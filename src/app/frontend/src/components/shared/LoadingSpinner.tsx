import React from 'react';

interface LoadingSpinnerProps {
  label?: string;
  size?: 'sm' | 'md' | 'lg';
  variant?: 'cyan' | 'amber' | 'purple' | 'pink' | 'slate';
}

const sizeClasses = {
  sm: 'w-4 h-4 border-2',
  md: 'w-6 h-6 border-2',
  lg: 'w-10 h-10 border-3',
};

const colorClasses = {
  cyan: 'border-cyan-500/20 border-t-cyan-400',
  amber: 'border-amber-500/20 border-t-amber-400',
  purple: 'border-purple-500/20 border-t-purple-400',
  pink: 'border-pink-500/20 border-t-pink-400',
  slate: 'border-slate-500/20 border-t-slate-300',
};

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  label = 'Processing inference...',
  size = 'md',
  variant = 'cyan',
}) => {
  return (
    <div className="flex flex-col items-center justify-center gap-3 p-8 text-center">
      <div
        className={`rounded-full animate-spin ${sizeClasses[size]} ${colorClasses[variant]}`}
      />
      {label && (
        <p className="text-xs font-medium text-slate-400 tracking-wide">
          {label}
        </p>
      )}
    </div>
  );
};
