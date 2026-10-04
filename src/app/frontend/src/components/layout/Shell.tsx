import React, { useState, useEffect } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  Sparkles,
  GitFork,
  Network,
  Palette,
  ChevronLeft,
  ChevronRight,
  ShieldCheck,
} from 'lucide-react';
import { Navbar } from './Navbar';
import { api } from '../../services/api';
import type { HealthResponse } from '../../types';

interface ShellProps {
  children: React.ReactNode;
}

const navItems = [
  {
    path: '/universal',
    title: 'Universal Restoration',
    subtitle: 'Task 1: Convolutional Autoencoder',
    icon: Sparkles,
    accent: 'text-cyan-400 border-cyan-500/50 bg-cyan-950/40',
  },
  {
    path: '/hard-routed',
    title: 'Hard-Routed Restoration',
    subtitle: 'Task 2: Classifier + Specialists',
    icon: GitFork,
    accent: 'text-amber-400 border-amber-500/50 bg-amber-950/40',
  },
  {
    path: '/soft-moe',
    title: 'Soft MoE Restoration',
    subtitle: 'Task 3: Differentiable Gating Blend',
    icon: Network,
    accent: 'text-purple-400 border-purple-500/50 bg-purple-950/40',
  },
  {
    path: '/face-to-sketch',
    title: 'Face-to-Sketch Generator',
    subtitle: 'Task 4: Conditional GAN (FS2K)',
    icon: Palette,
    accent: 'text-pink-400 border-pink-500/50 bg-pink-950/40',
  },
];

export const Shell: React.FC<ShellProps> = ({ children }) => {
  const [collapsed, setCollapsed] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);
  const location = useLocation();

  useEffect(() => {
    let mounted = true;
    const fetchHealth = async () => {
      try {
        const data = await api.getHealth();
        if (mounted) setHealth(data);
      } catch (e) {
        if (mounted) setHealth({ status: 'unreachable', provider: 'None', models_loaded: {} });
      } finally {
        if (mounted) setHealthLoading(false);
      }
    };
    fetchHealth();
    const timer = setInterval(fetchHealth, 15000);
    return () => {
      mounted = false;
      clearInterval(timer);
    };
  }, []);

  const currentWorkspace =
    navItems.find((item) => location.pathname.startsWith(item.path))?.title ||
    'Universal Restoration';

  return (
    <div className="flex h-screen bg-[#0F172A] text-slate-100 overflow-hidden font-sans">
      {/* Sidebar */}
      <aside
        className={`flex flex-col border-r border-slate-800 bg-[#141E33] transition-all duration-300 z-20 ${
          collapsed ? 'w-20' : 'w-72'
        }`}
      >
        {/* Sidebar Header */}
        <div className="h-16 flex items-center justify-between px-4 border-b border-slate-800">
          {!collapsed && (
            <div className="flex items-center gap-2">
              <span className="font-bold text-sm tracking-wide text-slate-100">WORKSPACES</span>
            </div>
          )}
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white transition-colors ml-auto"
            title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
          </button>
        </div>

        {/* Navigation Items */}
        <nav className="flex-1 py-4 px-3 space-y-1.5 overflow-y-auto">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-3 rounded-xl border text-sm font-medium transition-all ${
                    isActive
                      ? `${item.accent} shadow-sm shadow-black/40`
                      : 'border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                  }`
                }
                title={collapsed ? item.title : undefined}
              >
                <Icon className="w-5 h-5 shrink-0" />
                {!collapsed && (
                  <div className="truncate">
                    <p className="truncate text-xs font-semibold leading-tight">{item.title}</p>
                    <p className="text-[10px] text-slate-400 truncate mt-0.5">{item.subtitle}</p>
                  </div>
                )}
              </NavLink>
            );
          })}
        </nav>

        {/* Sidebar Footer */}
        {!collapsed && (
          <div className="p-4 border-t border-slate-800/80 bg-slate-900/40 text-[11px] text-slate-400 flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />
            <span className="truncate">Local ONNX Runtime • CPU/CUDA</span>
          </div>
        )}
      </aside>

      {/* Main Canvas Area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <Navbar health={health} healthLoading={healthLoading} workspaceTitle={currentWorkspace} />
        <main className="flex-1 overflow-y-auto p-6 bg-[#0F172A]">{children}</main>
      </div>
    </div>
  );
};
