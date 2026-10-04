/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        studio: {
          bg: '#0F172A',
          card: '#1E293B',
          hover: '#334155',
          border: '#334155',
          muted: '#64748B',
          secondary: '#94A3B8',
          text: '#F8FAFC',
        },
        task: {
          universal: '#06B6D4',
          hard: '#F59E0B',
          moe: '#8B5CF6',
          sketch: '#EC4899',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
    },
  },
  plugins: [],
}
