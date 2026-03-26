/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{vue,js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#6366f1',
          hover: '#818cf8',
          dark: '#4f46e5',
        },
        background: {
          DEFAULT: '#0f172a',
          secondary: '#1e293b',
          tertiary: '#334155',
        },
      },
    },
  },
  plugins: [],
}
