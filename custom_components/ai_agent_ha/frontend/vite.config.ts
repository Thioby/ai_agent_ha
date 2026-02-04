import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import path from 'path';

// Production build config
export default defineConfig({
  plugins: [svelte()],
  resolve: {
    alias: {
      '$lib': path.resolve(__dirname, './src/lib')
    }
  },
  build: {
    lib: {
      entry: './src/main.ts',
      name: 'AiAgentHaPanel',
      fileName: () => 'ai_agent_ha-panel.js',
      formats: ['iife']
    },
    outDir: '../',
    emptyOutDir: false, // Don't delete other files in custom_components
    minify: 'terser',
    sourcemap: false,
    rollupOptions: {
      output: {
        inlineDynamicImports: true, // Single file bundle
        manualChunks: undefined // Prevent code splitting
      }
    },
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true
      }
    }
  }
});
