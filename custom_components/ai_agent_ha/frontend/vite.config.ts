import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import path from 'path';

// Production build config
export default defineConfig({
  plugins: [
    svelte({
      compilerOptions: {
        css: 'injected' // Inject CSS into JavaScript for single-file bundle
      }
    })
  ],
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
    outDir: './',
    emptyOutDir: false,
    minify: 'terser',
    sourcemap: false,
    cssCodeSplit: false, // Don't split CSS
    rollupOptions: {
      output: {
        inlineDynamicImports: true,
        manualChunks: undefined,
        // Inline all assets including CSS
        assetFileNames: (assetInfo) => {
          return 'ai_agent_ha-panel.css';
        }
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
