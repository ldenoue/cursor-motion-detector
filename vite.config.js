import { resolve } from "node:path";
import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  build: {
    rollupOptions: {
      input: {
        article: resolve(import.meta.dirname, "index.html"),
        demo: resolve(import.meta.dirname, "demo.html")
      }
    }
  }
});
