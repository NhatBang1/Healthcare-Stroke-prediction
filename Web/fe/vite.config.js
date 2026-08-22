import { resolve } from "node:path";
import { defineConfig } from "vite";

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
        auth: resolve(__dirname, "auth.html"),
        forgotPassword: resolve(__dirname, "forgot-password.html"),
        settings: resolve(__dirname, "settings.html"),
        history: resolve(__dirname, "history.html"),
        progress: resolve(__dirname, "progress.html"),
        admin: resolve(__dirname, "admin.html"),
        adminDocuments: resolve(__dirname, "admin-documents.html"),
        adminUsers: resolve(__dirname, "admin-users.html"),
        adminInsights: resolve(__dirname, "admin-insights.html"),
        adminSettings: resolve(__dirname, "admin-settings.html")
      }
    }
  }
});
