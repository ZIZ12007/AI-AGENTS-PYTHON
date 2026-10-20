/*
 * Runtime configuration for static hosts (GitHub Pages, Vercel, Netlify, ...).
 *
 * The backend serves the frontend itself, so the default (empty) means
 * "same origin" - which is correct for a single-service deploy.
 *
 * For a frontend-only host, point it at a hosted backend by setting the URL
 * here (a base URL is enough, "/api/chat" is added automatically):
 *
 *   window.AGENT_CONFIG = { backendUrl: "https://your-service.onrender.com" };
 *
 * This file is read before app.js, so it is safe to edit in the browser
 * console too.
 */
window.AGENT_CONFIG = window.AGENT_CONFIG || { backendUrl: "" };
