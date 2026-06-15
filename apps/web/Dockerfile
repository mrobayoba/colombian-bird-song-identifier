# ── Web Frontend (React PWA) ─────────────────────────────────────────────────
# Stage 1 builds the static site with Node. Stage 2 serves it with nginx.
# The final image contains only static files — small and fast.
#
# Build context MUST be the repo root:
#   docker build -f apps/web/Dockerfile -t cbsi-web .
#
# VITE_API_BASE is baked in at BUILD time (Vite inlines env vars). For the
# compose setup we leave it empty and let nginx proxy /api to the index agent.

# ---- Stage 1: build ----
FROM node:20-slim AS build
WORKDIR /app
COPY apps/web/package.json apps/web/package-lock.json* ./
RUN npm install
COPY apps/web/ ./
# Empty API base → frontend calls same-origin /api, which nginx proxies.
ENV VITE_API_BASE=""
RUN npm run build

# ---- Stage 2: serve ----
FROM nginx:1.27-alpine
# SPA + /api proxy config (see apps/web/nginx.conf).
COPY apps/web/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
