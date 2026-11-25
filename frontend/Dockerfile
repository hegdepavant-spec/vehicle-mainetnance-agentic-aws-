# Stage 1: Build React/Vite SPA
FROM node:20-alpine AS build

WORKDIR /app

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# Stage 2: Serve static bundle and reverse-proxy API via Nginx
FROM nginx:alpine

# Remove default configuration and copy custom reverse proxy configuration
RUN rm -rf /etc/nginx/conf.d/*
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Copy compiled frontend assets from build stage
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=20s --timeout=5s --retries=3 \
    CMD wget -q -O - http://localhost/nginx-health || exit 1

CMD ["nginx", "-g", "daemon off;"]
