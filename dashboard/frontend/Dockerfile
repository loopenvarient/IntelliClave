FROM node:20-alpine AS build

WORKDIR /app
COPY dashboard/frontend/intelliclave-ui/package.json dashboard/frontend/intelliclave-ui/package-lock.json* ./
RUN npm install

COPY dashboard/frontend/intelliclave-ui/ ./
ARG VITE_API_URL=http://localhost:8001
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

FROM nginx:alpine
COPY dashboard/frontend/intelliclave-ui/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 3000
