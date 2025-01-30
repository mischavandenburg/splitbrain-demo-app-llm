FROM nginx:alpine

# Add bash for the entrypoint script
RUN apk add --no-cache bash

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy the static files
COPY index.html /usr/share/nginx/html/

# Copy entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Default to Docker Compose URL if not specified
ENV BACKEND_URL="http://backend:8000"

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]
