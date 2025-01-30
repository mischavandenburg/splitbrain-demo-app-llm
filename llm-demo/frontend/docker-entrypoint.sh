#!/bin/bash

# Replace the backend URL placeholder in index.html
sed -i "s|%%BACKEND_URL%%|${BACKEND_URL}|g" /usr/share/nginx/html/index.html

# Execute CMD
exec "$@"
