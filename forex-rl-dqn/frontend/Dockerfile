FROM nginx:alpine

# Copia o frontend
COPY *.html /usr/share/nginx/html/

# Copia configuração customizada do nginx (opcional)
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
