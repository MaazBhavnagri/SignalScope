# CPU-only image for judges / demos. GPU users should run natively (see README).
FROM node:22-slim AS web
WORKDIR /web
COPY app/frontend/package*.json ./
RUN npm ci
COPY app/frontend .
RUN npm run build

FROM python:3.11-slim
WORKDIR /srv
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY model ./model
COPY app/backend ./app/backend
COPY app/__init__.py ./app/__init__.py
COPY weights ./weights
COPY report ./report
COPY --from=web /web/dist ./app/frontend/dist
EXPOSE 8000
CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
