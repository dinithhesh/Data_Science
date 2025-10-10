FROM python:3.9-slim
WORKDIR /app
COPY api/requirements.txt .
RUN pip install -r requirements.txt
COPY api /app
CMD ['uvicorn', 'app:app', '--host', '0.0.0.0', '--port', '8000']