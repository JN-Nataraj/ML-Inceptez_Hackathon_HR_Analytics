FROM python:3.10.19-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r Requirements.txt

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["sh", "-c", "uvicorn src.fastapi.app.main:app --app-dir /app --host 0.0.0.0 --port $PORT"]
