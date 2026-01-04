FROM python:3.10.19-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r Requirements.txt

ENV PYTHONPATH=/app/src/fastapi
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT"]
