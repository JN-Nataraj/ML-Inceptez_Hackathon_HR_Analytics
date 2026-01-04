FROM python:3.10.19-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r src/fastapi/Requirements.txt

ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --app-dir src/fastapi --host 0.0.0.0 --port $PORT"]
