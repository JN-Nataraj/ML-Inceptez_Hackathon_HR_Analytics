FROM python:3.10.19-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r Requirements.txt

EXPOSE 8080

CMD ["uvicorn", "src.fastapi.app.main:app", "--host", "0.0.0.0", "--port", "8080"]