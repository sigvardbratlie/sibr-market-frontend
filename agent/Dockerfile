FROM python:3.11-slim
LABEL authors="sigvardbratlie"

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python","main.py"]