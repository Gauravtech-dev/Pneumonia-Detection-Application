FROM python:3.10-slim

WORKDIR /app

# Python output immediately show kare
ENV PYTHONUNBUFFERED=1

# Dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Project files
COPY . .

# Render / local dono ke liye
EXPOSE 8000

# Render ka PORT use karega; local par 8000 default rahega
CMD ["sh", "-c", "python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]