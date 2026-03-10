FROM python:3.10-slim-bullseye

# System dependencies for dlib (required by face_recognition)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY templates/ templates/
COPY static/ static/
COPY known_faces/ known_faces/
COPY known_faces_encodings.pkl .

EXPOSE 10000

CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--timeout", "120", "--workers", "1", "server:app"]
