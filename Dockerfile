FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# system deps required by Pillow / TF wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY . .

EXPOSE 8080

# Use the PORT env var provided by the host (Vercel sets $PORT at runtime)
CMD streamlit run main.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
