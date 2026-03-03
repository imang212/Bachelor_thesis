# Simulace linux/arm64 pro Raspberry Pi 5 s Hailo-8
FROM python:3.11-slim-bookworm
# Nastavení proměnných prostředí
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Instalace základních závislostí
RUN apt-get update && apt-get install -y \
    python3.11 \ 
    python3.11-venv \ 
    python3.11-pip \
    build-essential \ 
    cmake \ 
    git \ 
    wget \
    libopenblas-dev \ 
    libopencv-dev \
    python3-opencv \
    libatlas-base-dev \ 
    gcc g++ gfortran \
    libjpeg-dev \ 
    libpng-dev \ 
    libavcodec-dev \ 
    libavformat-dev \
    libswscale-dev \ 
    libv4l-dev \ 
    libxvidcore-dev \ 
    libx264-dev \
    libhdf5-dev \ 
    libhdf5-serial-dev \
    libcap-dev \
    libarchive-dev \
    libavdevice-dev \ 
    libavutil-dev \
    libswresample-dev \
    libfreetype6 \
    libcamera0 \
    ffmpeg \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libgstrtspserver-1.0-dev \
    gir1.2-gstreamer-1.0 \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-rtsp-server-1.0 \
    libgirepository1.0-dev \
    libcairo2-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# Vytvoření pracovního adresáře
WORKDIR /app
# Kopírování requirements.txt do containeru
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Vytvoření virtuálního prostředí
RUN python3.11 -m venv /app/venv --system-site-packages
# Aktivace virtuálního prostředí a instalace základních balíčků
RUN . /app/venv/bin/activate && \
    pip install --upgrade pip setuptools wheel
    
# Instalace dalších požadovaných balíčků z requirements.txt
RUN . /app/venv/bin/activate && \
pip install --no-cache-dir -r requirements.txt
# clone hailo repositories
RUN git clone https://github.com/hailo-ai/hailo-rpi5-examples
RUN git clone https://github.com/hailo-ai/hailo_model_zoo
# Kopírování všech Python souborů z aktuálního adresáře
COPY . /app/
# Vytvoření testovacího skriptu
RUN echo '#!/bin/bash\n\
source /app/venv/bin/activate\n\
echo "Python verze:"\n\
python --version\n\
echo "Test YOLO11:"\n\
python -c "import fastapi; print(f\"FastAPI: {fastapi.__version__}\")"\n\
python -c "import uvicorn; print(f\"Uvicorn: {uvicorn.__version__}\")"\n\
python -c "import websockets; print(f\"Websockets: {websockets.__version__}\")"\n\
python -c "import cv2; print(f\"OpenCV: {cv2.__version__}\")"\n\
python -c "import numpy; print(f\"NumPy: {numpy.__version__}\")"\n\
python -c "import scipy; print(f\"SciPy: {scipy.__version__}\")"\n\
python -c "import filterpy; print(\"FilterPy: OK\")"\n\
python -c "import pydantic; print(f\"Pydantic: {pydantic.__version__}\")"\n\
' > /app/test_install.sh && chmod +x /app/test_install.sh
# Vytvoření skriptu pro spuštění bashu s aktivovaným venv
RUN echo '#!/bin/bash\n\
source /app/venv/bin/activate\n\
exec bash "$@"\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh
# Nastavení entrypoint
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/test_install.sh"]
# Výchozí příkaz pro spuštění testu scriptu yoloTrafficDetectionSystem.py
CMD ["python", "yoloTrafficDetectionSystem.py"]