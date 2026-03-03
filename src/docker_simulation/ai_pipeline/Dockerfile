FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-gi \
    python3-dev \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-rtsp-server-1.0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    libgirepository1.0-dev \
    libcairo2-dev \
    build-essential \
    wget \
    curl \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://hailo-cdn.s3.amazonaws.com/software/hailort/4.23.0/hailort_4.23.0_arm64.deb \
    && dpkg -i hailort_4.23.0_arm64.deb || apt-get install -f -y \
    && rm hailort_4.23.0_arm64.deb

ENV GST_PLUGIN_PATH=""
ENV GST_PLUGIN_SYSTEM_PATH="/usr/lib/aarch64-linux-gnu/gstreamer-1.0"

WORKDIR /app
COPY DetectionWithGStreamer_pipeline.py MQTTClient.py WebSocket.py ./

RUN pip3 install --break-system-packages \
    paho-mqtt \
    websockets \
    shapely \
    requests \
    psycopg2-binary

CMD ["python3", "DetectionWithGStreamer_pipeline.py", "--enable-mqtt", "--debug"]