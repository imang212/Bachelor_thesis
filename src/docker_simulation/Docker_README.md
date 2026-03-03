# 🚦 AI Traffic Detection — Docker System Simulation
**Raspberry Pi 5 · Hailo AI HAT+ 26 TOPS · Debian 12 Bookworm**

> Full Docker simulation of your production system.  
> Based on your actual architecture diagram, `MQTTClient.py`, `autorun.sh`,  
> and `auto_setup_repair_hailo_rpi5.sh`.

---

## Architecture (matches architecture.png)

```
Network Interface                 Raspberry Pi (Edge)         External MQTT Server
┌──────────────────┐             ┌──────────────────────┐    ┌────────────────────┐
│  camera_simulator│──RTSP──────▶│     ai_pipeline      │───▶│  mqtt.portabo.cz   │
│  :8554           │             │  (Python+GStreamer    │    │  :8883  TLS        │
│  (sim Hikvision) │             │   +Hailo/Sim detect)  │    └────────┬───────────┘
└──────────────────┘             └──────────────────────┘             │ subscribe
                                                                       ▼
Backend                                                      ┌─────────────────────┐
┌──────────────────────────────────────────────────────────┐ │   database_mqtt     │
│  ┌──────────────┐    reading    ┌─────────────────────┐  │ │(MQTT subscriber+DB) │
│  │   FastAPI    │◀─────────────│     PostgreSQL       │◀─┤ └─────────────────────┘
│  │   :8000      │              │     :5432            │  │         saving
│  │  /ws/ai ◀───│──────────────│  hailo_detections    │  │
│  └──────┬───────┘              └─────────────────────┘  │
│         │ POST/GET                                        │
└─────────┼──────────────────────────────────────────────┘
          ▼
Frontend
┌─────────────────────────────┐
│  Streamlit Dashboard :8501  │
│  (via nginx :80)            │
└─────────────────────────────┘
```

| Container        | IP           | Port(s)     | Mirrors                              |
|------------------|--------------|-------------|--------------------------------------|
| `camera_simulator`| 172.20.0.10 | 8554 RTSP   | Hikvision IP cam (rtsp://192.168.37.99) |
| `postgres`       | 172.20.0.20  | 5432        | PostgreSQL on Pi                     |
| `database_mqtt`  | 172.20.0.30  | —           | `DatabaseManagerPostgre.py`          |
| `ai_pipeline`    | 172.20.0.40  | —           | `DetectionWithGStreamer_pipeline.py --enable-rtsp --enable-mqtt` |
| `api`            | 172.20.0.50  | 8000        | `API_with_streamlit.py` FastAPI part |
| `dashboard`      | 172.20.0.60  | 8501        | `API_with_streamlit.py` Streamlit part |
| `nginx`          | 172.20.0.70  | 80          | Tailscale funnel replacement         |

> **MQTT broker is external** — `mqtt.portabo.cz:8883` (TLS).  
> No local broker. Both `ai_pipeline` (publisher) and `database_mqtt` (subscriber)  
> connect to the same external broker, exactly like your `MQTTClient.py`.

---

## Quick Start

### 1. Copy your pipeline scripts
```bash
cp DetectionWithGStreamer_pipeline.py pipeline/
cp DatabaseManagerPostgre.py          pipeline/
cp MQTTClient.py                      pipeline/
cp API_with_streamlit.py              pipeline/
```

### 2. Configure (optional — defaults match your scripts)
```bash
# .env already contains your MQTT credentials from MQTTClient.py
# Edit only if needed:
nano .env
```

### 3. Build and launch
```bash
docker compose up --build
```

### 4. Access
| URL                              | Service                     |
|----------------------------------|-----------------------------|
| `http://localhost`               | Streamlit dashboard (nginx) |
| `http://localhost/docs`          | FastAPI Swagger UI          |
| `http://localhost/health`        | Health check                |
| `http://localhost/test-ws`       | WebSocket test page         |
| `rtsp://localhost:8554/live/cam0`| Simulated IP camera stream  |

### 5. Verify the RTSP stream
```bash
# With VLC:
vlc rtsp://localhost:8554/live/cam0

# With ffplay:
ffplay rtsp://localhost:8554/live/cam0

# With gst-launch (matches your test commands):
gst-launch-1.0 rtspsrc location="rtsp://localhost:8554/live/cam0" latency=0 \
  ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink
```

### 6. Monitor MQTT traffic
```bash
# From your host (requires mosquitto-clients):
mosquitto_sub -h mqtt.portabo.cz -p 8883 \
  -u videoanalyza -P phdA9ZNW1vfkXdJkhhbP \
  --capath /etc/ssl/certs \
  -t 'patrik/traffic_detection' -v

# Or watch the ai_pipeline container logs:
docker compose logs -f ai_pipeline
```

### 7. Query the database
```bash
docker exec -it postgres psql -U hailo -d hailo_detections -c \
  "SELECT class_name, COUNT(DISTINCT track_id) AS vehicles, COUNT(*) AS detections \
   FROM detections GROUP BY 1 ORDER BY 2 DESC;"
```

---

## Switching to Real Hailo Hardware

1. Run `auto_setup_repair_hailo_rpi5.sh` on the host Pi to install drivers
2. Edit `docker-compose.yml` → `ai_pipeline` service:

```yaml
services:
  ai_pipeline:
    # Uncomment:
    devices:
      - /dev/hailo0:/dev/hailo0
    privileged: true
    environment:
      SIMULATION_MODE: "false"
      MODEL_PATH: "/models/yolov8m.hef"
      RTSP_URL: "rtsp://admin:Dcuk.123456@192.168.37.99/Stream"
```

3. Place your `.hef` model file:
```bash
docker volume inspect docker-sim2_hailo_models
# Copy model to volume mount path
```

The `pipeline_runner.py` then imports your real:
```python
from DetectionWithGStreamer_pipeline import HailoDetectionPipeline
```

---

## Replacing Camera Simulator with Real IP Camera

Simply change the `RTSP_URL` in `ai_pipeline` and remove `camera_simulator` dependency:

```yaml
ai_pipeline:
  environment:
    RTSP_URL: "rtsp://admin:Dcuk.123456@192.168.37.99/Stream"
  # Remove: depends_on camera_simulator
```

---

## Systemd Services (real Pi — managed by autorun.sh)

The Docker simulation mirrors these services:

| systemd service          | Docker container | Script                              |
|--------------------------|------------------|-------------------------------------|
| `detection_pipeline`     | `ai_pipeline`    | `DetectionWithGStreamer_pipeline.py` |
| `database_mqtt`          | `database_mqtt`  | `DatabaseManagerPostgre.py`         |
| `api`                    | `api` + `dashboard`| `API_with_streamlit.py`           |
| `dashboard_funnel`       | `nginx`          | Tailscale funnel replacement        |

---

## Useful Commands

```bash
# Full logs
docker compose logs -f

# Single service
docker compose logs -f ai_pipeline
docker compose logs -f database_mqtt

# Restart single service
docker compose restart ai_pipeline

# Stop all
docker compose down

# Stop + wipe DB volume
docker compose down -v

# Rebuild single
docker compose up --build ai_pipeline
```

---

## File Structure

```
docker-sim/
├── .env                          ← secrets (MQTT creds, DB password)
├── docker-compose.yml
├── pipeline/                     ← put your .py scripts here
│   └── README.md
├── camera_simulator/
│   ├── Dockerfile
│   ├── mediamtx.yml
│   └── entrypoint.sh
├── ai_pipeline/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── pipeline_runner.py        ← orchestrator
│   └── sim_detector.py           ← realistic traffic sim
├── database_mqtt/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── db_mqtt_service.py        ← MQTT subscriber + DB writer
├── postgres/
│   └── init_hailo.sql
├── api/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py                   ← FastAPI + WebSocket /ws/ai
├── dashboard/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── dashboard.py              ← Streamlit live dashboard
│   └── .streamlit/config.toml
└── nginx/
    └── nginx.conf
```
