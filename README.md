[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MQTT](https://img.shields.io/badge/MQTT-3B82F6?style=flat&logo=mqtt&logoColor=white)](https://mqtt.org/)
[![PostgreSQL](https://img.shields.io/badge/postgres-15+-blue.svg)](https://www.postgresql.org/)
[![Edge Computing](https://img.shields.io/badge/Platform-Raspberry%20Pi-red)](https://www.raspberrypi.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Status](https://img.shields.io/badge/Status-In%20Development-orange)

# Traffic Monitoring and Analysis System
**Automated real-time traffic analysis using Computer Vision and Edge Computing.**

This repository contains the source code for my Bachelor's thesis. The system handles the entire data pipeline—from raw RTSP streams at the edge to a centralized dashboard for urban planning insights.

### Key features
- **Real-time Detection & Tracking:** Leverages Computer Vision to identify and follow vehicles in video streams.
- **Edge Processing:** Optimized to run on Raspberry Pi using GStreamer and Python.
- **Data Pipeline:** Reliable data transfer from Edge devices to a central server via MQTT.
- **Interactive Dashboard:** Visualize traffic flow, peak hours, and vehicle counts in real-time.

### System Architecture
The system is divided into an Edge processing unit (AI Module) and a central management layer.

<img width="1916" height="882" alt="architecture" src="https://github.com/user-attachments/assets/dfa4d98e-ba95-4ed1-a2a5-d6e2b99d04f2" />

### Visuals
**Detection Pipeline:**

<img width="752" height="558" alt="own_tracking" src="https://github.com/user-attachments/assets/12723d3f-3b10-4da5-9a65-fea0e5000194"/>


**Analytics Dashboard:**

<img width="100%" alt="Dashboard" src="https://github.com/user-attachments/assets/ec60489e-6bc8-4597-bac3-bda0a974deea" />

### Tech Stack
- **Edge Hardware:** Raspberry Pi
- **Language:** Python
- **Vision:** GStreamer
- **Communication:** MQTT
- **Database:** PostgreSQL
- **BackEnd:** FastApi
- **Visualization:** Streamlit
