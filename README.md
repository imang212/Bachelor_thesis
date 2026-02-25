# System for Traffic Monitoring and Analysis using Computer Vision (Bachelor_thesis)
This repository contains the source code and documentation for my Bachelor's thesis project: an automated system designed for real-time traffic monitoring and analysis. By leveraging Computer Vision and Edge Computing, the system detects, tracks, and analyzes vehicular flow to provide actionable insights for urban planning and traffic management.

The architecture is built to handle the full data pipeline - from capturing raw RTSP video streams at the edge to processing and visualizing data on a central dashboard.

**Detection from pipeline:**

<img width="752" height="558" alt="own_tracking" src="https://github.com/user-attachments/assets/12723d3f-3b10-4da5-9a65-fea0e5000194" />

### System architecture diagram (with data flow)
```mermaid
graph LR
    subgraph "Raspberry Pi (Edge)"
        AI["AI Module</br> (Python + GStreamer)"]
    end
    
    subgraph "Network interface"
        Camera["IP camera"]
    end
    
    subgraph "MQTT Server"
        MQTT[MQTT Broker]
    end

    subgraph "Backend"
        DB[(Database)]
        DBListener["DB Client (subscriber)"]
        FastAPI[FastAPI Server]
    end

    subgraph "Frontend"
        Streamlit[Streamlit Dashboard]
    end
    
    Camera -->|RTSP stream| AI
    AI -->|sending| MQTT
    DBListener --> |listen| MQTT
    DBListener -->|saving| DB
    DB -->|reading| DBListener
    FastAPI -->|reading| DB
    Streamlit -->|POST/GET| FastAPI
    
    style AI fill:#90EE90
    style Camera fill:#87CEEB
    style MQTT fill:#FFB6C1
    style DB fill:#DDA0DD
    style FastAPI fill:#F0E68C
    style Streamlit fill:#FFA07A
```
