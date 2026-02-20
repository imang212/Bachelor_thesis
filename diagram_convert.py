import subprocess

# Váš upravený mermaid diagram
mermaid_code = '''graph LR
    subgraph "Raspberry Pi (Edge)"
        AI["Detection Module</br> (Python + GStreamer)"]
    end
    
    subgraph "Network interface"
        Camera["IP camera"]
    end
    
    subgraph "MQTT Server"
        MQTT["MQTT Broker</br> (mqtt.portabo.cz)"]
    end

    subgraph "Backend"
        DB[(Database)]
        DBListener["Data module<br> (MQTT subscriber)"]
        FastAPI[FastAPI Server]
    end

    subgraph "Frontend"
        Streamlit[Streamlit Dashboard]
    end
    
    Camera -->|RTSP stream| AI
    AI -->|sending| MQTT
    DBListener --> |listen| MQTT
    DBListener -->|saving| DB
    FastAPI -->|reading| DB
    Streamlit -->|POST/GET| FastAPI
    
    style AI fill:#90EE90
    style Camera fill:#87CEEB
    style MQTT fill:#FFB6C1
    style DB fill:#DDA0DD
    style FastAPI fill:#F0E68C
    style Streamlit fill:#FFA07A
'''

# Uložení mermaid kódu do souboru
with open('architecture.mmd', 'w', encoding='utf-8') as f:
    f.write(mermaid_code)

# Konverze do PNG
subprocess.run([
    'mmdc',
    '-i', 'architecture.mmd',
    '-o', 'architecture.png',
    '-b', 'white',
    '-w', '2400',
    '-H', '800',
    '--scale', '2'
], check=True)

print("Aktualizovaný diagram byl úspěšně vytvořen!")