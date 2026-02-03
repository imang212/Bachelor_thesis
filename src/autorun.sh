#!/bin/bash
# Autorun script to setup AI Traffic Detection services on Raspberry Pi
# This script creates systemd service files and enables them
set -e  # Exit on any error
# Configuration
BASE_DIR="$HOME/AI_traffic_detection_hailo"
LOG_DIR="$BASE_DIR/logs"
USER="imang"

echo "AI Traffic Detection Service Setup"
# Check if base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "ERROR: Base directory $BASE_DIR does not exist!"
    exit 1
fi
# Create log directory
mkdir -p "$LOG_DIR"
echo "✓ Log directory created: $LOG_DIR"
# Check if Python scripts exist
SCRIPTS=("DetectionWithGStreamer_pipeline.py" "DatabaseManagerPostgre.py" "API_with_streamlit.py")
for script in "${SCRIPTS[@]}"; do
    if [ ! -f "$BASE_DIR/$script" ]; then
        echo "WARNING: $script not found in $BASE_DIR"
    else
        echo "✓ Found: $script"
    fi
done
echo ""
echo "Creating systemd service files..."
# Create detection_pipeline.service
sudo tee /etc/systemd/system/detection_pipeline.service > /dev/null <<EOF
[Unit]
Description=AI Traffic Detection Pipeline (Hailo)
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$BASE_DIR
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/python3 -u $BASE_DIR/DetectionWithGStreamer_pipeline.py --enable-mqtt
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
echo "✓ Created: detection_pipeline.service"

# Create database_mqtt.service
sudo tee /etc/systemd/system/database_mqtt.service > /dev/null <<EOF
[Unit]
Description=Database Manager PostgreSQL with MQTT
After=network.target postgresql.service
Wants=postgresql.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$BASE_DIR
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/python3 -u $BASE_DIR/DatabaseManagerPostgre.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
echo "✓ Created: database_mqtt.service"

# Create api.service
sudo tee /etc/systemd/system/api.service > /dev/null <<EOF
[Unit]
Description=API Dashboard (Streamlit + FastAPI)
After=network.target database_mqtt.service
Wants=database_mqtt.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$BASE_DIR
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/python3 -u $BASE_DIR/API_with_streamlit.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
echo "✓ Created: api.service"

# create reverse proxy service
sudo tee /etc/systemd/system/dashboard_funnel.service > /dev/null <<EOF
[Unit]
Description=Enable Tailscale Funnel
After=tailscaled.service
Requires=tailscaled.service

[Service]
Type=oneshot
ExecStart=/usr/bin/tailscale funnel 8501
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
echo "✓ Created: dashboard_funnel.service"

echo ""
echo "Reloading systemd daemon..."
sudo systemctl daemon-reload
echo "✓ Daemon reloaded"
echo ""
echo "Enabling services to start on boot..."
sudo systemctl enable detection_pipeline.service
sudo systemctl enable database_mqtt.service
sudo systemctl enable api.service
echo "✓ Services enabled"
echo ""
echo "Starting services..."
sudo systemctl start detection_pipeline.service
sudo systemctl start database_mqtt.service
sudo systemctl start api.service
echo "✓ Services started"
echo ""
echo "Service Status:"
sudo systemctl status detection_pipeline.service --no-pager -l
echo ""
sudo systemctl status database_mqtt.service --no-pager -l
echo ""
sudo systemctl status api.service --no-pager -l
echo ""
echo "Setup Complete!"
echo "Useful commands:"
echo "  View logs:"
echo "    journalctl -u detection_pipeline -f"
echo "    journalctl -u database_mqtt -f"
echo "    journalctl -u api -f"
echo "  Check status:"
echo "    sudo systemctl status detection_pipeline"
echo "    sudo systemctl status database_mqtt"
echo "    sudo systemctl status api"
echo "  Stop services:"
echo "    sudo systemctl stop detection_pipeline"
echo "    sudo systemctl stop database_mqtt"
echo "    sudo systemctl stop api"
echo "  Restart services:"
echo "    sudo systemctl restart detection_pipeline"
echo "    sudo systemctl restart database_mqtt"
echo "    sudo systemctl restart api"
echo "  Disable autostart:"
echo "    sudo systemctl disable detection_pipeline"
echo "    sudo systemctl disable database_mqtt"
echo "    sudo systemctl disable api"