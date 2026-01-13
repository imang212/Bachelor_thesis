# Detection pipeline guide

## Setup, installing required modules

#### Install required modules
```bash
# Required modules
sudo apt-get install -y rsync ffmpeg x11-utils python3-dev python3-pip python3-setuptools python3-virtualenv python-gi-dev libgirepository1.0-dev gcc-12 g++-12 cmake git libzmq3-dev pkg-config

# GStreamer
sudo apt-get install -y libcairo2-dev libgirepository1.0-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio gcc-12 g++-12 python-gi-dev

sudo apt-get install -y gstreamer1.0-*

#PyGobject
sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-3.0
```

#### PciE driver installation
```bash
wget https://path-to-hailo-package/hailort-pcie-driver_X.Y.Z_amd64.deb

sudo dpkg -i hailort-pcie-driver_X.Y.Z_amd64.deb

sudo apt-get install -f

sudo reboot

# Check driver is loaded
lsmod | grep hailo

```

#### Tappas installation
```bash
git clone https://github.com/hailo-ai/tappas
#or
unzip tappas_VERSION_linux_installer.zip

# get to directory
cd /tappas/
```

```bash
mkdir hailort
git clone https://github.com/hailo-ai/hailort.git hailort/sources
```

```bash
#cache clear
rm -rf ~/.cache/gstreamer-1.0/
rm /usr/lib/$(uname -m)-linux-gnu/gstreamer-1.0/libgsthailotools.so

#installation
sudo ./install.sh --skip-hailort --target-platform rockchip

# find the file
find /usr -name "libgsthailometa.so*"

# bad file mapping path repair 
sudo ln -s /usr/lib/aarch64-linux-gnu/libgsthailometa.so.5 /usr/lib/aarch64-linux-gnu/libgsthailometa.so.3

# Check drivers are loaded
lsmod | grep hailo
# check packages
pkg-config --list-all | grep hailo
```
#### rpi5-examples repository install and Detection test 
```bash
git clone https://github.com/hailo-ai/hailo-rpi5-examples.git
cd hailo-rpi5-examples
./install.sh
source setup_env.sh
python basic_pipelines/detection_simple.py
```

### Commands (DetectionWithGStreamer_pipeline.py script)
```bash
# Analyze video file with only statistics
python3 DetectionWithGStreamer_pipeline.py --video input.mp4

# Analyze and record output
python3 DetectionWithGStreamer_pipeline.py --video input.mp4 --enable-recording --output analyzed.mp4

python3 AI_traffic_detection_hailo/DetectionWithGStreamer_pipeline.py --video outputx264-202601091302.mp4 --enable-recording --output analyzed.mp4 --enable-mqtt 
python3 AI_traffic_detection_hailo/DetectionWithGStreamer_pipeline.py --video outputc264-202601121303.mp4 --enable-recording --output analyzed_outputc264-202601121303.mp4

# Live IP camera with recording
python3 DetectionWithGStreamer_pipeline.py --enable-recording --output live_recording.mp4

### Full features enabled
python3 DetectionWithGStreamer_pipeline.py --video test.mp4 --enable-recording --enable-mqtt

# From IP Camera with rtsp pipeline detection streaming and mqtt sending 
python3 AI_traffic_detection_hailo/DetectionWithGStreamer_pipeline.py --enable-rtsp --enable-mqtt


```