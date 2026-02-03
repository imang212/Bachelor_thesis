# Detection pipeline guide
## Setup, installing required modules
#### Install required modules
```bash
sudo apt-get update && apt-get install -y 
    python3 \ python3-venv \ python3-pip \ python3-dev \ python3-setuptools \ python3-virtualenv \
    build-essential \ cmake \ git \ wget \ pkg-config \
    libopenblas-dev \ libopencv-dev \ python3-opencv \ libatlas-base-dev \ gfortran \
    libjpeg-dev \ libpng-dev \ libavcodec-dev \ libavformat-dev \ libswscale-dev \ 
    libv4l-dev \ libxvidcore-dev \ libx264-dev \ libhdf5-dev \ libhdf5-serial-dev \
    libcap-dev \ libarchive-dev \ libavdevice-dev \ libavutil-dev \ libswresample-dev \
    libfreetype6 \ libcamera0 \ rsync \ ffmpeg \ x11-utils \
    python-gi-dev \ libgirepository1.0-dev \ gcc-12 \ g++-12 \ libzmq3-dev \ pkg-config \ libcairo2-dev \ libgirepository1.0-dev \ libgstreamer1.0-dev \ libgstreamer-plugins-base1.0-dev \ libgstreamer-plugins-bad1.0-dev \ gstreamer1.0-plugins-base \ gstreamer1.0-plugins-good \ gstreamer1.0-plugins-bad \ gstreamer1.0-plugins-ugly \ gstreamer1.0-libav \ gstreamer1.0-tools \ gstreamer1.0-x \  gstreamer1.0-alsa \ gstreamer1.0-gl \ gstreamer1.0-gtk3 \ gstreamer1.0-qt5 \ gstreamer1.0-pulseaudio \ gcc-12 \ g++-12 \ python-gi-dev \ gstreamer1.0-* \ python3-gi \ python3-gi-cairo \ gir1.2-gtk-3.0

# Required modules
sudo apt-get install -y rsync ffmpeg x11-utils python3-dev python3-pip python3-setuptools python3-virtualenv python-gi-dev libgirepository1.0-dev gcc-12 g++-12 cmake git libzmq3-dev pkg-config

# GStreamer
sudo apt-get install -y libcairo2-dev libgirepository1.0-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio gcc-12 g++-12 python-gi-dev

sudo apt-get install -y gstreamer1.0-*

#PyGobject
sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-3.0

# Install RTSP plugins
sudo apt-get install -y gir1.2-gst-rtsp-server-1.0 gir1.2-gstreamer-1.0

# Install mqtt plugin
sudo apt install -y python3-paho-mqtt python3-websockets
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
sudo rm -rf /root/.cache/gstreamer-1.0
sudo rm /usr/lib/$(uname -m)-linux-gnu/gstreamer-1.0/libgsthailotools.so

#installation
sudo ./install.sh --skip-hailort

# find the file
find /usr -name "libgsthailometa.so*"

# bad file mapping path repair 
sudo ln -s /usr/lib/aarch64-linux-gnu/libgsthailometa.so.5.2.0 /usr/lib/aarch64-linux-gnu/libgsthailometa.so.3

unset $GST_PLUGIN_PATH
unset $LD_LIBRARY_PATH

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

python3 DetectionWithGStreamer_pipeline.py --enable-recording --output live_recording.mp4 --enable-mqtt

### Full features enabled
python3 DetectionWithGStreamer_pipeline.py --video test.mp4 --enable-recording --enable-mqtt

# From IP Camera with rtsp pipeline detection streaming and mqtt sending 
python3 AI_traffic_detection_hailo/DetectionWithGStreamer_pipeline.py --enable-rtsp --enable-mqtt


```
```bash

GStreamer console testing commands:
gst-launch-1.0 \
  libcamerasrc ! \
  video/x-raw,format=NV12,width=1536,height=864,framerate=30/1 ! \
  videoconvert ! \
  videoscale ! \
  video/x-raw,format=RGB, width=640,height=640 ! \
  hailonet hef-path=/home/imang/hailo-rpi5-examples/resources/models/hailo8/yolov8m.hef name=hailonet ! \
  hailofilter function-name=yolov8 \
    config-path=/home/imang/hailo_model_zoo/hailo_model_zoo/cfg/postprocess_config/yolov8m_nms_config.json \
    name=hailofilter ! \
  hailotracker kalman-dist-thr=0.7 iou-thr=0.3 keep-tracked-frames=30 keep-new-frames=3 keep-lost-frames=10 name=hailotracker ! \
  hailooverlay show-confidence=true line-thickness=2 name=hailooverlay ! \
  textoverlay text='Hailo Tracker - YOLOv8m' valignment=top halignment=left font-desc='Sans 12' ! \
  videoconvert ! \
  video/x-raw,format=I420 ! \
  tee name=t \
    t. ! queue ! fakesink

Test with rtsp:
gst-launch-1.0 \
  libcamerasrc ! \
  video/x-raw,format=NV12,width=1536,height=864,framerate=30/1 ! \
  videoconvert ! \
  videoscale ! \
  video/x-raw,format=RGB, width=640, height=640 ! \
  hailonet hef-path=/home/imang/hailo-rpi5-examples/resources/models/hailo8/yolov8m.hef name=hailonet ! \
  hailofilter function-name=yolov8 \
    config-path=/home/imang/hailo_model_zoo/hailo_model_zoo/cfg/postprocess_config/yolov8m_nms_config.json \
    name=hailofilter ! \
  hailotracker kalman-dist-thr=0.7 iou-thr=0.3 keep-tracked-frames=30 keep-new-frames=3 keep-lost-frames=10 
  name=hailotracker ! \
  hailooverlay show-confidence=true line-thickness=2 name=hailooverlay ! \
  textoverlay text='Hailo Tracker - YOLOv8m' valignment=top halignment=left font-desc='Sans 12' ! \
  videoconvert ! \
  video/x-raw,format=I420 ! \
  tee name=t \
     t. ! queue ! videoconvert ! video/x-raw,format=I420,width=1536,height=864 ! appsink name=rtsp_sink emit-signals=true sync=false \
      t. ! queue ! fakesink

Small test:  
  gst-launch-1.0 \
  libcamerasrc ! video/x-raw,format=NV12,width=1536,height=864,framerate=30/1 ! \
  videoconvert ! video/x-raw ! videoscale ! video/x-raw,width=640,height=640 ! \
  hailonet hef-path=/home/imang/hailo-rpi5-examples/resources/models/hailo8/yolov8m.hef name=hailonet ! \
  hailofilter function-name=yolov8 \
    config-path=/home/imang/hailo_model_zoo/hailo_model_zoo/cfg/postprocess_config/yolov8m_nms_config.json \
    so-path=/home/imang/hailo-rpi5-examples/resources/so/libyolov8_postprocess.so \
    name=hailofilter ! \
  fakesink

Test with window:
gst-launch-1.0 \
  rtspsrc location="rtsp://admin:Dcuk.123456@192.168.37.99/Stream" latency=0 ! rtph264depay ! h264parse ! avdec_h264 ! \
  videoconvert ! \
  videoscale ! \
  video/x-raw,format=RGB,width=640,height=640 ! \
  hailonet hef-path=/home/imang/hailo-rpi5-examples/resources/models/hailo8/yolov8m.hef batch-size=1 ! \
  queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! \
  hailofilter function-name=yolov8m \
    so-path=/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so \
    qos=false ! \
  queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! \
  hailotracker kalman-dist-thr=0.7 iou-thr=0.3 keep-tracked-frames=30 keep-new-frames=3 keep-lost-frames=10 ! \
  hailooverlay show-confidence=true line-thickness=2 ! \
  textoverlay text='Hailo Tracker - YOLOv8m' valignment=top halignment=left font-desc='Sans 12' ! \
  videoconvert ! \
  autovideosink

GStreamer test
gst-launch-1.0 \
    rtspsrc location="rtsp://admin:Dcuk.123456@192.168.37.99/Stream" latency=0 ! \
    rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! \
    videoscale ! \
    video/x-raw,format=RGB, width=640,height=640 ! \
    hailonet hef-path=/home/imang/hailo-rpi5-examples/resources/models/hailo8/yolov8m.hef name=hailonet ! \
    hailofilter function-name=yolov8 \
        config-path=/home/imang/hailo_model_zoo/hailo_model_zoo/cfg/postprocess_config/yolov8m_nms_config.json \
        name=hailofilter ! \
    hailotracker kalman-dist-thr=0.7 iou-thr=0.3 keep-tracked-frames=30 keep-new-frames=3 keep-lost-frames=10 name=hailotracker ! \
    hailooverlay show-confidence=true line-thickness=2 name=hailooverlay ! \
    textoverlay text='Hailo Tracker - YOLOv8m' valignment=top halignment=left font-desc='Sans 12' ! \
    autovideosink !

Launch RSTP stream from camera:
gst-launch-1.0 -v rtspsrc location=rtsp://192.168.37.205:8554/hailo_stream latency=0 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink
Launch real stream from camera:
gst-launch-1.0 rtspsrc location="rtsp://admin:Dcuk.123456@192.168.37.99/Streaming/Channels" latency=0 !   rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink
"""
```
