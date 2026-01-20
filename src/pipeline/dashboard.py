import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="Detection Analytics Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://192.168.37.205:8000"  # Change to your API URL

# Helper functions
@st.cache_data(ttl=30)
def fetch_health():
    """Fetch health status from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/health/ai", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error fetching health data: {e}")
        return None

@st.cache_data(ttl=10)
def fetch_detections(minutes=60, limit=1000):
    """Fetch recent detections from API"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/detections",
            params={"minutes": minutes, "limit": limit},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            if data:
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching detections: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=30)
def fetch_statistics(hours=24):
    """Fetch statistics from API"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/statistics",
            params={"hours": hours},
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception as e:
        st.error(f"Error fetching statistics: {e}")
        return {}

# Header
st.title("🎯 Detection Analytics Dashboard")
st.markdown("---")

# Helper functions for management
@st.cache_data(ttl=5)
def fetch_db_status():
    """Fetch database status"""
    try:
        response = requests.get(f"{API_BASE_URL}/database/status", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None

@st.cache_data(ttl=5)
def fetch_pipeline_status():
    """Fetch pipeline status"""
    try:
        response = requests.get(f"{API_BASE_URL}/pipeline/status", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None

def start_database():
    """Start database"""
    try:
        response = requests.post(f"{API_BASE_URL}/database/start", timeout=10)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def stop_database():
    """Stop database"""
    try:
        response = requests.post(f"{API_BASE_URL}/database/stop", timeout=10)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def start_pipeline(config):
    """Start pipeline with config"""
    try:
        response = requests.post(f"{API_BASE_URL}/pipeline/start", json=config, timeout=10)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def stop_pipeline():
    """Stop pipeline"""
    try:
        response = requests.post(f"{API_BASE_URL}/pipeline/stop", timeout=10)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

# Sidebar
with st.sidebar:
    st.header("⚙️ System Control")
    
    # Database Status & Control
    st.subheader("🗄️ Database")
    db_status = fetch_db_status()
    
    if db_status and db_status.get('connected'):
        st.success(f"✅ {db_status['status'].upper()}")
        if db_status.get('health'):
            st.metric("Total Detections", f"{db_status['health']['total_detections']:,}")
            st.metric("Database Size", db_status['health']['database_size'])
        
        if st.button("🛑 Stop Database", use_container_width=True):
            success, result = stop_database()
            if success:
                st.success("Database stopped")
                st.cache_data.clear()
                time.sleep(1)
                st.rerun()
            else:
                st.error(f"Failed: {result}")
    else:
        st.error("❌ Not Connected")
        if st.button("▶️ Start Database", use_container_width=True):
            with st.spinner("Starting database..."):
                success, result = start_database()
                if success:
                    st.success("Database started!")
                    st.cache_data.clear()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Failed: {result}")
    
    st.markdown("---")
    
    # Pipeline Status & Control
    st.subheader("🎯 AI Pipeline")
    pipeline_status = fetch_pipeline_status()
    
    if pipeline_status and pipeline_status.get('running'):
        st.success(f"✅ Running ({pipeline_status.get('mode', 'unknown')})")
        if pipeline_status.get('fps'):
            st.metric("FPS", f"{pipeline_status['fps']:.1f}")
        if pipeline_status.get('total_tracked'):
            st.metric("Tracked Objects", pipeline_status['total_tracked'])
        
        if st.button("🛑 Stop Pipeline", use_container_width=True):
            with st.spinner("Stopping pipeline..."):
                success, result = stop_pipeline()
                if success:
                    st.success("Pipeline stopped")
                    st.cache_data.clear()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Failed: {result}")
    else:
        st.warning("⏸️ Not Running")
        
        # Pipeline configuration
        with st.expander("▶️ Start Pipeline", expanded=False):
            mode = st.radio(
                "Source Mode",
                ["RTSP Stream", "Pi Camera", "Video File"],
                index=0
            )
            
            video_file = None
            if mode == "Video File":
                video_file = st.text_input("Video Path", "/path/to/video.mp4")
            
            col1, col2 = st.columns(2)
            with col1:
                enable_mqtt = st.checkbox("MQTT", value=True)
                enable_rtsp = st.checkbox("RTSP", value=False)
                enable_websocket = st.checkbox("WebSocket", value=False)
            with col2:
                enable_webrtc = st.checkbox("WebRTC", value=False)
                enable_recording = st.checkbox("Recording", value=False)
                enable_debug = st.checkbox("Debug", value=False)
            
            if st.button("🚀 Start", use_container_width=True):
                config = {
                    "rpicamera": mode == "Pi Camera",
                    "video_file": video_file if mode == "Video File" else None,
                    "enable_mqtt": enable_mqtt,
                    "enable_rtsp": enable_rtsp,
                    "enable_websocket": enable_websocket,
                    "enable_webrtc": enable_webrtc,
                    "enable_recording": enable_recording,
                    "enable_debug": enable_debug
                }
                
                with st.spinner("Starting pipeline..."):
                    success, result = start_pipeline(config)
                    if success:
                        st.success("Pipeline started!")
                        st.cache_data.clear()
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(f"Failed: {result.get('detail', result)}")
    
    st.markdown("---")
    
    # API connection status
    st.subheader("🌐 API Status")
    health = fetch_health()
    if health:
        st.success("✅ Connected")
    else:
        st.error("❌ Disconnected")
    
    st.markdown("---")
    
    # Time range selector
    st.subheader("📅 Time Range")
    time_range = st.selectbox(
        "Select time range",
        options=[5, 15, 30, 60, 180, 360, 720, 1440],
        format_func=lambda x: f"Last {x} minutes" if x < 60 else f"Last {x//60} hours",
        index=3  # Default to 60 minutes
    )
    
    stats_hours = st.selectbox(
        "Statistics window",
        options=[1, 6, 12, 24, 48, 168],
        format_func=lambda x: f"Last {x} hours" if x < 24 else f"Last {x//24} days",
        index=3  # Default to 24 hours
    )
    
    st.markdown("---")
    
    # Refresh settings
    st.subheader("🔄 Auto Refresh")
    auto_refresh = st.checkbox("Enable auto-refresh", value=True)
    refresh_interval = st.slider("Refresh interval (seconds)", 5, 60, 10)
    
    if st.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()

# Main content
# Fetch data
df_detections = fetch_detections(minutes=time_range, limit=2000)
stats = fetch_statistics(hours=stats_hours)

if df_detections.empty:
    st.warning("No detection data available for the selected time range.")
else:
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_detections = len(df_detections)
        st.metric("Total Detections", f"{total_detections:,}")
    
    with col2:
        unique_classes = df_detections['class_name'].nunique()
        st.metric("Unique Classes", unique_classes)
    
    with col3:
        if 'track_id' in df_detections.columns:
            unique_tracks = df_detections['track_id'].dropna().nunique()
            st.metric("Unique Tracks", unique_tracks)
        else:
            st.metric("Unique Tracks", "N/A")
    
    with col4:
        avg_confidence = df_detections['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
    
    st.markdown("---")
    
    # Row 1: Detection timeline and class distribution
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Detection Timeline")
        
        # Group by time intervals
        df_timeline = df_detections.copy()
        df_timeline['time_bucket'] = df_timeline['timestamp'].dt.floor('1min')
        timeline_data = df_timeline.groupby(['time_bucket', 'class_name']).size().reset_index(name='count')
        
        fig_timeline = px.line(
            timeline_data,
            x='time_bucket',
            y='count',
            color='class_name',
            title='Detections Over Time',
            labels={'time_bucket': 'Time', 'count': 'Detection Count', 'class_name': 'Class'}
        )
        fig_timeline.update_layout(
            height=400,
            hovermode='x unified',
            showlegend=True
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Class Distribution")
        
        class_counts = df_detections['class_name'].value_counts()
        
        fig_pie = px.pie(
            values=class_counts.values,
            names=class_counts.index,
            title='Detection by Class',
            hole=0.4
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    # Row 2: Confidence distribution and heatmap
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Confidence Distribution")
        
        fig_conf = px.box(
            df_detections,
            x='class_name',
            y='confidence',
            color='class_name',
            title='Confidence by Class',
            labels={'class_name': 'Class', 'confidence': 'Confidence'}
        )
        fig_conf.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_conf, use_container_width=True)
    
    with col2:
        st.subheader("🔥 Detection Heatmap")
        
        # Create hourly heatmap
        df_heatmap = df_detections.copy()
        df_heatmap['hour'] = df_heatmap['timestamp'].dt.hour
        df_heatmap['day'] = df_heatmap['timestamp'].dt.day_name()
        
        heatmap_data = df_heatmap.groupby(['day', 'hour']).size().reset_index(name='count')
        heatmap_pivot = heatmap_data.pivot(index='day', columns='hour', values='count').fillna(0)
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_pivot = heatmap_pivot.reindex([d for d in day_order if d in heatmap_pivot.index])
        
        fig_heatmap = px.imshow(
            heatmap_pivot,
            labels=dict(x="Hour of Day", y="Day", color="Detections"),
            x=heatmap_pivot.columns,
            y=heatmap_pivot.index,
            color_continuous_scale='YlOrRd',
            aspect='auto'
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.markdown("---")
    
    # Row 3: Statistics table and tracking analysis
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Statistics Summary")
        
        if stats:
            stats_data = []
            for class_name, data in stats.items():
                stats_data.append({
                    'Class': class_name,
                    'Total Detections': data['total_detections'],
                    'Unique Tracks': data['unique_tracks'],
                    'Avg Confidence': f"{data['avg_confidence']:.2%}",
                    'Min Confidence': f"{data['min_confidence']:.2%}",
                    'Max Confidence': f"{data['max_confidence']:.2%}"
                })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        else:
            st.info("No statistics available for the selected time range.")
    
    with col2:
        st.subheader("🎯 Top Tracks")
        
        if 'track_id' in df_detections.columns:
            track_counts = df_detections[df_detections['track_id'].notna()].groupby('track_id').agg({
                'class_name': 'first',
                'confidence': 'mean',
                'timestamp': 'count'
            }).rename(columns={'timestamp': 'appearances'}).sort_values('appearances', ascending=False).head(10)
            
            track_counts['confidence'] = track_counts['confidence'].apply(lambda x: f"{x:.2%}")
            track_counts = track_counts.reset_index()
            track_counts.columns = ['Track ID', 'Class', 'Avg Confidence', 'Appearances']
            
            st.dataframe(track_counts, use_container_width=True, hide_index=True)
        else:
            st.info("Track ID data not available.")
    
    st.markdown("---")
    
    # Row 4: Bounding box visualization
    st.subheader("📦 Detection Spatial Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Calculate bbox centers
        df_bbox = df_detections.copy()
        df_bbox['center_x'] = (df_bbox['bbox'].apply(lambda x: x[0]) + df_bbox['bbox'].apply(lambda x: x[2])) / 2
        df_bbox['center_y'] = (df_bbox['bbox'].apply(lambda x: x[1]) + df_bbox['bbox'].apply(lambda x: x[3])) / 2
        
        fig_scatter = px.scatter(
            df_bbox,
            x='center_x',
            y='center_y',
            color='class_name',
            size='confidence',
            title='Detection Positions (Bbox Centers)',
            labels={'center_x': 'X Position', 'center_y': 'Y Position', 'class_name': 'Class'},
            opacity=0.6
        )
        fig_scatter.update_yaxes(autorange="reversed")
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Bbox size distribution
        df_bbox['bbox_width'] = df_bbox['bbox'].apply(lambda x: x[2] - x[0])
        df_bbox['bbox_height'] = df_bbox['bbox'].apply(lambda x: x[3] - x[1])
        df_bbox['bbox_area'] = df_bbox['bbox_width'] * df_bbox['bbox_height']
        
        fig_size = px.histogram(
            df_bbox,
            x='bbox_area',
            color='class_name',
            title='Bounding Box Size Distribution',
            labels={'bbox_area': 'Bbox Area (pixels²)', 'class_name': 'Class'},
            nbins=30
        )
        fig_size.update_layout(height=400)
        st.plotly_chart(fig_size, use_container_width=True)
    
    st.markdown("---")
    
    # Row 5: Recent detections table
    st.subheader("📜 Recent Detections")
    
    # Format dataframe for display
    df_display = df_detections.copy()
    df_display['timestamp'] = df_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df_display['confidence'] = df_display['confidence'].apply(lambda x: f"{x:.2%}")
    df_display['bbox'] = df_display['bbox'].apply(lambda x: f"[{x[0]:.0f}, {x[1]:.0f}, {x[2]:.0f}, {x[3]:.0f}]")
    
    # Select columns to display
    display_cols = ['timestamp', 'frame_id', 'class_name', 'confidence', 'track_id', 'bbox']
    df_display = df_display[display_cols].sort_values('timestamp', ascending=False).head(100)
    
    st.dataframe(df_display, use_container_width=True, hide_index=True, height=400)

# Auto-refresh logic
if auto_refresh:
    time.sleep(refresh_interval)
    st.cache_data.clear()
    st.rerun()