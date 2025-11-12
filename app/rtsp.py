"""
Camera Stream Capture Module

Handles RTSP/USB camera capture using OpenCV.
For Jetson production, use GStreamer pipeline for NVDEC hardware acceleration.
"""

import cv2
import numpy as np
from typing import Optional, Generator, Tuple
import os


def build_gstreamer_pipeline(rtsp_url: str, width: int, height: int, fps: int = 30) -> str:
    """
    Build GStreamer pipeline for Jetson NVDEC hardware decoding.
    
    Args:
        rtsp_url: RTSP stream URL
        width: Frame width
        height: Frame height
        fps: Frame rate
        
    Returns:
        GStreamer pipeline string
    """
    # GStreamer pipeline for RTSP with NVDEC on Jetson
    pipeline = (
        f"rtspsrc location={rtsp_url} latency=0 ! "
        "rtph264depay ! "
        "h264parse ! "
        "nvv4l2decoder ! "
        "nvvidconv ! "
        f"video/x-raw, width={width}, height={height}, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink"
    )
    return pipeline


def open_stream(rtsp_url: str, width: int, height: int, fps_cap: int = 30, use_gstreamer: bool = False) -> Optional[cv2.VideoCapture]:
    """
    Open camera stream (RTSP or USB).
    
    Args:
        rtsp_url: RTSP URL or device index (e.g., "0" for USB camera)
        width: Expected frame width
        height: Expected frame height
        fps_cap: Frame rate cap
        use_gstreamer: Use GStreamer pipeline (recommended for Jetson)
        
    Returns:
        OpenCV VideoCapture object or None if failed
    """
    if use_gstreamer and rtsp_url.startswith('rtsp://'):
        # Use GStreamer pipeline for RTSP on Jetson
        pipeline = build_gstreamer_pipeline(rtsp_url, width, height, fps_cap)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    elif rtsp_url.isdigit() or rtsp_url == '0':
        # USB camera
        device_idx = int(rtsp_url)
        cap = cv2.VideoCapture(device_idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps_cap)
    else:
        # RTSP with OpenCV
        cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print(f"Failed to open stream: {rtsp_url}")
        return None
    
    return cap


def frame_generator(cap: cv2.VideoCapture) -> Generator[Tuple[bool, Optional[np.ndarray]], None, None]:
    """
    Generator that yields frames from a VideoCapture.
    
    Args:
        cap: OpenCV VideoCapture object
        
    Yields:
        Tuple of (success, frame) where frame is BGR image or None
    """
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield ret, frame
    
    cap.release()


def test_stream(rtsp_url: str, width: int, height: int, use_gstreamer: bool = False) -> bool:
    """
    Test if a stream can be opened and read.
    
    Args:
        rtsp_url: RTSP URL or device index
        width: Expected width
        height: Expected height
        use_gstreamer: Use GStreamer pipeline
        
    Returns:
        True if stream is accessible, False otherwise
    """
    cap = open_stream(rtsp_url, width, height, use_gstreamer=use_gstreamer)
    if cap is None:
        return False
    
    ret, frame = cap.read()
    cap.release()
    return ret and frame is not None


