# ai-focus-monitor
Real-time productivity and focus analysis system using YOLOv8 and OpenCV.

This project is a real-time computer vision application designed to analyze and track focus levels during study or work sessions. It provides data-driven insights into productivity by monitoring user behavior and environment.

## Core Features
* Focus Tracking: Monitors the presence and engagement of the user in real-time.
* Distraction Detection: Identifies potential interruptions such as mobile phone usage or the presence of multiple people in the frame.
* Efficiency Scoring: Calculates a productivity percentage by comparing focus time against distracted time.
* Session Reporting: Automatically logs every session summary into a session_report.txt file for long-term tracking.

## Technical Overview
The system is built with Python, utilizing the YOLOv8 model for object detection and OpenCV for real-time video processing. The logic is optimized for low latency to ensure a seamless monitoring experience.

## Installation and Usage
1. Clone the repository to your local machine.
2. Install the required dependencies: pip install opencv-python ultralytics
3. Run the application: python StudyApp.py

This project represents my initial work in computer vision and artificial intelligence. I am actively working on further enhancements.
