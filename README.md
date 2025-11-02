
# AI Assignment – Footfall Counter – Akash S

## 📘 Project Summary
This project implements a computer vision-based **Footfall Counter** using the YOLO model for person detection and tracking.  
It counts the number of people **entering** and **exiting** through a defined virtual line in a video.

---

## Objectives
- Detect humans in a video stream  
- Track movements frame-by-frame  
- Define a line or Region of Interest (ROI)  
- Count entries and exits when people cross the ROI  

---

## Approach
1. **Detection:** YOLOv8 (Ultralytics) used for human detection  
2. **Tracking:** Centroid tracking logic for movement tracking  
3. **Counting Logic:**  
   - When a person crosses the ROI from one direction → `IN` count  
   - When they cross back → `OUT` count  
4. **Visualization:** Displays live bounding boxes and count overlay  

---

## Installation & Setup

### Clone the repository
```bash
git clone https://github.com/Aaka-shh/Football_counter.git
cd Football_counter
