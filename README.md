# SudoPsPsPs — Assistive Robot Cat

SudoPsPsPs is an assistive robotic cat designed to provide companionship and proactive social interaction. Using a pan-tilt head mechanism, computer vision (Intel RealSense D455), local speech processing, and a Liquid Foundation Model (LFM), the robot maintains attention on users and engages in meaningful, mood-aware conversations.

All interactions are driven by **vocal commands and natural conversation**, processed locally via Faster-Whisper, with heavy visual and LLM reasoning offloaded to a dedicated inference server.


![Alt Text](Images/Image%201.png)

![Alt Text](Images/Completed%20Build%20Image%201.png)

---

## Hardware

### Pan-Tilt Mechanism

| Component | Description |
|-----------|-------------|
| Pan Servo | Dynamixel XL430-W250-T (ID 1) |
| Tilt Servo | Dynamixel XL430-W250-T (ID 2) |
| Controller | OpenCR |
| Interface | USB Serial |
| Protocol | Dynamixel Protocol 2.0 |

### Sensor & Audio Suite

| Component | Details |
|-----------|---------|
| Vision | Intel RealSense D455 |
| Microphone | Steinberg UR22mkII audio interface |
| Speaker | Local ALSA-routed speaker |

---

## Software Architecture

The system uses **ROS 2 (Humble)** as middleware connecting hardware control, perception, and reasoning. Two machines communicate over a dedicated Ethernet link.

| Machine | Role |
|---------|------|
| **Macbook** (Inference Server) | Runs `model_server.py` and the LFM. Handles NLP, task scheduling reasoning, and vision-based VAD emotion inference. |
| **Lenovo LOQ** (Robot Host) | Runs all ROS 2 nodes. Handles hardware control (Dynamixels, RealSense), STT/TTS, image preprocessing, and the behavioral state machine. |
| **Ethernet `10.42.0.x`** | High-speed local link passing image crops and JSON payloads between the two machines. |


![Alt Text](Images/rosgraph.png)
All the nodes, aside from cat_brain_planner and lfm_bridge_node, run on the Lenovo LOQ side while the cat_brain_planner and lfm_bridge_node acts as the interface to the Inference server hosted on the Macbook.

---

## Codebase Overview

### Hardware Control

- **`dynamixel_driver.py`** — OpenCR communication, torque enablement, and startup calibration. Converts degrees to raw Dynamixel 2.0 protocol positions.
- **`head_node.py`** — ROS 2 hardware abstraction node. Subscribes to `/head/pan_target` and `/head/tilt_target`; applies Exponential Moving Average (EMA) smoothing to motor commands.

### Vision & Tracking

- **`person_targetting_node.py`** — Core visual state machine (`SEARCHING` → `TRACKING` → `TALKING`). Uses MediaPipe Pose for bounding boxes and implements cat-like kinematics: discrete saccades, generous dead-zones, and curious idle head-tilts.
- **`image_preprocessor.py`** — Active only when locked onto a user. Crops and resizes RealSense frames (224×224 person crop, 336×336 spatial context) and publishes them for inference.

### Audio & Speech Processing

- **`stt_node.py`** — Local ASR via `faster-whisper` on GPU. Features an RMS noise threshold gate; publishes transcriptions to `/cat/stt_input`. This is the **primary trigger** for robot interaction.
- **`tts_node.py`** — Offline TTS using Piper. Subscribes to `/cat/robot_actions`, extracts `message_to_user` from JSON payloads, and routes audio to the local ALSA system.
- **`speaker_finder.py`** — PyAudio utility to enumerate hardware audio devices and identify the correct speaker index.

### AI & Reasoning

- **`lfm_bridge_node.py`** — Async bridge node. Packages cropped user images as base64 payloads and POSTs them to the Macbook vision model. Publishes Valence, Arousal, and Dominance (VAD) scores to `/cat/emotional_state`.
- **`cat_brain_planner.py`** — The cognitive core. Manages multi-turn conversation, schedule assignment, and proactive reminders. Fuses STT input with VAD emotional state to generate mood-aware suggestions (e.g., suggesting rest if valence is low).
- **`train_lora.py`** — Standalone QLoRA fine-tuning script for the LFM2 vision-language model using HuggingFace `peft` and `trl`. Aligns the robot's persona and reasoning with the Sudo dataset.

---

## Setup & Launch

### Network Configuration

Connect both machines over Ethernet and assign static IPs:

| Machine | IP |
|---------|----|
| Macbook (Inference Server) | `10.42.0.2` |
| Lenovo LOQ (Robot Host) | `10.42.0.1` |

---

### 1. Macbook — Start the Inference Server

```bash
python model_server.py --host 0.0.0.0 --port 8000
```

---

### 2. Lenovo LOQ — Source the Workspace

```bash
source /opt/ros/humble/setup.bash
source ~/ri_one_master_ws/install/setup.bash
```

---

### 3. Launch Hardware Drivers

```bash
ros2 launch realsense2_camera rs_launch.py
ros2 run sudo_pspsps head_node
```

---

### 4. Spin Up Perception & Audio Nodes

Open separate terminals for each:

```bash
ros2 run sudo_pspsps tts_node
```

---

### Manual Head Test

To verify servo movement before launching the planner:

```bash
ros2 topic pub --once /head/pan_target std_msgs/msg/Float32 "{data: 30.0}"
```

---

## Current Status

| Feature | Status |
|---------|--------|
| Dynamixel communication & ROS 2 head node | Complete |
| Cat-like saccade motion profiling | Complete |
| MediaPipe person tracking & targeting | Complete |
| Local Whisper STT & Piper TTS | Complete |
| Async LFM HTTP bridging to Macbook | Complete |
| VAD emotional state assessment from RealSense crops | Complete |
| Conversational task planning & proactive mood suggestions | Complete |
| QLoRA training pipeline for LFM persona alignment | Complete |

---

## Team Vision

SudoPsPsPs aims to be a robotic companion capable of maintaining attention, understanding conversational context, and proactively checking in on users — not by reacting to physical triggers, but by acting as a supportive social presence guided by vocal input, multimodal perception, and advanced LLM reasoning.
