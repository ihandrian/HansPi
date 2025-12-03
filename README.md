# HansPi - Raspberry Pi Robot Controller

A web-based robot controller for Raspberry Pi with motor control, camera streaming, and **person detection/following** capabilities. Optimized for **Raspberry Pi 5** with gpiozero library support.

## 🚀 Quick Start Guide (Beginner Friendly)

### Prerequisites

- **Raspberry Pi 5** (or Pi 4) running Raspberry Pi OS
- Motor driver board (H-bridge, e.g., L298N) connected to GPIO pins
- Camera module or USB webcam
- Basic knowledge of terminal commands
- Internet connection (for downloading AI models on first run)

### Step 1: Update Your System

First, make sure your Raspberry Pi is up to date:

```bash
sudo apt update
sudo apt upgrade -y
```

### Step 2: Install System Dependencies

Install the GPIO libraries that work with Raspberry Pi 5:

```bash
sudo apt install -y python3-gpiozero python3-lgpio python3-rpi-lgpio python3-pip
```

**Why these packages?**
- `python3-gpiozero`: High-level GPIO library for easy motor control
- `python3-lgpio`: Low-level GPIO library (required for Pi 5)
- `python3-rpi-lgpio`: Raspberry Pi specific GPIO support
- `python3-pip`: Python package installer

### Step 3: Create Virtual Environment (Recommended)

Using a virtual environment keeps your project dependencies isolated:

```bash
# Navigate to HansPi directory
cd ~/Desktop/HansPi

# Create virtual environment
python3 -m venv PiRobot

# Activate virtual environment
source PiRobot/bin/activate

# Your prompt should now show (PiRobot)
```

**Note:** Always activate the virtual environment before running the robot:
```bash
source PiRobot/bin/activate
```

### Step 4: Install Python Dependencies

Install the required Python packages:

```bash
# Make sure virtual environment is activated
source PiRobot/bin/activate

# Install basic dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**What gets installed:**
- `Flask>=2.0.0` - Web framework for control panel
- `opencv-python>=4.5.0` - Computer vision for camera streaming
- `psutil>=5.8.0` - System monitoring (CPU, memory, temperature)
- `requests>=2.25.0` - For downloading AI models

### Step 5: Install TensorFlow Lite (For Person Detection)

**HansPi uses TensorFlow Lite for accurate person detection.** Choose one option:

#### Option A: Install tflite_runtime (Recommended - Smaller & Faster)

**⚠️ Important: Check your Python version first:**
```bash
python3 --version
```

**Available wheels for Raspberry Pi 5 (64-bit, aarch64):**
- **Python 3.11**: ✅ Available
- **Python 3.12**: ✅ Available  
- **Python 3.13+**: ❌ Not available - Use Option B instead

**Installation for Python 3.11 or 3.12:**
```bash
# Activate virtual environment first
source PiRobot/bin/activate

# For Python 3.11
pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.14.0-cp311-cp311-linux_aarch64.whl

# For Python 3.12 (if available)
pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.14.0-cp312-cp312-linux_aarch64.whl

# Check all available wheels at:
# https://github.com/google-coral/pycoral/releases
```

**If you get "wheel not supported" error:**
- Your Python version (3.13+) doesn't have a pre-built wheel yet
- **Solution**: Use Option B (full TensorFlow) - it works with all Python versions

For Raspberry Pi 4 (32-bit, armv7l):
```bash
# For Python 3.9
pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.14.0-cp39-cp39-linux_armv7l.whl
```

#### Option B: Install Full TensorFlow (Recommended for Python 3.13+)

**This is the easiest option and works with ALL Python versions (including 3.13+):**

```bash
source PiRobot/bin/activate
pip install tensorflow
```

**Why use this:**
- ✅ Works with Python 3.13+ (no wheel compatibility issues)
- ✅ Easier installation (no need to find specific wheels)
- ✅ More features available
- ⚠️ Larger download (~500MB) but worth it for compatibility

**Note:** 
- If TensorFlow is not installed, person detection will be disabled, but the robot will still work for manual control
- **For Python 3.13+ users**: This is the recommended option

### Step 6: Check Your File Structure

Make sure your project folder looks like this:

```
HansPi/
├── main.py                   # Main robot controller
├── autonav.py                # Autonomous navigation module
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore                # Git ignore file (excludes cache/logs)
├── templates/
│   ├── index.html            # Web interface HTML
│   └── style.css             # Web interface styles
├── models/                   # (created automatically - AI models stored here)
└── robot.log                 # (created automatically - excluded from git)
```

### Step 7: Connect Your Hardware

**Motor Connections (BCM GPIO numbering):**
- **Motor 1:**
  - IN1 → GPIO 17
  - IN2 → GPIO 18
  - ENABLE A → GPIO 12
- **Motor 2:**
  - IN1 → GPIO 27
  - IN2 → GPIO 22
  - ENABLE B → GPIO 13

**Power Connections:**
- Connect 12V power supply to motor driver board
- Connect 5V to Raspberry Pi
- **IMPORTANT:** Connect all grounds together (motor driver GND, Pi GND, power supply GND)

**Camera:**
- Connect your camera module or USB webcam
- The system will automatically detect available cameras

### Step 8: Run the Robot Controller

```bash
# Make sure you're in the HansPi directory
cd ~/Desktop/HansPi

# Activate virtual environment
source PiRobot/bin/activate

# Run the controller
python3 main.py
```

**First Run Notes:**
- On first run, the system will automatically download the MobileNet SSD AI model (~27MB)
- This may take a few minutes depending on your internet speed
- The model will be saved in the `models/` directory for future use

You should see output like:
```
2025-11-26 14:52:53 - RobotController - INFO - Using LGPIOFactory for gpiozero pin control
2025-11-26 14:52:53 - RobotController - INFO - Motor controller initialized with gpiozero
2025-11-26 14:52:53 - RobotController - INFO - Person detector initialized successfully
2025-11-26 14:52:53 - RobotController - INFO - Camera controller initialized. Found cameras: [0]
2025-11-26 14:52:53 - RobotController - INFO - Web server running at http://0.0.0.0:5002
```

### Step 9: Access the Control Panel

1. Find your Raspberry Pi's IP address:
   ```bash
   hostname -I
   ```

2. Open a web browser on any device connected to the same network

3. Navigate to: `http://<your_pi_ip>:5002`
   - Example: `http://192.168.1.100:5002`

4. You should see the robot control panel with:
   - **Fullscreen camera feed** (background)
   - **System info panel** (top) - CPU, Memory, Temp, Battery, Model, TPI
   - **Joystick control** (center bottom) - Drag to control robot movement
   - **Settings panel** (right bottom) - Camera selection, speed control, person detection

## 🛠️ Troubleshooting

### Problem: "LGPIOFactory not available"
**Solution:** Install the system packages:
```bash
sudo apt install python3-lgpio python3-rpi-lgpio
```

### Problem: "Cannot determine SOC peripheral base address"
**Solution:** This means gpiozero is using the wrong backend. Make sure you installed:
```bash
sudo apt install python3-gpiozero python3-lgpio python3-rpi-lgpio
```

### Problem: "TensorFlow Lite not available" or "wheel not supported"
**Solution:** 
- **For Python 3.13+**: Use full TensorFlow instead: `pip install tensorflow`
- **For Python 3.11-3.12**: Check your Python version: `python3 --version`
- Make sure you activated the virtual environment: `source PiRobot/bin/activate`
- If tflite_runtime wheel doesn't work, use full TensorFlow (Option B in Step 5)
- If you don't need person detection, the robot will work fine without it

### Problem: "Error downloading MobileNet model"
**Solution:**
- Check your internet connection
- The model download happens automatically on first run
- If it fails, you can manually download from:
  `https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip`
- Extract to `HansPi/models/` directory

### Problem: Camera not detected
**Solution:**
- Check camera connection
- For USB cameras, try different USB ports
- For Pi camera module, enable it: `sudo raspi-config` → Interface Options → Camera → Enable
- Check logs: `tail -f robot.log`

### Problem: Motors not responding
**Solution:**
- Check GPIO pin connections match the configuration
- Verify motor driver board power supply (12V)
- Check that pins match: GPIO 17, 18, 12, 27, 22, 13
- Ensure you're using BCM GPIO numbering (not physical pin numbers)
- Test with low speed first (20-30%)

### Problem: Motors spinning on startup
**Solution:**
- ✅ **Fixed in latest version**: Motors are now properly initialized with enable pin set to 0 before direction pins
- The motor controller ensures motors are disabled during initialization
- If you experience this issue, make sure you're using the latest code from the repository
- Motors should remain stopped when the program starts

### Problem: "Module not found" errors
**Solution:** 
```bash
# Activate virtual environment
source PiRobot/bin/activate

# Install missing packages
pip install -r requirements.txt
```

### Problem: Person detection not working
**Solution:**
- Make sure TensorFlow Lite is installed (see Step 5)
- Check logs: `tail -f robot.log | grep -i detection`
- Enable detection in the web interface (toggle switch in settings panel)
- Ensure good lighting conditions for better detection

### Problem: Virtual environment not activating
**Solution:**
```bash
# Recreate virtual environment
cd ~/Desktop/HansPi
rm -rf PiRobot
python3 -m venv PiRobot
source PiRobot/bin/activate
pip install -r requirements.txt
```

## 📋 Features

- ✅ **Motor Control**: Forward, backward, left, right, and stop via joystick
- ✅ **Safe Motor Initialization**: Motors are properly disabled on startup (prevents unintended spinning)
- ✅ **PWM Speed Control**: Adjustable motor speed (0-100%)
- ✅ **Camera Streaming**: Real-time fullscreen video feed
- ✅ **Multi-Camera Support**: Switch between multiple cameras
- ✅ **System Monitoring**: CPU, memory, temperature, battery, and TPI display
- ✅ **Person Detection**: AI-powered person detection using TensorFlow Lite
- ✅ **Person Following**: Automatic person following mode
- ✅ **Web Interface**: Modern, responsive control panel accessible from any device
- ✅ **Mobile Friendly**: Works on phones, tablets, and computers

## 🎮 How to Use

### Manual Control
1. Use the **joystick** (center bottom) to control the robot
2. Drag the joystick in the direction you want to move
3. Adjust speed using the **Speed slider** in the settings panel

### Person Following
1. Enable **Person Detection** toggle in settings panel
2. Wait for green boxes to appear around detected persons
3. Click **Start Following** button
4. The robot will automatically follow the closest person
5. Adjust **Follow Speed** slider to control following speed
6. Click **Stop Following** to stop

### Camera Controls
- **Camera Selection**: Choose from available cameras in dropdown
- **Rotate Camera**: Toggle to flip camera view 180 degrees
- **Speed Control**: Adjust motor speed (0-100%)

## 🔧 Configuration

### Change Motor Pins

Edit `main.py` and modify the `MOTOR_PINS` dictionary:

```python
self.MOTOR_PINS = {
    "motor1": {"in1": 17, "in2": 18, "enable": 12},
    "motor2": {"in1": 27, "in2": 22, "enable": 13}
}
```

### Change Web Server Port

Edit the `Robot` class initialization in `main.py`:

```python
self.web_server = RobotWebServer(
    self.motor_controller, 
    self.camera_controller,
    person_detector=self.person_detector,
    person_follower=self.person_follower,
    host="0.0.0.0",
    port=5002  # Change this to your desired port
)
```

### Change Detection Model

The system uses MobileNet SSD by default (faster). To use YOLOv3 (more accurate):
- The model will be downloaded automatically on first use
- You can switch models via the web interface (if implemented) or code

## 🚦 Usage Tips

1. **Start with low speed**: Test motors at 20-30% speed first
2. **Check connections**: Double-check all GPIO connections before running
3. **Monitor logs**: Check `robot.log` for detailed error messages
4. **Use same network**: Control panel only works on the same network as your Pi
5. **Safe shutdown**: Press `Ctrl+C` to stop the robot controller safely
6. **Person detection**: Works best in good lighting conditions
7. **Following mode**: Start with low follow speed (30-40%) for safety
8. **Motor safety**: Motors are automatically disabled on startup - they won't spin until you control them

## 📦 Dependencies Summary

### System Packages (via apt)
- `python3-gpiozero` - GPIO control
- `python3-lgpio` - Low-level GPIO (Pi 5)
- `python3-rpi-lgpio` - Raspberry Pi GPIO support
- `python3-pip` - Python package manager

### Python Packages (via pip)
- `Flask>=2.0.0` - Web framework
- `opencv-python>=4.5.0` - Computer vision
- `psutil>=5.8.0` - System monitoring
- `requests>=2.25.0` - HTTP requests (for model downloads)

### Optional (for Person Detection)
- `tflite_runtime` - TensorFlow Lite runtime (recommended)
- OR `tensorflow` - Full TensorFlow (alternative)

## 🔮 Future Development

1. **Enhanced Sensors**: Add ultrasonic, IR, and other sensors
2. **Autonomous Navigation**: Implement path planning and obstacle avoidance
3. **Advanced AI**: Add object detection, face recognition
4. **Data Logging**: Implement data collection for training
5. **Mobile App**: Create a dedicated mobile app

## 📝 License

This project is open source. Feel free to modify and use as needed.

## 💡 Need Help?

- Check the logs: `tail -f robot.log`
- Verify all hardware connections
- Ensure all dependencies are installed correctly
- Make sure you're using Raspberry Pi OS
- Activate virtual environment before running: `source PiRobot/bin/activate`

## 🎯 Quick Reference

### Start Robot
```bash
cd ~/Desktop/HansPi
source PiRobot/bin/activate
python3 main.py
```

### Stop Robot
Press `Ctrl+C` in the terminal

### Check Status
```bash
tail -f robot.log
```

### Reinstall Dependencies
```bash
source PiRobot/bin/activate
pip install -r requirements.txt
```

---

**Happy Robot Building! 🤖**
