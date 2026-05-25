# SkinSense AI — Hardware Integration Guide

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Raspberry Pi 4 (8GB)                  │
│                                                          │
│  ┌──────────────┐    ┌──────────────┐   ┌────────────┐  │
│  │  Camera Mod  │    │  Python App  │   │  AI Model  │  │
│  │  3 NoIR      │───▶│  (main loop) │──▶│  (TFLite)  │  │
│  │  CSI ribbon  │    │              │   └────────────┘  │
│  └──────────────┘    │  picamera2   │                   │
│                      │  OpenCV      │                   │
│                      │  serial      │                   │
│                      └──────┬───────┘                   │
│                             │ UART /dev/ttyS0            │
└─────────────────────────────┼───────────────────────────┘
                              │ TX/RX 3.3V TTL
                    ┌─────────▼──────────┐
                    │ Nextion NX4827T043  │
                    │  4.3" 480×272 TFT   │
                    │  Touch + own CPU    │
                    └────────────────────┘
```

## Hardware Wiring

| Pi 4 GPIO Pin | Nextion Pin | Signal |
|--------------|-------------|--------|
| Pin 6 (GND)  | GND         | Ground |
| Pin 4 (5V)   | +5V         | Power  |
| Pin 8 (TX)   | RX          | Data → |
| Pin 10 (RX)  | TX          | Data ← |

⚠️  Nextion RX/TX logic is 5V tolerant but Pi GPIO is 3.3V.
    Use a 1kΩ + 2kΩ voltage divider on Pi TX → Nextion RX.

## Nextion Editor Page Design

Design these 6 pages in **Nextion Editor** and upload via microSD:

### Page 0 – Welcome
- Background: #061B28 (RGB565: 0x030D)
- Text "SkinSense AI" — font 32px white
- Picture component: logo (pre-loaded)
- Button b0: "TAP TO BEGIN" — green 0x07E0

### Page 1 – Prep
- 3 × Text components with icons
- Button b0: "I'm Ready" (green)
- Button b1: "Back" (grey outline)

### Page 2 – Position
- Picture p0: alignment illustration (480×160)
- Text t0: hint text
- Button b0: "Open Camera"
- Button b1: "Back"

### Page 3 – Capture
- Picture p0: camera preview crop (160×120, centre-right)
- Text x0: alignment feedback (changes color dynamically)
- Button b0: "Start Scan"
- Button b1: "Back"
- Note: Pi sends status via `x0.txt` and `x0.bco` commands

### Page 4 – Analyzing
- Progress bar j0 (480×12, bottom strip)
- Text t0: stage label (updates every ~0.8s)
- Spinning animation: use Nextion timer + picture swap

### Page 5 – Result
- Text t0: risk label (color-coded)
- Text t1: score (e.g. "92/100")
- Text t2: recommendation note
- Button b0: "New Scan"
- Button b1: "Save Report"

## Nextion Serial Protocol

All commands are ASCII strings terminated by 3×0xFF:

```
page 3\xFF\xFF\xFF          → go to page 3
t0.txt="Hello"\xFF\xFF\xFF  → set text
t0.bco=2016\xFF\xFF\xFF     → set bg color (RGB565)
j0.val=75\xFF\xFF\xFF       → set progress bar to 75%
vis b1,1\xFF\xFF\xFF        → show component
```

Touch event received from Nextion (7 bytes):
```
0x65  PAGE  COMP  0x01  0xFF  0xFF  0xFF   (press)
0x65  PAGE  COMP  0x00  0xFF  0xFF  0xFF   (release)
```

## Pi Setup Commands

```bash
# 1. Enable UART (disable Bluetooth, free /dev/ttyS0)
echo "dtoverlay=disable-bt" >> /boot/config.txt
echo "enable_uart=1" >> /boot/config.txt

# 2. Install Python deps
pip install picamera2 pyserial opencv-python numpy

# 3. Install system deps
sudo apt install libcamera-apps python3-libcamera

# 4. Run on boot
sudo nano /etc/systemd/system/skinsense.service
```

```ini
[Unit]
Description=SkinSense AI
After=multi-user.target

[Service]
Type=idle
ExecStart=/usr/bin/python3 /home/pi/skinsense/skinsense_pi.py
Restart=always
User=pi

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable skinsense
sudo systemctl start skinsense
```

## Camera Optimization for NoIR

```python
# For indoor skin imaging without IR filter:
controls = {
    "AwbMode": 1,          # Daylight white balance
    "ExposureTime": 16666, # ~60fps shutter
    "AnalogueGain": 2.0,   # Mild ISO boost
    "Sharpness": 2.0,      # Edge detail for lesion borders
    "Saturation": 1.2,     # Skin tone detail
}
```

## UI Color Palette (RGB565 for Nextion)

| Color    | Hex    | RGB565  | Use              |
|----------|--------|---------|------------------|
| Teal     | #00E5C8| 0x0738  | Primary accent   |
| Green    | #00FF88| 0x07F1  | Good/Low risk    |
| Red      | #FF4757| 0xFA0A  | Warning/Bad      |
| Yellow   | #FFD32A| 0xFE85  | Caution          |
| Dark BG  | #061B28| 0x030D  | Screen background|
| Mid blue | #0A2535| 0x012A  | Card background  |
