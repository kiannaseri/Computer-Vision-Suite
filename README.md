# 🚀 Computer Vision Suite

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Introduction

This is a comprehensive and modular project for learning and implementing various computer vision techniques. Using this project, you can implement different applications such as face recognition, smart attendance system, people counter, hand gesture recognition, and drowsiness detection.

### ✨ Main Features

- **🔍 Face Recognition**: Identify and recognize faces with high accuracy
- **📊 Smart Attendance**: Automatic check-in and check-out recording
- **🔐 Face Login System**: System login with face recognition
- **📸 Auto Save**: Save images of unknown people
- **👥 Multi-Face Detection**: Simultaneous recognition of multiple faces
- **🔊 Voice Alert**: Play alerts for unknown people
- **🧮 People Counter**: Count people traffic (suitable for stores)
- **🖐️ Hand Gesture Recognition**: Control computer with hand movements
- **😴 Drowsiness Detection**: Alert for tired drivers

---

## 🛠️ Technologies Used

- **Python 3.7+** - Main programming language
- **OpenCV** - Image and video processing
- **NumPy** - Numerical computations
- **Pandas** - Data management
- **MediaPipe** - Hand detection (optional)
- **Pyttsx3** - Text to speech conversion
- **PyAutoGUI** - Keyboard and volume control

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.7 or higher
- Webcam or camera

### Installation Steps

1. **Clone the project**
```bash
git clone https://github.com/kiannaseri/computer-vision-suite.git
cd computer-vision-suite
```

2. **Create virtual environment (optional but recommended)**
```bash
python -m venv venv
# Activate on Windows
venv\Scripts\activate
# Activate on Linux/Mac
source venv/bin/activate
```

3. **Install required libraries**
```bash
pip install -r requirements.txt
```

4. **Install additional libraries (optional)**
```bash
# For hand gesture recognition (optional)
pip install mediapipe pyautogui

# For voice playback
pip install pyttsx3
```

---

## 🚀 How to Use

### Run the main program
```bash
python face_system.py
```

### Main Program Menu
After running, the following menu will be displayed:

```
============================================================
🚀 Computer Vision Projects
============================================================
1️⃣  Hand Gesture Recognition (Control volume with fingers)
2️⃣  Drowsiness Detection (Alert for drivers)
3️⃣  Object Detection (only person)
------------------------------------------------------------
9️⃣  Exit
============================================================
```

### Advanced Modules (if activated)

```
============================================================
🚀 Computer Vision Suite
============================================================
📌 Main Projects (Face Recognition):
1️⃣  Auto Save Unknown Faces
2️⃣  Advanced Attendance System
3️⃣  Voice Alert for Unknown
4️⃣  Multi-Face Detection
5️⃣  Face Login System
------------------------------------------------------------
📌 New Projects:
6️⃣  People Counter
7️⃣  Hand Gesture Recognition
8️⃣  Driver Drowsiness Detection
------------------------------------------------------------
0️⃣  Collect New Face Samples
9️⃣  Exit
============================================================
```

---

## 📁 Project Structure

```
computer-vision-suite/
│
├── face_system.py          # Main program file
├── requirements.txt        # Project dependencies
├── README.md               # This file
│
├── data/                   # Program data
│   ├── database/           # Face database
│   ├── unknown_faces/      # Unknown people images
│   ├── attendance_logs/    # Attendance logs
│   └── sounds/             # Sound files
│
└── modules/                # Program modules (optional)
    ├── 01_auto_save.py
    ├── 02_attendance.py
    └── ...
```

---

## 📝 requirements.txt

```txt
opencv-python==4.8.1.78
numpy==1.24.3
pandas==2.0.3
pyttsx3==2.90
pyautogui==0.9.54
mediapipe==0.10.7  # optional
```

---

## 🎯 Modules and Applications

### 1️⃣ Hand Gesture Recognition
- **Application**: Control volume and execute commands with hand movements
- **Commands**: 
  - 1 finger = Mute
  - 2 fingers = Volume down
  - 3 fingers = Volume up
  - 4 fingers = Take photo
  - 5 fingers = Hello

### 2️⃣ Drowsiness Detection
- **Application**: Alert for tired drivers
- **Function**: If eyes are closed for more than 2 seconds, plays an alarm

### 3️⃣ Smart Attendance
- **Application**: Automatic employee check-in/check-out
- **Features**: Date and time recording, daily reports

### 4️⃣ People Counter
- **Application**: Customer traffic counting for stores
- **Function**: Count entries/exits and show current people inside

---

## 🔧 Advanced Settings

### Camera Settings
If your camera doesn't work, change the camera index in the code:
```python
self.cap = cv2.VideoCapture(0)  # 0 for default camera, 1 for second camera
```

### Recognition Threshold
To adjust face recognition accuracy:
```python
RECOGNITION_THRESHOLD = 6000  # Lower number = higher accuracy
```

---

## ⚠️ Troubleshooting Common Issues

### Issue: Camera doesn't work
**Solution**: 
- Make sure camera is not being used by other apps (Zoom, Teams)
- Change camera index (0, 1, 2)

### Issue: Face recognition is not accurate
**Solution**:
- Improve lighting conditions
- Collect more face samples
- Decrease `RECOGNITION_THRESHOLD` value

### Issue: MediaPipe won't install
**Solution**:
```bash
pip install mediapipe==0.10.7
# Or use pure OpenCV method
```

---

## 🤝 Contributing

If you want to improve this project:
1. Fork the project
2. Apply your changes
3. Send a Pull Request

---

## 📄 License

This project is released under the MIT license. You are free to use and modify it.

---

## 📞 Contact

- **Email**: kian.n.8484@gmail.com
- **GitHub**: [https://github.com/kiannaseri](https://github.com/kiannaseri)
- **Telegram**: [@kian_naseri](https://t.me/kian_naseri)

---

## 🌟 Acknowledgments

Thanks to everyone who helped improve this project.

---

**Developed with ❤️ for learning and advancing in computer vision**
**By: Kian Naseri**

---

================================================================================

---

# 🚀 پروژه جامع بینایی کامپیوتر

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 معرفی پروژه

این یک پروژه جامع و ماژولار برای یادگیری و پیاده‌سازی تکنیک‌های مختلف بینایی کامپیوتر است. با استفاده از این پروژه می‌توانید برنامه‌های مختلفی مانند تشخیص چهره، حضور و غیاب هوشمند، شمارشگر افراد، تشخیص ژست دست و تشخیص خواب‌آلودگی را پیاده‌سازی کنید.

### ✨ ویژگی‌های اصلی

- **🔍 تشخیص چهره**: شناسایی و تشخیص چهره افراد با دقت بالا
- **📊 حضور و غیاب هوشمند**: ثبت خودکار ورود و خروج افراد
- **🔐 سیستم لاگین با چهره**: ورود به سیستم با تشخیص چهره
- **📸 ذخیره خودکار**: ذخیره تصاویر افراد ناشناس
- **👥 تشخیص چند نفر**: شناسایی همزمان چند چهره
- **🔊 هشدار صوتی**: پخش هشدار برای افراد ناشناس
- **🧮 شمارشگر افراد**: شمارش تردد افراد (مناسب فروشگاه‌ها)
- **🖐️ تشخیص ژست دست**: کنترل کامپیوتر با حرکات دست
- **😴 تشخیص خواب‌آلودگی**: هشدار به رانندگان خسته

---

## 🛠️ تکنولوژی‌های استفاده شده

- **Python 3.7+** - زبان اصلی برنامه‌نویسی
- **OpenCV** - پردازش تصویر و ویدئو
- **NumPy** - محاسبات عددی
- **Pandas** - مدیریت داده‌ها
- **MediaPipe** - تشخیص دست (اختیاری)
- **Pyttsx3** - تبدیل متن به صدا
- **PyAutoGUI** - کنترل کیبورد و صدا

---

## 📦 نصب و راه‌اندازی

### پیش‌نیازها
- Python 3.7 یا بالاتر
- Webcam یا دوربین

### مراحل نصب

1. **کلون کردن پروژه**
```bash
git clone https://github.com/kiannaseri/computer-vision-suite.git
cd computer-vision-suite
```

2. **ایجاد محیط مجازی (اختیاری اما توصیه می‌شود)**
```bash
python -m venv venv
# فعال‌سازی در ویندوز
venv\Scripts\activate
# فعال‌سازی در لینوکس/مک
source venv/bin/activate
```

3. **نصب کتابخانه‌های مورد نیاز**
```bash
pip install -r requirements.txt
```

4. **نصب کتابخانه‌های اضافی (اختیاری)**
```bash
# برای تشخیص ژست دست (اختیاری)
pip install mediapipe pyautogui

# برای پخش صدا
pip install pyttsx3
```

---

## 🚀 نحوه استفاده

### اجرای برنامه اصلی
```bash
python face_system.py
```

### منوی اصلی برنامه
پس از اجرا، منوی زیر نمایش داده می‌شود:

```
============================================================
🚀 پروژه‌های بینایی کامپیوتر
============================================================
1️⃣  تشخیص ژست دست (کنترل صدا با انگشتان)
2️⃣  تشخیص خواب‌آلودگی (هشدار به راننده)
3️⃣  تشخیص اشیاء (فقط person)
------------------------------------------------------------
9️⃣  خروج
============================================================
```

### ماژول‌های پیشرفته (در صورت فعال‌سازی)

```
============================================================
🚀 سوپرمنوی پروژه‌های بینایی کامپیوتر
============================================================
📌 پروژه‌های اصلی (تشخیص چهره):
1️⃣  ذخیره خودکار تصاویر ناشناس
2️⃣  حضور و غیاب پیشرفته
3️⃣  هشدار صوتی برای ناشناس
4️⃣  تشخیص چند نفر همزمان
5️⃣  لاگین با چهره
------------------------------------------------------------
📌 پروژه‌های جدید:
6️⃣  شمارشگر افراد (People Counter)
7️⃣  تشخیص ژست دست
8️⃣  تشخیص خواب‌آلودگی راننده
------------------------------------------------------------
0️⃣  جمع‌آوری نمونه چهره جدید
9️⃣  خروج
============================================================
```

---

## 📁 ساختار پروژه

```
computer-vision-suite/
│
├── face_system.py          # فایل اصلی برنامه
├── requirements.txt        # وابستگی‌های پروژه
├── README.md               # این فایل
│
├── data/                   # داده‌های برنامه
│   ├── database/           # دیتابیس چهره‌ها
│   ├── unknown_faces/      # تصاویر افراد ناشناس
│   ├── attendance_logs/    # لاگ‌های حضور و غیاب
│   └── sounds/             # فایل‌های صوتی
│
└── modules/                # ماژول‌های برنامه (اختیاری)
    ├── 01_auto_save.py
    ├── 02_attendance.py
    └── ...
```

---

## 📝 فایل requirements.txt

```txt
opencv-python==4.8.1.78
numpy==1.24.3
pandas==2.0.3
pyttsx3==2.90
pyautogui==0.9.54
mediapipe==0.10.7  # اختیاری
```

---

## 🎯 ماژول‌ها و کاربردها

### 1️⃣ تشخیص ژست دست
- **کاربرد**: کنترل صدا و اجرای فرمان‌ها با حرکات دست
- **دستورات**: 
  - ۱ انگشت = قطع صدا
  - ۲ انگشت = کم کردن صدا
  - ۳ انگشت = زیاد کردن صدا
  - ۴ انگشت = عکس گرفتن
  - ۵ انگشت = سلام

### 2️⃣ تشخیص خواب‌آلودگی
- **کاربرد**: هشدار به رانندگان خسته
- **عملکرد**: اگه چشم‌ها بیش از ۲ ثانیه بسته باشند، بوق هشدار پخش می‌کند

### 3️⃣ حضور و غیاب هوشمند
- **کاربرد**: ثبت خودکار ورود و خروج پرسنل
- **ویژگی‌ها**: ذخیره تاریخ و ساعت، گزارش روزانه

### 4️⃣ شمارشگر افراد
- **کاربرد**: آمارگیری از تعداد مشتریان فروشگاه
- **عملکرد**: شمارش ورود و خروج و نمایش افراد حاضر

---

## 🔧 تنظیمات پیشرفته

### تنظیم دوربین
اگر دوربین شما کار نمی‌کند، ایندکس دوربین را در کد تغییر دهید:
```python
self.cap = cv2.VideoCapture(0)  # 0 برای دوربین پیش‌فرض، 1 برای دوربین دوم
```

### تنظیم آستانه تشخیص
برای تنظیم دقت تشخیص چهره:
```python
RECOGNITION_THRESHOLD = 6000  # عدد کمتر = دقت بیشتر
```

---

## ⚠️ رفع مشکلات رایج

### مشکل: دوربین کار نمی‌کند
**راه حل**: 
- مطمئن شوید دوربین به برنامه‌های دیگر (Zoom, Teams) وصل نیست
- ایندکس دوربین را تغییر دهید (0, 1, 2)

### مشکل: تشخیص چهره دقیق نیست
**راه حل**:
- نور محیط را بهتر کنید
- نمونه‌های بیشتری از چهره جمع‌آوری کنید
- مقدار `RECOGNITION_THRESHOLD` را کاهش دهید

### مشکل: MediaPipe نصب نمی‌شود
**راه حل**:
```bash
pip install mediapipe==0.10.7
# یا از روش OpenCV خالص استفاده کنید
```

---

## 🤝 مشارکت در پروژه

اگر مایل به بهبود پروژه هستید:
1. پروژه را Fork کنید
2. تغییرات خود را اعمال کنید
3. Pull Request ارسال کنید

---

## 📄 لایسنس

این پروژه تحت لایسنس MIT منتشر شده است. آزادانه می‌توانید از آن استفاده و تغییرش دهید.

---

## 📞 تماس با ما

- **ایمیل**: kian.n.8484@gmail.com
- **گیت‌هاب**: [https://github.com/kiannaseri](https://github.com/kiannaseri)
- **تلگرام**: [@kian_naseri](https://t.me/kian_naseri)

---

## 🌟 قدردانی

از همه کسانی که در بهبود این پروژه کمک کردند، سپاسگزاریم.

---

**توسعه داده شده با ❤️ برای یادگیری و پیشرفت در زمینه بینایی کامپیوتر**
**توسط: کیان ناصری**
