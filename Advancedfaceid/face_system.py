import cv2
import numpy as np
import pickle
import os
import sys
import datetime
import time
import csv
import pandas as pd
from pathlib import Path

# ==================== تنظیمات ====================
BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / 'data' / 'database'
UNKNOWN_DIR = BASE_DIR / 'data' / 'unknown_faces'
ATTENDANCE_DIR = BASE_DIR / 'data' / 'attendance_logs'
SOUNDS_DIR = BASE_DIR / 'data' / 'sounds'
LOGIN_LOGS_DIR = BASE_DIR / 'data' / 'login_logs'

# ایجاد پوشه‌ها
for dir_path in [DB_DIR, UNKNOWN_DIR, ATTENDANCE_DIR, SOUNDS_DIR, LOGIN_LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ==================== کلاس اصلی ====================
class FaceUtils:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.face_database = {}
        self.face_averages = {}
        self.load_database()

    def load_database(self):
        db_file = DB_DIR / 'face_database.pkl'
        if db_file.exists():
            try:
                with open(db_file, 'rb') as f:
                    self.face_database = pickle.load(f)
                self.calculate_averages()
                print(f"✅ دیتابیس با {len(self.face_database)} نفر بارگذاری شد")
            except:
                print("❌ خطا در بارگذاری دیتابیس")

    def save_database(self):
        db_file = DB_DIR / 'face_database.pkl'
        with open(db_file, 'wb') as f:
            pickle.dump(self.face_database, f)
        print("💾 دیتابیس ذخیره شد")

    def calculate_averages(self):
        self.face_averages = {}
        for name, samples in self.face_database.items():
            if len(samples) > 0:
                self.face_averages[name] = np.mean(samples, axis=0)

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces, gray

    def extract_face(self, gray, x, y, w, h):
        face_roi = gray[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (100, 100))
        return face_roi.flatten()

    def recognize_face(self, face_sample):
        if len(self.face_averages) == 0:
            return "Unknown", float('inf')

        best_match = "Unknown"
        min_distance = float('inf')

        for name, avg_face in self.face_averages.items():
            distance = np.linalg.norm(face_sample - avg_face)
            if distance < min_distance and distance < 6000:
                min_distance = distance
                best_match = name

        return best_match, min_distance

    def collect_samples(self):
        print("\n📸 جمع‌آوری نمونه چهره")
        name = input("اسم فرد را وارد کنید: ").strip()
        if not name:
            print("❌ اسم نمی‌تونه خالی باشه!")
            return

        if name not in self.face_database:
            self.face_database[name] = []

        cap = cv2.VideoCapture(0)
        sample_count = len(self.face_database[name])
        new_samples = 0

        print(f"\nبرای {name} نمونه جمع‌آوری می‌شود")
        print("کلید 's' = ذخیره نمونه")
        print("کلید 'q' = خروج")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces, gray = self.detect_faces(frame)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_sample = self.extract_face(gray, x, y, w, h)

                cv2.putText(frame, f"Name: {name}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Samples: {sample_count + new_samples}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow('Collect Samples', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and len(faces) > 0:
                self.face_database[name].append(face_sample)
                new_samples += 1
                print(f"✅ نمونه {new_samples} ذخیره شد")
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if new_samples > 0:
            self.calculate_averages()
            self.save_database()
            print(f"✅ {new_samples} نمونه جدید برای {name} ذخیره شد")


# ==================== ماژول 1: ذخیره خودکار ====================
class AutoSaveSystem:
    def __init__(self):
        self.fu = FaceUtils()
        self.unknown_counter = 0

    def run(self):
        print("🚀 سیستم ذخیره خودکار تصاویر")
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces, gray = self.fu.detect_faces(frame)

            for (x, y, w, h) in faces:
                face_sample = self.fu.extract_face(gray, x, y, w, h)
                name, _ = self.fu.recognize_face(face_sample)

                if name == "Unknown":
                    color = (0, 0, 255)
                    # ذخیره عکس
                    face_img = frame[y:y + h, x:x + w]
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = UNKNOWN_DIR / f"unknown_{timestamp}.jpg"
                    cv2.imwrite(str(filename), face_img)
                    self.unknown_counter += 1
                    print(f"📸 عکس ناشناس ذخیره شد")
                else:
                    color = (0, 255, 0)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.putText(frame, f"Unknown saved: {self.unknown_counter}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Auto Save System', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# ==================== ماژول 2: حضور و غیاب پیشرفته ====================
# ==================== ماژول 2: حضور و غیاب پیشرفته ====================
# ==================== ماژول 2: حضور و غیاب پیشرفته ====================
# ==================== ماژول 2: حضور و غیاب پیشرفته ====================
class AttendanceSystem:
    def __init__(self):
        self.fu = FaceUtils()
        self.attendance_log = {}  # {name: {'entry': time, 'exit': time, 'date': date}}
        self.daily_report = []
        self.present_faces = {}  # برای ردیابی چهره‌های حاضر
        self.load_today_log()
        self.load_present_state()  # وضعیت حاضرین رو هم بارگذاری کن

        # راه‌اندازی صدا
        self.sound_enabled = False
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)  # سرعت صحبت
            self.engine.setProperty('volume', 0.9)  # بلندی صدا
            self.sound_enabled = True
            print("✅ سیستم صوتی فعال شد")
        except ImportError:
            print("⚠️ pyttsx3 نصب نیست. برای نصب: pip install pyttsx3")
        except Exception as e:
            print(f"⚠️ خطا در راه‌اندازی صدا: {e}")

    def speak(self, message):
        """پخش پیام صوتی"""
        if self.sound_enabled:
            try:
                self.engine.say(message)
                self.engine.runAndWait()
            except:
                print(f"🔊 {message}")
        else:
            print(f"🔊 {message}")

    def load_present_state(self):
        """بارگذاری وضعیت حاضرین از فایل"""
        state_file = ATTENDANCE_DIR / 'present_state.pkl'
        if state_file.exists():
            try:
                with open(state_file, 'rb') as f:
                    self.present_faces = pickle.load(f)
                print(f"👥 {len(self.present_faces)} نفر از قبل حاضر بودند")
            except:
                self.present_faces = {}

    def save_present_state(self):
        """ذخیره وضعیت حاضرین"""
        state_file = ATTENDANCE_DIR / 'present_state.pkl'
        with open(state_file, 'wb') as f:
            pickle.dump(self.present_faces, f)

    def load_today_log(self):
        """بارگذاری لاگ امروز"""
        today = datetime.datetime.now().strftime("%Y%m%d")
        log_file = ATTENDANCE_DIR / f'attendance_{today}.csv'

        if log_file.exists():
            try:
                df = pd.read_csv(log_file)
                for _, row in df.iterrows():
                    name = row['name']
                    entry_time = row['entry'] if pd.notna(row['entry']) else None
                    exit_time = row['exit'] if pd.notna(row['exit']) else None

                    self.attendance_log[name] = {
                        'entry': entry_time,
                        'exit': exit_time,
                        'date': row['date'] if 'date' in row else datetime.datetime.now().strftime("%Y-%m-%d")
                    }
                print(f"📂 لاگ امروز بارگذاری شد: {len(self.attendance_log)} نفر")
            except Exception as e:
                print(f"⚠️ خطا در بارگذاری لاگ: {e}")

    def save_attendance(self):
        """ذخیره گزارش حضور و غیاب"""
        today = datetime.datetime.now().strftime("%Y%m%d")
        filename = ATTENDANCE_DIR / f'attendance_{today}.csv'

        data = []
        for name, info in self.attendance_log.items():
            data.append({
                'name': name,
                'entry': info['entry'] if info['entry'] else '',
                'exit': info['exit'] if info['exit'] else '',
                'date': info['date']
            })

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False, encoding='utf-8-sig')

        # همچنین یه فایل کلی هم داشته باشیم
        all_log_file = ATTENDANCE_DIR / 'all_attendance.csv'
        if all_log_file.exists():
            all_df = pd.read_csv(all_log_file)
            all_df = pd.concat([all_df, df], ignore_index=True)
            all_df.drop_duplicates(subset=['name', 'date', 'entry'], keep='last', inplace=True)
        else:
            all_df = df
        all_df.to_csv(all_log_file, index=False, encoding='utf-8-sig')

        # ذخیره وضعیت حاضرین
        self.save_present_state()

        print(f"💾 گزارش حضور و غیاب ذخیره شد")

    def toggle_attendance(self, name):
        """تغییر وضعیت ورود/خروج"""
        current_time = datetime.datetime.now()
        current_time_str = current_time.strftime("%H:%M:%S")
        today = current_time.strftime("%Y-%m-%d")

        # اگه اسم توی present_faces هست یعنی الان حضور داره
        if name in self.present_faces:
            # خروج
            # پیدا کنیم آخرین ورودش کدوم بوده
            latest_entry = None
            latest_key = None
            for key, info in self.attendance_log.items():
                if key.startswith(name) and info['exit'] is None:
                    if latest_entry is None or info['entry'] > latest_entry:
                        latest_entry = info['entry']
                        latest_key = key

            if latest_key:
                self.attendance_log[latest_key]['exit'] = current_time_str
                del self.present_faces[name]
                message = f"{name} خارج شد"
                print(f"🚪 {message} - ساعت {current_time_str}")
                self.speak(message)
                self.save_attendance()
                return "خروج"
        else:
            # ورود جدید
            # ببینیم امروز قبلا ورود داشته؟
            today_entries = [k for k in self.attendance_log.keys()
                           if k.startswith(name) and self.attendance_log[k]['date'] == today]

            if today_entries:
                # ورود مجدد
                new_key = f"{name}_{len(today_entries)}"
                display_name = name
            else:
                # اولین ورود امروز
                new_key = name
                display_name = name

            self.attendance_log[new_key] = {
                'entry': current_time_str,
                'exit': None,
                'date': today
            }
            self.present_faces[name] = time.time()
            message = f"{display_name} وارد شد"
            print(f"✅ {message} - ساعت {current_time_str}")
            self.speak(message)
            self.save_attendance()
            return "ورود"

    def show_report(self):
        """نمایش گزارش کامل ورود و خروج‌ها"""
        print("\n" + "="*80)
        print("📊 گزارش حضور و غیاب")
        print("="*80)
        print(f"{'نام':<25} {'ورود':<15} {'خروج':<15} {'تاریخ':<15} {'وضعیت':<10}")
        print("-"*80)

        for name, info in sorted(self.attendance_log.items(), key=lambda x: x[1]['entry'] if x[1]['entry'] else ''):
            entry = info['entry'] if info['entry'] else '---'
            exit_time = info['exit'] if info['exit'] else '---'
            date = info['date']

            # تشخیص وضعیت
            base_name = name.split('_')[0]  # اسم اصلی بدون اندیس
            status = "حاضر" if base_name in self.present_faces and info['exit'] is None else "خارج شده"
            status_color = "✅" if status == "حاضر" else "⬅️"

            print(f"{name:<25} {entry:<15} {exit_time:<15} {date:<15} {status_color} {status:<10}")

        print("-"*80)

        # آمار
        total_entries = len(self.attendance_log)
        present_now = len(self.present_faces)
        completed = sum(1 for info in self.attendance_log.values() if info['exit'] is not None)

        print(f"\n📈 آمار:")
        print(f"   کل ورودها: {total_entries}")
        print(f"   افراد حاضر: {present_now}")
        print(f"   تکمیل شده (ورود و خروج): {completed}")

        if self.present_faces:
            print(f"\n👥 افراد حاضر در حال حاضر:")
            for name in self.present_faces:
                print(f"   ✅ {name}")

        input("\n🔹 Enter را بزنید...")

    def run(self):
        print("🚀 سیستم حضور و غیاب پیشرفته")
        print("="*60)
        print("✅ وضعیت افراد ذخیره می‌شود و با بستن برنامه از بین نمی‌رود")
        print("🔊 هر ورود و خروج با صدا اعلام می‌شود")
        print("🔄 کلید 't' برای تغییر دستی وضعیت")
        print("📊 کلید 'r' برای نمایش گزارش")
        print("❌ کلید 'q' برای خروج (وضعیت ذخیره می‌شود)")
        print("="*60)

        # تست صدا در شروع
        self.speak("سیستم حضور و غیاب فعال شد")

        cap = cv2.VideoCapture(0)
        last_toggle_time = {}
        last_announcement = {}  # برای جلوگیری از تکرار صدا

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces, gray = self.fu.detect_faces(frame)
            current_faces = []

            for (x, y, w, h) in faces:
                face_sample = self.fu.extract_face(gray, x, y, w, h)
                name, confidence = self.fu.recognize_face(face_sample)

                if name != "Unknown":
                    current_faces.append(name)

                    # ورود/خروج خودکار با فاصله زمانی مناسب
                    current_time = time.time()
                    if name not in last_toggle_time or current_time - last_toggle_time[name] > 3:
                        result = self.toggle_attendance(name)
                        last_toggle_time[name] = current_time

                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

                # نمایش با درصد اطمینان
                if name != "Unknown":
                    confidence_percent = max(0, min(100, 100 - (confidence / 100)))
                    display_text = f"{name} ({confidence_percent:.0f}%)"
                else:
                    display_text = "Unknown"

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, display_text, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # نمایش آمار روی تصویر
            y_pos = 30
            cv2.putText(frame, f"Present: {len(self.present_faces)}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_pos += 25
            cv2.putText(frame, f"Total Today: {len(self.attendance_log)}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # نمایش اسامی افراد حاضر
            if self.present_faces:
                y_pos += 25
                cv2.putText(frame, "Present:", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                present_list = list(self.present_faces.keys())
                for i, name in enumerate(present_list[:3]):
                    y_pos += 20
                    cv2.putText(frame, f"  ✅ {name}", (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            cv2.imshow('Attendance System', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.speak("سیستم حضور و غیاب بسته شد")
                self.save_present_state()
                print("💾 وضعیت حاضرین ذخیره شد")
                break
            elif key == ord('r'):
                self.show_report()
            elif key == ord('t'):
                # تغییر دستی وضعیت برای چهره اول
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    face_sample = self.fu.extract_face(gray, x, y, w, h)
                    name, _ = self.fu.recognize_face(face_sample)
                    if name != "Unknown":
                        self.toggle_attendance(name)

        cap.release()
        cv2.destroyAllWindows()
# ==================== ماژول 3: هشدار صوتی پیشرفته ====================
class VoiceAlertSystem:
    def __init__(self):
        self.fu = FaceUtils()
        self.last_alert = {}
        self.sound_enabled = False

        # تلاش برای نصب pyttsx3
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.sound_enabled = True
            print("✅ سیستم صوتی فعال شد")
        except ImportError:
            print("⚠️ pyttsx3 نصب نیست. برای نصب:")
            print("   pip install pyttsx3")
        except Exception as e:
            print(f"⚠️ خطا در راه‌اندازی صدا: {e}")

    def play_beep(self):
        """پخش صدای بوق برای افراد ناشناس"""
        try:
            import winsound
            winsound.Beep(1000, 500)  # فرکانس 1000، مدت 500 میلی‌ثانیه
        except:
            print("\a")  # بوق سیستمی

    def speak(self, message):
        if self.sound_enabled:
            try:
                self.engine.say(message)
                self.engine.runAndWait()
            except:
                print(f"🔊 {message}")

    def run(self):
        print("🚀 سیستم هشدار صوتی پیشرفته")
        print("✅ افراد شناسایی: پیام خوش‌آمد")
        print("⚠️ افراد ناشناس: صدای بوق")
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces, gray = self.fu.detect_faces(frame)

            for (x, y, w, h) in faces:
                face_sample = self.fu.extract_face(gray, x, y, w, h)
                name, _ = self.fu.recognize_face(face_sample)

                if name != "Unknown":
                    color = (0, 255, 0)
                    # پخش پیام خوش‌آمد
                    if name not in self.last_alert or time.time() - self.last_alert[name] > 10:
                        self.speak(f"سلام {name}")
                        self.last_alert[name] = time.time()
                else:
                    color = (0, 0, 255)
                    # پخش بوق برای ناشناس
                    current_time = time.time()
                    if 'unknown' not in self.last_alert or current_time - self.last_alert['unknown'] > 3:
                        self.play_beep()
                        self.last_alert['unknown'] = current_time
                        print("🚨 ناشناس شناسایی شد!")

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow('Voice Alert System', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# ==================== ماژول 4: تشخیص چند نفر ====================
class MultiFaceSystem:
    def __init__(self):
        self.fu = FaceUtils()

    def run(self):
        print("🚀 سیستم تشخیص چند نفر")
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces, gray = self.fu.detect_faces(frame)

            for (x, y, w, h) in faces:
                face_sample = self.fu.extract_face(gray, x, y, w, h)
                name, confidence = self.fu.recognize_face(face_sample)

                if name != "Unknown":
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # نمایش آمار
            known_count = sum(1 for (x, y, w, h) in faces if self.fu.recognize_face(
                self.fu.extract_face(gray, x, y, w, h))[0] != "Unknown")

            cv2.putText(frame, f"Total: {len(faces)} | Known: {known_count} | Unknown: {len(faces) - known_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Multi-Face System', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# ==================== ماژول 5: لاگین با چهره پیشرفته ====================
class FaceLoginSystem:
    def __init__(self):
        self.fu = FaceUtils()
        self.logged_in = False
        self.current_user = None
        self.login_logs = []
        self.load_login_logs()

    def load_login_logs(self):
        """بارگذاری لاگین‌های قبلی"""
        log_file = LOGIN_LOGS_DIR / 'login_history.csv'
        if log_file.exists():
            try:
                df = pd.read_csv(log_file)
                self.login_logs = df.to_dict('records')
                print(f"📂 تاریخچه لاگین بارگذاری شد: {len(self.login_logs)} ورود")
            except:
                print("📂 فایل تاریخچه لاگین خالی است")

    def save_login_log(self, name, status):
        """ذخیره لاگ لاگین"""
        log_entry = {
            'name': name,
            'status': status,  # 'success' یا 'failed'
            'datetime': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'date': datetime.datetime.now().strftime("%Y-%m-%d"),
            'time': datetime.datetime.now().strftime("%H:%M:%S")
        }
        self.login_logs.append(log_entry)

        # ذخیره در فایل CSV
        log_file = LOGIN_LOGS_DIR / 'login_history.csv'
        df = pd.DataFrame(self.login_logs)
        df.to_csv(log_file, index=False, encoding='utf-8-sig')

        # ذخیره روزانه هم جداگانه
        daily_file = LOGIN_LOGS_DIR / f"login_{datetime.datetime.now().strftime('%Y%m%d')}.csv"
        daily_df = pd.DataFrame([log_entry])
        if daily_file.exists():
            existing = pd.read_csv(daily_file)
            daily_df = pd.concat([existing, daily_df], ignore_index=True)
        daily_df.to_csv(daily_file, index=False, encoding='utf-8-sig')

        return log_entry

    def show_login_history(self):
        """نمایش تاریخچه لاگین"""
        print("\n📊 تاریخچه لاگین:")
        print("=" * 60)

        # آمار کلی
        total = len(self.login_logs)
        success = sum(1 for log in self.login_logs if log['status'] == 'success')
        failed = sum(1 for log in self.login_logs if log['status'] == 'failed')

        print(f"📈 کل تلاش‌ها: {total}")
        print(f"✅ موفق: {success}")
        print(f"❌ ناموفق: {failed}")
        print("-" * 60)

        # 10 لاگ آخر
        print("🕐 ۱۰ لاگ آخر:")
        for log in self.login_logs[-10:]:
            status_icon = "✅" if log['status'] == 'success' else "❌"
            print(f"   {status_icon} {log['datetime']} - {log['name']}")

        input("\n🔹 Enter را بزنید...")

    def run(self):
        print("🚀 سیستم لاگین با چهره پیشرفته")
        print("✅ لاگین موفق - ❌ لاگین ناموفق")
        print("📊 کلید 'h' برای نمایش تاریخچه")
        cap = cv2.VideoCapture(0)

        login_attempts = {}  # برای جلوگیری از لاگین مکرر

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces, gray = self.fu.detect_faces(frame)

            for (x, y, w, h) in faces:
                face_sample = self.fu.extract_face(gray, x, y, w, h)
                name, _ = self.fu.recognize_face(face_sample)

                if name != "Unknown":
                    # چک برای لاگین
                    if not self.logged_in:
                        current_time = time.time()
                        if name not in login_attempts or current_time - login_attempts[name] > 5:
                            self.logged_in = True
                            self.current_user = name
                            self.save_login_log(name, 'success')
                            print(f"✅ {name} با موفقیت وارد شد")
                            login_attempts[name] = current_time

                    color = (0, 255, 0) if name == self.current_user else (255, 255, 0)
                else:
                    color = (0, 0, 255)
                    # ثبت تلاش ناموفق برای ناشناس
                    if not self.logged_in:
                        current_time = time.time()
                        if 'unknown' not in login_attempts or current_time - login_attempts['unknown'] > 3:
                            self.save_login_log('Unknown', 'failed')
                            print("❌ تلاش ناموفق برای ورود")
                            login_attempts['unknown'] = current_time

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # نمایش وضعیت
            status = f"✅ {self.current_user}" if self.logged_in else "❌ Not logged in"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if self.logged_in else (0, 0, 255), 2)

            # نمایش آمار امروز
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            today_logins = sum(1 for log in self.login_logs
                               if log['date'] == today and log['status'] == 'success')
            cv2.putText(frame, f"Today: {today_logins} logins", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('Face Login System', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                # ثبت خروج
                if self.logged_in:
                    self.save_login_log(self.current_user, 'logout')
                break
            elif key == ord('h'):
                self.show_login_history()
            elif key == ord('l'):
                # خروج دستی
                if self.logged_in:
                    self.save_login_log(self.current_user, 'logout')
                    self.logged_in = False
                    self.current_user = None
                    print("🚪 خروج دستی از سیستم")

        cap.release()
        cv2.destroyAllWindows()


# ==================== ماژول جدید: People Counter ====================
class PeopleCounter:
    def __init__(self):
        self.fu = FaceUtils()
        self.total_count = 0
        self.entry_count = 0
        self.exit_count = 0
        self.tracked_people = {}  # {id: {'name': name, 'last_seen': time, 'counted': bool}}
        self.next_id = 0
        self.counter_log = []
        self.load_counter_data()

        # خط ورود و خروج (مثل فروشگاه)
        self.line_position = 0.5  # وسط تصویر
        self.entry_line_color = (0, 255, 0)  # سبز
        self.exit_line_color = (0, 0, 255)  # قرمز

    def load_counter_data(self):
        """بارگذاری آمار قبلی"""
        counter_file = Path('data') / 'counter_stats.pkl'
        if counter_file.exists():
            try:
                with open(counter_file, 'rb') as f:
                    data = pickle.load(f)
                    self.total_count = data.get('total', 0)
                    self.entry_count = data.get('entry', 0)
                    self.exit_count = data.get('exit', 0)
                print(f"📊 آمار قبلی: {self.total_count} نفر")
            except:
                pass

    def save_counter_data(self):
        """ذخیره آمار"""
        counter_file = Path('data') / 'counter_stats.pkl'
        with open(counter_file, 'wb') as f:
            pickle.dump({
                'total': self.total_count,
                'entry': self.entry_count,
                'exit': self.exit_count
            }, f)

    def check_crossing(self, old_y, new_y, frame_height):
        """بررسی عبور از خط"""
        line_y = int(frame_height * self.line_position)

        # اگه از بالا به پایین رد شد (ورود)
        if old_y < line_y and new_y >= line_y:
            return "entry"
        # اگه از پایین به بالا رد شد (خروج)
        elif old_y > line_y and new_y <= line_y:
            return "exit"
        return None

    def log_event(self, event_type, name="Unknown"):
        """ثبت رویداد"""
        time_str = datetime.datetime.now().strftime("%H:%M:%S")
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")

        log_entry = {
            'time': time_str,
            'date': date_str,
            'type': event_type,
            'name': name
        }
        self.counter_log.append(log_entry)

        # ذخیره در فایل
        log_file = Path('data') / f"counter_log_{date_str}.csv"
        df = pd.DataFrame(self.counter_log)
        df.to_csv(log_file, index=False, encoding='utf-8-sig')

        # اعلام صوتی
        if event_type == "entry":
            if name != "Unknown":
                print(f"🔊 {name} وارد شد")
            else:
                print("🔊 یک نفر وارد شد")
        else:
            if name != "Unknown":
                print(f"🔊 {name} خارج شد")
            else:
                print("🔊 یک نفر خارج شد")

    def show_report(self):
        """نمایش گزارش"""
        print("\n" + "=" * 60)
        print("📊 آمار تردد")
        print("=" * 60)
        print(f"📈 کل تردد: {self.total_count}")
        print(f"✅ ورود: {self.entry_count}")
        print(f"❌ خروج: {self.exit_count}")
        print(f"👥 حاضرین: {self.entry_count - self.exit_count}")
        print("-" * 60)

        # ۱۰ تردد آخر
        print("🕐 آخرین ترددها:")
        for log in self.counter_log[-10:]:
            icon = "✅" if log['type'] == "entry" else "❌"
            print(f"   {icon} {log['time']} - {log['name']}")

        input("\n🔹 Enter را بزنید...")

    def run(self):
        print("🚀 سیستم شمارشگر افراد (People Counter)")
        print("=" * 60)
        print("✅ خط سبز = ورود | خط قرمز = خروج")
        print("📊 کلید 'r' = گزارش")
        print("🔄 کلید '+' و '-' = تنظیم خط")
        print("❌ کلید 'q' = خروج")
        print("=" * 60)

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_height, frame_width = frame.shape[:2]
            line_y = int(frame_height * self.line_position)

            faces, gray = self.fu.detect_faces(frame)
            current_people = []

            for (x, y, w, h) in faces:
                # مرکز صورت
                center_y = y + h // 2
                center_x = x + w // 2

                face_sample = self.fu.extract_face(gray, x, y, w, h)
                name, _ = self.fu.recognize_face(face_sample)

                # بررسی آیا این فرد قبلاً دیده شده؟
                found = False
                for pid, info in self.tracked_people.items():
                    if abs(info['last_x'] - center_x) < 100 and abs(info['last_y'] - center_y) < 100:
                        # فرد قبلی
                        found = True
                        crossing = self.check_crossing(info['last_y'], center_y, frame_height)

                        if crossing and not info['counted']:
                            if crossing == "entry":
                                self.total_count += 1
                                self.entry_count += 1
                                self.log_event("entry", name)
                            else:
                                self.total_count += 1
                                self.exit_count += 1
                                self.log_event("exit", name)
                            self.tracked_people[pid]['counted'] = True

                        self.tracked_people[pid]['last_y'] = center_y
                        self.tracked_people[pid]['last_x'] = center_x
                        self.tracked_people[pid]['last_seen'] = time.time()
                        break

                if not found:
                    # فرد جدید
                    self.tracked_people[self.next_id] = {
                        'name': name,
                        'last_y': center_y,
                        'last_x': center_x,
                        'last_seen': time.time(),
                        'counted': False
                    }
                    self.next_id += 1

                # نمایش با رنگ بر اساس شناخته شده یا نه
                if name != "Unknown":
                    color = (0, 255, 0)
                else:
                    color = (0, 255, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # نمایش نقطه مرکزی
                cv2.circle(frame, (center_x, center_y), 4, color, -1)

            # پاک کردن افرادی که دیگه نیستن
            current_time = time.time()
            self.tracked_people = {pid: info for pid, info in self.tracked_people.items()
                                   if current_time - info['last_seen'] < 2}

            # کشیدن خطوط ورود و خروج
            cv2.line(frame, (0, line_y), (frame_width, line_y), (0, 255, 0), 2)
            cv2.putText(frame, "Entry", (10, line_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # نمایش آمار روی تصویر
            y_pos = 30
            cv2.putText(frame, f"Total: {self.total_count}", (frame_width - 200, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 25
            cv2.putText(frame, f"Entry: {self.entry_count}", (frame_width - 200, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            y_pos += 25
            cv2.putText(frame, f"Exit: {self.exit_count}", (frame_width - 200, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            y_pos += 25
            cv2.putText(frame, f"Inside: {self.entry_count - self.exit_count}", (frame_width - 200, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            cv2.imshow('People Counter', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.save_counter_data()
                break
            elif key == ord('r'):
                self.show_report()
            elif key == ord('+'):
                self.line_position = min(1.0, self.line_position + 0.05)
            elif key == ord('-'):
                self.line_position = max(0.0, self.line_position - 0.05)

        cap.release()
        cv2.destroyAllWindows()

        # ==================== ماژول 7: تشخیص ژست دست ====================


# ==================== ماژول 7: تشخیص ژست دست ====================
# ==================== ماژول 7: تشخیص ژست دست (ساده) ====================
# ==================== ماژول 7: تشخیص ژست دست با MediaPipe ====================
# ==================== ماژول 7: تشخیص ژست دست با MediaPipe (نسخه جدید) ====================
class HandGestureSystem:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.setup_mediapipe()
        self.last_gesture_time = time.time()

    def setup_mediapipe(self):
        """راه‌اندازی MediaPipe برای تشخیص دست"""
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mediapipe_available = True
            print("✅ MediaPipe آماده است")
        except ImportError:
            print("❌ MediaPipe نصب نیست. نصب کن:")
            print("pip install mediapipe")
            self.mediapipe_available = False
        except Exception as e:
            print(f"❌ خطا: {e}")
            self.mediapipe_available = False

    def count_fingers(self, hand_landmarks):
        """شمارش انگشتان"""
        if not hand_landmarks:
            return 0

        finger_count = 0
        h, w = self.frame_shape

        # نوک انگشتان
        fingertips = [4, 8, 12, 16, 20]

        for tip in fingertips:
            x = int(hand_landmarks.landmark[tip].x * w)
            y = int(hand_landmarks.landmark[tip].y * h)

            # نقطه پایینی انگشت
            if tip == 4:  # شست
                x2 = int(hand_landmarks.landmark[3].x * w)
                y2 = int(hand_landmarks.landmark[3].y * h)
                if x < x2:  # شست باز
                    finger_count += 1
            else:
                x2 = int(hand_landmarks.landmark[tip - 2].x * w)
                y2 = int(hand_landmarks.landmark[tip - 2].y * h)
                if y < y2:  # انگشت باز
                    finger_count += 1

        return finger_count

    def perform_action(self, finger_count):
        """انجام کار بر اساس تعداد انگشتان"""
        current_time = time.time()
        if current_time - self.last_gesture_time < 2:
            return

        if finger_count == 1:
            print("🔇 1 انگشت - قطع صدا")
            import os
            os.system("nircmd mutesysvolume 1")  # نیاز به nircmd داره

        elif finger_count == 2:
            print("🔉 2 انگشت - کم کردن صدا")
            import os
            os.system("nircmd changesysvolume -2000")

        elif finger_count == 3:
            print("🔊 3 انگشت - زیاد کردن صدا")
            import os
            os.system("nircmd changesysvolume +2000")

        elif finger_count == 4:
            print("📸 4 انگشت - عکس گرفتن")
            ret, frame = self.cap.read()
            if ret:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"gesture_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"✅ عکس ذخیره شد: {filename}")

        elif finger_count == 5:
            print("📋 5 انگشت - باز کردن منو")

        self.last_gesture_time = current_time

    def run(self):
        if not self.mediapipe_available:
            print("❌ MediaPipe در دسترس نیست")
            return

        print("🚀 تشخیص ژست دست")
        print("=" * 50)
        print("🖐️ انگشتات رو نشون بده:")
        print("   1 = قطع صدا")
        print("   2 = کم کردن صدا")
        print("   3 = زیاد کردن صدا")
        print("   4 = عکس گرفتن")
        print("   5 = هیچی")
        print("-" * 50)
        print("❌ کلید 'q' برای خروج")
        print("=" * 50)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            self.frame_shape = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            finger_count = 0

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # رسم نقاط دست
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )

                    # شمارش انگشتان
                    finger_count = self.count_fingers(hand_landmarks)

                    # نمایش تعداد
                    cv2.putText(frame, f"Fingers: {finger_count}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # انجام کار
                    self.perform_action(finger_count)

            cv2.imshow('Hand Gesture', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
# ==================== ماژول 9: تشخیص اشیاء با YOLO ====================
# ==================== ماژول 9: تشخیص اشیاء ====================
# ==================== ماژول 9: تشخیص اشیاء (ساده) ====================
class ObjectDetectionSystem:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        # فقط تشخیص چهره به عنوان person
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def run(self):
        print("🚀 تشخیص اشیاء (ساده)")
        print("=" * 60)
        print("📦 فقط 'person' تشخیص داده می‌شود")
        print("❌ کلید 'q' برای خروج")
        print("=" * 60)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "person", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(frame, f"People: {len(faces)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Object Detection (Simple)', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
# ==================== ماژول 11: تشخیص خواب‌آلودگی راننده ====================
# ==================== ماژول 11: تشخیص خواب‌آلودگی ====================
# ==================== ماژول 11: تشخیص خواب‌آلودگی ====================
# ==================== ماژول 11: تشخیص خواب‌آلودگی دقیق ====================
# ==================== ماژول 11: تشخیص خواب‌آلودگی (ساده و تضمینی) ====================
class DrowsinessSystem:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        self.eyes_closed_count = 0
        self.alarm_on = False

    def play_beep(self):
        """پخش صدای بوق"""
        try:
            import winsound
            winsound.Beep(1000, 500)
        except:
            print("\a")

    def run(self):
        print("🚀 تشخیص خواب‌آلودگی (ساده)")
        print("=" * 50)
        print("😴 چشمات رو ببند تا بوق بزنم!")
        print("❌ کلید 'q' برای خروج")
        print("=" * 50)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            eyes_found = False

            for (x, y, w, h) in faces:
                # رسم مستطیل دور صورت
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # تشخیص چشم‌ها
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)

                if len(eyes) >= 2:
                    eyes_found = True
                    # رسم مستطیل دور چشم‌ها
                    for (ex, ey, ew, eh) in eyes[:2]:
                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            # بررسی وضعیت چشم‌ها
            if len(faces) > 0 and not eyes_found:
                self.eyes_closed_count += 1
                cv2.putText(frame, "چشم‌ها بسته!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.eyes_closed_count = max(0, self.eyes_closed_count - 1)
                cv2.putText(frame, "چشم‌ها باز", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # نمایش شمارنده
            cv2.putText(frame, f"Closed count: {self.eyes_closed_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # هشدار اگه چشم‌ها بسته باشن
            if self.eyes_closed_count > 10:
                cv2.putText(frame, "⚠️ خواب آلودگی! ⚠️", (frame.shape[1] // 2 - 100, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                self.play_beep()
                self.eyes_closed_count = 5  # ریست نسبی

            cv2.imshow('Drowsiness Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

# ==================== منوی اصلی ====================
def main():
    fu = FaceUtils()

    while True:
        print("\n" + "=" * 70)
        print("🚀 سوپرمنوی پروژه‌های بینایی کامپیوتر")
        print("=" * 70)
        print("📌 پروژه‌های اصلی (تشخیص چهره):")
        print("1️⃣  ذخیره خودکار تصاویر ناشناس")
        print("2️⃣  حضور و غیاب پیشرفته (ورود/خروج خودکار)")
        print("3️⃣  هشدار صوتی (بوق برای ناشناس)")
        print("4️⃣  تشخیص چند نفر همزمان")
        print("5️⃣  لاگین با چهره (با تاریخچه)")
        print("6️⃣  شمارشگر افراد (People Counter) - فروشگاهی")
        print("7️⃣  تشخیص ژست دست (حالت ماینوریتی ریپورت)")
        print("9️⃣  تشخیص اشیاء (YOLO - ماشین، آدم، گربه)")
        print("1️⃣1️⃣ تشخیص خواب‌آلودگی راننده")
        print("-" * 70)
        print("📌 پروژه‌های جدید:")
        print("8️⃣  اسکنر اسناد (مثل Adobe Scan)")
        print("🔟  فیلتر اینستاگرامی (عینک، تاج، کلاه)")
        print("1️⃣2️⃣ نقاش هوش مصنوعی (نقاشی با انگشت)")
        print("-" * 70)
        print("0️⃣  جمع‌آوری نمونه چهره جدید")
        print("🚪  q  خروج")
        print("=" * 70)

        choice = input("\n🔹 انتخاب شما: ").strip()

        if choice == '1':
            system = AutoSaveSystem()
            system.run()
        elif choice == '2':
            system = AttendanceSystem()
            system.run()
        elif choice == '3':
            system = VoiceAlertSystem()
            system.run()
        elif choice == '4':
            system = MultiFaceSystem()
            system.run()
        elif choice == '5':
            system = FaceLoginSystem()
            system.run()
        elif choice == '6':
            print("\n🚀 شمارشگر افراد (People Counter)")
            print("این پروژه تعداد افرادی که از جلوی دوربین رد میشن رو میشماره")
            print("مثل فروشگاه‌ها که میخوان بدونن چند نفر اومدن")
            system = PeopleCounter()
            system.run()
        elif choice == '7':
            print("\n🚀 تشخیص ژست دست (Hand Gesture Recognition)")
            print("با نشان دادن تعداد انگشتات، کارهای مختلف انجام بده")
            print("مثلاً ۲ انگشت = قطع صدا، ۳ انگشت = عکس گرفتن")
            system = HandGestureSystem()
            system.run()
        elif choice == '9':
            print("\n🚀 تشخیص اشیاء (Object Detection)")
            print("تشخیص ماشین، آدم، گربه، موبایل، بطری و ...")
            print("با YOLO - همون تکنولوژی ماشین‌های خودران")
            system = ObjectDetectionSystem()
            system.run()

        elif choice == '11':
            print("\n🚀 تشخیص خواب‌آلودگی راننده")
            print("اگه چشمات بسته باشه، بوق میزنه بیدارت کنه")
            print("پروژه‌ای که جون آدما رو نجات میده!")
            system = DrowsinessSystem()
            system.run()
            print("🚧 در حال توسعه...")
            input("🔹 Enter را بزنید...")
        elif choice == '0':
            fu.collect_samples()
        elif choice.lower() == 'q':
            print("\n🙏 از استفاده شما متشکریم. خدانگهدار!")
            break
        else:
            print("\n❌ انتخاب نامعتبر!")
            input("🔹 Enter را بزنید...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🚪 برنامه با موفقیت بسته شد")