import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pyttsx3
import time

# --- 1. VOICE FUNCTION ---
# వాయిస్ హ్యాంగ్ అవ్వకుండా ఉండటానికి ఈ ఫంక్షన్ వాడతాం
def speak(text_eng, text_tel):
    print(f"🔊 Speaking: {text_eng} | {text_tel}")
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        # ఇంగ్లీష్ మరియు తెలుగు మిశ్రమం
        full_text = text_eng + " . . . " + text_tel
        engine.say(full_text)
        engine.runAndWait()
        engine.stop() 
    except Exception as e:
        print("Voice Error:", e)

# --- 2. LOAD MODEL & LABELS ---
# ఫైల్ పేర్లు కరెక్ట్ గా ఉన్నాయో లేదో చూసుకో
model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# తెలుగు పేర్ల మ్యాపింగ్
tel_map = {"10":"padi", "20":"iravai", "50":"yabhai", "100":"vanda", "200":"rendu vandala", "500":"aidu vandala"}

# --- 3. START CAMERA ---
# ఒకవేళ కెమెరా రాకపోతే 0 ని 1 గా మార్చు
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
last_speech_time = 0
last_label = ""

# వెల్కమ్ మెసేజ్
speak("Welcome to Drishti", "Namaskaram")

while True:
    ret, frame = cap.read()
    if not ret: break

    # --- 4. IMAGE PREPROCESSING ---
    # రంగులను మార్చడం చాలా ముఖ్యం (BGR to RGB)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
    img = (img / 127.5) - 1.0 

    # --- 5. PREDICTION ---
    prediction = model.predict(img, verbose=0)
    index = np.argmax(prediction)
    
    # ఎర్రర్ ఫిక్స్: prediction[0][index] అని వాడాలి
    confidence = float(prediction[0][index])
    
    # Label Cleaning (నంబర్లు తీసేయడానికి)
    raw_label = class_names[index].strip()
    clean_label = raw_label.split(' ', 1)[-1] 
    amount = "".join(filter(str.isdigit, clean_label))
    
    current_time = time.time()

    # --- 6. DETECTION LOGIC ---
    # 85% పైన కచ్చితత్వం ఉంటేనే మాట్లాడాలి
    if confidence > 0.85:
        if (clean_label != last_label) or (current_time - last_speech_time > 5):
            if "Background" in clean_label or "Nothing" in clean_label:
                # ఏమీ లేనప్పుడు 8 సెకన్ల గ్యాప్ తో చెప్పాలి
                if current_time - last_speech_time > 8:
                    speak("No note detected", "Emee ledhu")
                    last_speech_time = current_time
                    last_label = clean_label
            else:
                # కరెన్సీ నోటును గుర్తించినప్పుడు
                tel_word = tel_map.get(amount, amount)
                speak(f"This is {amount} rupees", f"Idhee {tel_word} rupayalu")
                last_speech_time = current_time
                last_label = clean_label
    
    # పొజిషన్ సరిగ్గా లేకపోతే (Move to Center)
    elif 0.40 < confidence < 0.85:
        if current_time - last_speech_time > 7:
            speak("Move to center", "Madhyaloki jarupandi")
            last_speech_time = current_time

    # --- 7. DISPLAY ---
    cv2.putText(frame, f"Detected: {clean_label} ({int(confidence*100)}%)", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Drishti Camera Output", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# క్లోజింగ్
cap.release()
cv2.destroyAllWindows()