import numpy as np
import sounddevice as sd
import librosa
import time
import pickle
import tensorflow as tf
tflite = tf.lite
import paho.mqtt.client as mqtt
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# === CONFIGURACIÃN ===
INPUT_SAMPLE_RATE = 48000  # Frecuencia real del micrófono USB
SAMPLE_RATE = 16000        # Lo que espera el modelo
SEGMENT_DURATION = 5       # En segundos
DEVICE_INDEX = 1           # KT USB Audio
MACHINE_ID_STR = "id_08"
MQTT_BROKER = "localhost"  # O IP local
MQTT_TOPIC = "logs/app"

# === CARGAR MODELO Y DATOS ===
interpreter = tflite.Interpreter(model_path="encoder2.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

emb_train = np.load("emb_train2.npy")
ids_train = np.load("ids_train2.npy")
threshold = np.load("threshold2.npy")

with open("id_to_int2.pkl", "rb") as f:
    id_to_int = pickle.load(f)

machine_idx = id_to_int[MACHINE_ID_STR]
train_subset = emb_train[ids_train == machine_idx]

# === FUNCIONES ===


def apply_pre_emphasis(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def convert_to_spectrogram(segment):
    segment = apply_pre_emphasis(segment)
    melspec = librosa.feature.melspectrogram(
        y=segment, sr=SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=128)
    melspec_db = librosa.power_to_db(melspec)
    spec_min, spec_max = melspec_db.min(), melspec_db.max()
    norm_spec = (melspec_db - spec_min) / (spec_max - spec_min + 1e-8)
    resized = np.zeros((128, 128))
    h, w = norm_spec.shape
    resized[:min(128, h), :min(128, w)] = norm_spec[:min(128, h), :min(128, w)]
    return resized[np.newaxis, ..., np.newaxis].astype(np.float32)


def run_inference(spec, machine_id):
    for input in input_details:
        shape = list(input['shape'])
        if shape == [1, 1]:
            interpreter.set_tensor(input['index'], np.array(
                [[machine_id]], dtype=np.float32))

        elif shape == [1, 128, 128, 1]:
            interpreter.set_tensor(input['index'], spec.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]


def compute_min_distance(embedding, reference_embeddings):
    return np.min(np.linalg.norm(reference_embeddings - embedding, axis=1))


# === MQTT CLIENT ===
mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, 1883)
mqtt_client.loop_start()

# === LOOP PRINCIPAL ===
# Mostrar dispositivos de entrada
print("\nDispositivos disponibles:")
devices = sd.query_devices()
input_devices = [i for i, d in enumerate(devices) if d['max_input_channels'] > 0]
for i in input_devices:
    print(f"{i}: {devices[i]['name']}")

# Selección de micrófono
sd.default.device = int(input("\nSelecciona el número del micrófono a usar: "))

print("? Escuchando audio en tiempo real...")

while True:
    print("? Grabando segmento...")
    audio = sd.rec(int(INPUT_SAMPLE_RATE * SEGMENT_DURATION),
                   samplerate=INPUT_SAMPLE_RATE,
                   channels=1)
    sd.wait()
    segment = audio.flatten()

    # Resample de 48000 ? 16000
    segment = librosa.resample(
        segment, orig_sr=INPUT_SAMPLE_RATE, target_sr=SAMPLE_RATE)

    print(f"Amplitud media: {np.mean(np.abs(segment)):.5f}")
    if np.mean(np.abs(segment)) < 0.0002:
        print("? Segmento silencioso. Saltando.")
        time.sleep(1)
        continue

    try:
        spectrogram = convert_to_spectrogram(segment)
        embedding = run_inference(spectrogram, machine_idx)
        distance = compute_min_distance(embedding, train_subset)

        estado = "anomalo" if distance > threshold else "normal"
        # Emit timestamp in RFC3339 format with 'Z' (UTC) and no microseconds for Telegraf compatibility
        timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

        # Emit log as a valid JSON string (not Python dict string)
        payload = {
            "timestamp": timestamp,
            "machine_id": MACHINE_ID_STR,
            "estado": estado,
            "distancia": float(distance)
        }
        import json
        mqtt_client.publish(MQTT_TOPIC, json.dumps(payload))
        print(f"? Publicado en MQTT: {json.dumps(payload)}")

    except Exception as e:
        print(f"? Error durante la inferencia: {e}")

    time.sleep(1)
