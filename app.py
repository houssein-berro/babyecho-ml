import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import shutil
import librosa
import numpy as np
import tensorflow as tf
from pydub import AudioSegment
import csv
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust based on your setup
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = 'upload'
CSV_FOLDER = 'features/csv'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(('.mp3', '.wav', '.m4a', '.mp4')):
        raise HTTPException(status_code=400, detail="Invalid file format. Only audio files are allowed.")
    print(file.filename)
    if file.filename == '':
        raise HTTPException(status_code=400, detail="No selected file")
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    prediction = process_audio(file.filename)
    clear_csv(os.path.join(CSV_FOLDER, 'dataset.csv'))
    return JSONResponse(content={"prediction": prediction[0]}, status_code=200)

def add_uploaded_to_dataset(filename):
    with open(os.path.join(CSV_FOLDER, 'dataset.csv'), 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([filename])
    shutil.copy(os.path.join(UPLOAD_FOLDER, filename), os.path.join(CSV_FOLDER, filename))

def convert_to_wav(file_path):
    try:
        print(f"Attempting to convert file at: {file_path}")
        
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
        
        audio = AudioSegment.from_file(file_path)
        wav_path = file_path.replace(file_path.split('.')[-1], 'wav')
        audio.export(wav_path, format="wav")
        
        print(f"File converted to: {wav_path}")
        return wav_path
    except Exception as e:
        print(f"Error converting file {file_path} to WAV: {e}")
        return None

# Method to check if audio contains silence
def check_silence(file_path, silence_threshold=0.01):
    try:
        y, sr = librosa.load(file_path, sr=None)
        rms = librosa.feature.rms(y=y)  # Calculate Root Mean Square (RMS) energy
        avg_rms = np.mean(rms)
        print(f"RMS energy for {file_path}: {avg_rms}")

        if avg_rms < silence_threshold:
            return True  # Audio is silent
        else:
            return False  # Audio has sound
    except Exception as e:
        print(f"Error checking silence in {file_path}: {e}")
        return True  # Treat as silent in case of error

def convert_to_spectrogram(file_path):
    try:
        x, sr = librosa.load(file_path, sr=None)
        spectrogram = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=128)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        spectrogram = np.expand_dims(spectrogram, axis=-1)  # Add channel dimension
        
        # Resize the spectrogram to (128, 128, 1)
        target_shape = (128, 128)
        if spectrogram.shape[1] < target_shape[1]:
            # Pad with zeros if the width is less than 128
            padding = target_shape[1] - spectrogram.shape[1]
            spectrogram = np.pad(spectrogram, ((0, 0), (0, padding), (0, 0)), mode='constant')
        elif spectrogram.shape[1] > target_shape[1]:
            # Crop the spectrogram if the width is more than 128
            spectrogram = spectrogram[:, :target_shape[1], :]
        
        return spectrogram
    except Exception as e:
        print(f"Error converting {file_path} to spectrogram: {e}")
        return None

def process_audio(filename):
    file_path = os.path.join('upload', filename)
    wav_path = convert_to_wav(file_path)
    
    if wav_path is None:
        return ["Error: Could not convert to WAV"]
    
    # Check if the audio is silent
    if check_silence(wav_path):
        return ["Error: The audio is silent"]

    spectrogram = convert_to_spectrogram(wav_path)
    
    if spectrogram is None:
        return ["Error: Could not convert to spectrogram"]
    
    # Check if the spectrogram matches the expected shape
    if spectrogram.shape != (128, 128, 1):
        return [f"Error: Spectrogram shape {spectrogram.shape} is not valid"]

    # Ensure input shape is (1, 128, 128, 1) for batch processing
    spectrogram = np.expand_dims(spectrogram, axis=0)
    
    result = predict(spectrogram)

    # Delete the .wav file after processing
    try:
        os.remove(wav_path)
        print(f"Deleted file: {wav_path}")
    except Exception as e:
        print(f"Error deleting file {wav_path}: {e}")
    
    return result

def normalize(row):
    min_values = []
    max_values = []
    csv1 = os.path.join(CSV_FOLDER, 'min_max_values.csv')
    
    if not os.path.exists(csv1):
        with open(csv1, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['feature', 'min_value', 'max_value'])
            for i in range(len(row)):
                min_val = -100.0
                max_val = 100.0
                csv_writer.writerow([f'mfcc_{i+1}', min_val, max_val])
                min_values.append(min_val)
                max_values.append(max_val)
    else:
        with open(csv1, 'r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)
            for line in csv_reader:
                min_val = float(line[1])
                max_val = float(line[2])
                min_values.append(min_val)
                max_values.append(max_val)

    row1 = []
    for i in range(len(row)):
        normalized_val = (row[i] - min_values[i]) / (max_values[i] - min_values[i])
        row1.append(normalized_val)
    
    return row1

def decode(labels):
    label_map = {1: 'belly_pain', 2: 'burping', 3: 'discomfort', 4: 'hungry', 5: 'tired'}
    return [label_map.get(label, 'unknown') for label in labels]

def predict(spectrogram):
    loaded_model = tf.keras.models.load_model('features/modelV3.keras')
    predictions = loaded_model.predict(spectrogram)
    predicted_labels = np.argmax(predictions, axis=1) + 1
    return decode(predicted_labels)

def clear_csv(file_path):
    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
        print(f"Cleared all rows in {file_path}.")
    except Exception as e:
        print(f"Error clearing rows in {file_path}: {e}")
        raise

def clear_csv_except_header(file_path):
    try:
        df = pd.read_csv(file_path)
        df.iloc[0:1].to_csv(file_path, index=False)
        print(f"Cleared all rows in {file_path} except for the header.")
    except Exception as e:
        print(f"Error clearing rows in {file_path}: {e}")
        raise
 
if __name__ == '__main__': 
    import uvicorn 
    uvicorn.run(app, host="192.168.0.101", port=50314, debug=True) 
