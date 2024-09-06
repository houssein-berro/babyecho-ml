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



if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="192.168.0.101", port=50314, debug=True)
