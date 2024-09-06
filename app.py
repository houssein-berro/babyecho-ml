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



if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="192.168.0.101", port=50314, debug=True)
