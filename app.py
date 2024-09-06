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


