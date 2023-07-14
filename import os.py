import os
import torch
from torch.utils.data import Dataset, DataLoader

#/home/ec22156/MDB-audio/3_audio
os.environ["CUDA_VISIBLE_DEVICES"] = "1]"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on device: {device}")