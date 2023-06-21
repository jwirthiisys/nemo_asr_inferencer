from models import ConformerTransducer
import torch
import logging
import time
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("--files", metavar='N', type=str)

args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"


model = ConformerTransducer(device)

input_files = Path(args.files).glob("*.wav")
model_inputs = [str(p) for p in input_files]



logging.info(f"--- inferencing started ---")
start_time = time.time()
outputs = model.batch_inference(model_inputs)
end_time = time.time()
logging.info(f"--- inferencing finished. Time elapsed: {end_time - start_time}s ---")


with open(f"outputs.txt", "w") as f:
    for model_input, output in zip(model_inputs, outputs):
        f.write(f"{model_input}:\t{output}\n")