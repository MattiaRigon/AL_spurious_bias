import os
from PIL import Image
from lang_sam import LangSAM
import numpy as np
import hydra
from omegaconf import DictConfig
from hydra.core.global_hydra import GlobalHydra

model = LangSAM()
image_pil = Image.open("data/waterbirds_v1.0/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg").convert("RGB")
text_prompt = "bird."
results = model.predict([image_pil], [text_prompt])

output_dir = "./output/"
os.makedirs(output_dir, exist_ok=True)

for idx, result in enumerate(results):
    # Salvataggio delle maschere come immagini
    for mask_idx, mask in enumerate(result["masks"]):

        mask_img = Image.fromarray((mask * 255).astype(np.uint8))  # Convertire in scala di grigi 8-bit
        mask_path = f"{output_dir}/mask_{idx}_{mask_idx}.png"
        mask_img.save(mask_path)