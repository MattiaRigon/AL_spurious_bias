import os
from PIL import Image
from lang_sam import LangSAM
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":

    model = LangSAM()

    dataset_root = "data"
    masks_root = "masks"
    text_prompt = "bird."

    dataset_folder = "waterbirds_v1.0"
    dataset_explanations_folder = os.path.join(masks_root,f"{dataset_folder}")
    dataset_path = os.path.join(dataset_root, dataset_folder)

    all_folders = os.listdir(dataset_path)
    all_folders = [folder for folder in all_folders if os.path.isdir(os.path.join(dataset_path, folder))]
    num = 0
    for folder in tqdm(all_folders, desc="Processing folders"):
        folder_path = os.path.join(dataset_path, folder)
        save_folder = os.path.join(dataset_explanations_folder, folder)
        os.makedirs(save_folder, exist_ok=True)
        images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
        num += len(images)
        for image in images:
            if os.path.exists(os.path.join(save_folder, image)):
                continue
            image_pil = Image.open(os.path.join(folder_path,image)).convert("RGB")
            results = model.predict([image_pil], [text_prompt])
            os.makedirs(dataset_explanations_folder, exist_ok=True)
            combined_mask = None
            for idx, result in enumerate(results):
                for mask_idx, mask in enumerate(result["masks"]):
                    if combined_mask is None:
                        combined_mask = mask
                    else:
                        combined_mask = np.logical_or(combined_mask, mask)

            if combined_mask is not None:
                combined_mask_img = Image.fromarray((combined_mask * 255).astype(np.uint8))
                
                combined_mask_path = os.path.join(save_folder, image)
                combined_mask_img.save(combined_mask_path)
            else:
                print(f"No mask found for {image}")
                black_image = Image.new("RGB", image_pil.size, (0, 0, 0))
                black_image.save(os.path.join(save_folder, image))