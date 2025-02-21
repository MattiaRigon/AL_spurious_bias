import os
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


def get_explanations(data, dataset_name, model_explanation):
    if dataset_name == "waterbirds":
        return get_masks(data, "bird.", model_explanation)
        


def get_masks(data, prompt, segmentation_model):
    """
        Function that given a dataset of images and the prompt, for each image in the dataset
        it will generate a mask that will be used as explanation in the rrr loss.
        The mask will be generated using the prompt and a segmentation model.
    """

    
    images_pil = []
    masks = []
    for i,image_tensor in enumerate(data):
        image_np = denormalize_image(image_tensor)
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
        images_pil.append(image_pil)

        results = segmentation_model.predict([image_pil], [prompt])

        output_dir = "./output/"
        os.makedirs(output_dir, exist_ok=True)
        _mask = torch.zeros(image_tensor.shape[1], image_tensor.shape[2], dtype=torch.uint8)
        for idx, result in enumerate(results):
            for mask_idx, mask in enumerate(result["masks"]):
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))  # Convertire in scala di grigi 8-bit
                mask_path = f"{output_dir}/mask{i}.png"
                mask_img.save(mask_path)
                mask_tensor = torch.tensor(mask, dtype=torch.uint8)
                _mask += mask_tensor
        
        # clip mask to 0,1
        _mask[_mask > 0] = 1
        masks.append(_mask)
    return torch.stack(masks)

    
            
        

def denormalize_image(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    De-normalizes an image tensor for visualization.
    
    Args:
        image_tensor (torch.Tensor): Normalized image tensor of shape (C, H, W).
        mean (list): Mean values used for normalization.
        std (list): Standard deviation values used for normalization.

    Returns:
        np.ndarray: De-normalized image as a NumPy array (H, W, C).
    """
    mean = torch.tensor(mean).view(3, 1, 1).to(image_tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(image_tensor.device)
    image = image_tensor * std + mean  # Reverse normalization
    image = torch.clamp(image, 0, 1)  # Clip values to [0,1]
    return image.permute(1, 2, 0).cpu().detach().numpy()  # Convert to (H, W, C) format

def overlay_gradcam(image_tensor, gradcam_heatmap, alpha=0.4):
    """
    Overlays the Grad-CAM heatmap on the original image.
    
    Args:
        image_tensor (torch.Tensor): Original image tensor (C, H, W), normalized.
        gradcam_heatmap (torch.Tensor): Grad-CAM heatmap tensor (1, H', W').
        alpha (float): Transparency factor for overlay.

    Returns:
        np.ndarray: Image with Grad-CAM overlay.
    """
    # De-normalize the image
    image = denormalize_image(image_tensor)

    # Resize heatmap to match the image size
    gradcam_heatmap = gradcam_heatmap.squeeze().cpu().detach().numpy()  # Convert to NumPy (H', W')
    gradcam_heatmap = cv2.resize(gradcam_heatmap, (image.shape[1], image.shape[0]))  # Resize

    # Convert heatmap to colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * gradcam_heatmap), cv2.COLORMAP_JET)
    heatmap = heatmap / 255.0  # Normalize to [0,1]

    # Overlay heatmap on image
    overlay = (1 - alpha) * image + alpha * heatmap  # Blend images

    return np.clip(overlay, 0, 1)  # Ensure values are in range [0,1]

def show_gradcam(images, gradcams, num_samples=5):
    """
    Displays Grad-CAM results for a batch of images.
    
    Args:
        images (torch.Tensor): Batch of input images (B, C, H, W).
        gradcams (torch.Tensor): Batch of Grad-CAM heatmaps (B, 1, H', W').
        num_samples (int): Number of images to display.
    """
    num_samples = min(num_samples, images.shape[0])  # Limit to batch size
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))

    for i in range(num_samples):
        original = denormalize_image(images[i])
        overlay = overlay_gradcam(images[i], gradcams[i])

        axes[i, 0].imshow(original)
        axes[i, 0].axis("off")
        axes[i, 0].set_title("Original Image")

        axes[i, 1].imshow(overlay)
        axes[i, 1].axis("off")
        axes[i, 1].set_title("Grad-CAM Overlay")

    save_path = "gradcam/gradcam_results.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create directory if it doesn't exist
    plt.draw()
    plt.savefig(save_path)
