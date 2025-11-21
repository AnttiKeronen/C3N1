import os
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import pandas as pd
from transformers import CLIPProcessor, CLIPModel

# -----------------------------
# PATHS
# -----------------------------
data_dir = r"C:\Users\keron\OneDrive\Työpöytä\C3N\data\weibo\row"
processed_dir = r"C:\Users\keron\OneDrive\Työpöytä\C3N\data\weibo\processed"
device = "cpu"

CROP_NUM = 5


# -----------------------------
# Simple 5-crop function
# -----------------------------
def make_5_crops(img: Image.Image):
    w, h = img.size
    half_w, half_h = w // 2, h // 2

    return [
        img,  # full
        img.crop((0, 0, half_w, half_h)),          # top-left
        img.crop((half_w, 0, w, half_h)),          # top-right
        img.crop((0, half_h, half_w, h)),          # bottom-left
        img.crop((half_w, half_h, w, h)),          # bottom-right
    ]


# -----------------------------
# MAIN
# -----------------------------
def main():

    # load split files
    df_train = np.load(os.path.join(processed_dir, "train_EANN_frozen.npy"), allow_pickle=True)
    df_valid = np.load(os.path.join(processed_dir, "valid_EANN_frozen.npy"), allow_pickle=True)
    df_test = np.load(os.path.join(processed_dir, "test_EANN_frozen.npy"), allow_pickle=True)

    columns = ['original_post', 'label', 'image_id', 'post_id']
    df_train = pd.DataFrame(df_train, columns=columns)
    df_valid = pd.DataFrame(df_valid, columns=columns)
    df_test = pd.DataFrame(df_test, columns=columns)

    all_df = pd.concat([df_train, df_valid, df_test], ignore_index=True)

    # -----------------------------
    # Load HuggingFace CLIP
    # -----------------------------
    print("Loading CLIP...")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    image_dict = {}

    print("Processing images...")
    for _, row in tqdm(all_df.iterrows(), total=len(all_df)):
        image_id = row["image_id"]

        img_path1 = os.path.join(data_dir, "nonrumor_images", image_id + ".jpg")
        img_path2 = os.path.join(data_dir, "rumor_images", image_id + ".jpg")

        img_path = img_path1 if os.path.exists(img_path1) else img_path2

        if not os.path.exists(img_path):
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except:
            continue

        crops = make_5_crops(img)

        crop_tensors = []
        for crop in crops:
            # Resize every crop to 224x224
            crop = crop.resize((224, 224))

            inputs = processor(images=crop, return_tensors="pt")
            tensor = inputs["pixel_values"][0]  # (3,224,224)

            crop_tensors.append(tensor)

        crop_tensors = torch.stack(crop_tensors, dim=0)  # [5, 3, 224, 224]
        image_dict[image_id] = crop_tensors.numpy()

    save_path = os.path.join(processed_dir, "clip_image_preprocess.npy")
    np.save(save_path, image_dict, allow_pickle=True)

    print(f"Saved image crops → {save_path}")


if __name__ == "__main__":
    main()
