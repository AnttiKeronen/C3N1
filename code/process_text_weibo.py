import os
import pandas as pd
import numpy as np
import re
from tqdm import tqdm

# Path to processed data
processed_dir = r"C:\Users\keron\OneDrive\Työpöytä\C3N\data\weibo\processed"


def clean_text(text: str):
    """Basic cleanup."""
    if not isinstance(text, str):
        text = str(text)

    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_punctuation(text):
    """Remove timestamps and weird characters."""
    text = re.sub(r"\w{3}\s\w{3}\s\d{2}\s\d{2}:\d{2}:\d{2}\s\+\d{4}\s\d{4}", "", text)
    text = re.sub(r"[\n\\]|&quot", "", text)
    return text


def main():
    # Load EANN frozen data
    df_train = np.load(os.path.join(processed_dir, "train_EANN_frozen.npy"), allow_pickle=True)
    df_valid = np.load(os.path.join(processed_dir, "valid_EANN_frozen.npy"), allow_pickle=True)
    df_test = np.load(os.path.join(processed_dir, "test_EANN_frozen.npy"), allow_pickle=True)

    columns = ["original_post", "label", "image_id", "post_id"]

    df_train = pd.DataFrame(df_train, columns=columns)
    df_valid = pd.DataFrame(df_valid, columns=columns)
    df_test = pd.DataFrame(df_test, columns=columns)

    all_df = pd.concat([df_train, df_valid, df_test], ignore_index=True)

    text_dic = {}

    print("Processing text...")
    for _, row in tqdm(all_df.iterrows(), total=len(all_df)):
        post_id = row["post_id"]

        text = clean_text(row["original_post"])
        text = remove_punctuation(text)

        text_dic[post_id] = text

    # Save cleaned raw text
    save_path = os.path.join(processed_dir, "clean_text_inputs.npy")
    np.save(save_path, text_dic)
    print(f"Saved text → {save_path}")


if __name__ == "__main__":
    main()
