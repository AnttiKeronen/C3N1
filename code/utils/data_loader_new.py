import torch
from torch.utils.data import Dataset


class FakeNewsDataset(Dataset):
    """
    Updated for HuggingFace CLIP pipeline.
    Returns:
      - text_input: raw string (full post text)
      - n_word_input: list[str] (keywords from n_words.npy)
      - crop_input: tensor of shape [crop_num, 3, 224, 224]
      - label: int
    """

    def __init__(self, data_df, crop_num, st_num, dataset, n_words, crop_input, text_input):
        self.data_df = data_df
        self.crop_num = crop_num
        self.st_num = st_num
        self.dataset = dataset

        # These dictionaries now contain:
        # n_words: dict[post_id] -> list[str]
        # crop_input: dict[image_id] -> 5×(3×224×224) tensor
        # text_input: dict[post_id] -> raw text string
        self.n_words = n_words
        self.crop_input = crop_input
        self.text_input = text_input

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]

        post_id = row["post_id"]
        image_id = row["image_id"]
        label = torch.tensor(row["label"], dtype=torch.long)

        # Raw text string
        text_str = self.text_input[post_id]          # e.g. "A village in East Germany..."

        # List of keyword strings
        keyword_list = self.n_words[post_id]         # e.g. ["village", "media", "migration", ...]

        # 5 image crops (tensor: [5, 3, 224, 224])
        crops = self.crop_input[image_id]

        sample = {
            "post_id": post_id,
            "label": label,
            "text_input": text_str,
            "n_word_input": keyword_list,
            "crop_input": crops
        }

        return sample
