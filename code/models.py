import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel


CNNSim2_param = {
    "st_fc1_in": 512,
    "st_fc1_out": 256,
    "st_fc2_out": 128,
    "st_fc3_out": 64,
    "consis_fc1_out": 128,
    "consis_fc2_out": 64,
    "fusion_fc1_out": 64,
    "fusion_fc2_out": 32,
}


class TransformerEncoder(nn.Module):
    """General attention for tgt & src from different modalities."""

    def __init__(self, model_dim, layer_num, head, tgt_seq, src_seq):
        super(TransformerEncoder, self).__init__()
        self.layer_num = layer_num

        self.multihead_attns_t = nn.ModuleList(
            [nn.MultiheadAttention(model_dim, head) for _ in range(self.layer_num)]
        )
        self.multihead_attns_s = nn.ModuleList(
            [nn.MultiheadAttention(model_dim, head) for _ in range(self.layer_num)]
        )

        # Make LayerNorm independent of sequence length (more robust)
        self.LN_ts1 = nn.ModuleList(
            [nn.LayerNorm(model_dim) for _ in range(self.layer_num)]
        )
        self.LN_ss1 = nn.ModuleList(
            [nn.LayerNorm(model_dim) for _ in range(self.layer_num)]
        )

        self.LN_ts2 = nn.ModuleList(
            [nn.LayerNorm(model_dim) for _ in range(self.layer_num)]
        )
        self.LN_ss2 = nn.ModuleList(
            [nn.LayerNorm(model_dim) for _ in range(self.layer_num)]
        )

        FF_K = 4
        self.ff_ts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(model_dim, FF_K * model_dim),
                    nn.ReLU(),
                    nn.Linear(FF_K * model_dim, model_dim),
                )
                for _ in range(self.layer_num)
            ]
        )
        self.ff_ss = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(model_dim, FF_K * model_dim),
                    nn.ReLU(),
                    nn.Linear(FF_K * model_dim, model_dim),
                )
                for _ in range(self.layer_num)
            ]
        )

    def forward(self, tgt, src):
        """
        tgt: [B, tgt_seq, D]
        src: [B, src_seq, D]
        """
        for (
            multihead_attn_t,
            multihead_attn_s,
            ff_t,
            ff_s,
            LN_t1,
            LN_s1,
            LN_t2,
            LN_s2,
        ) in zip(
            self.multihead_attns_t,
            self.multihead_attns_s,
            self.ff_ts,
            self.ff_ss,
            self.LN_ts1,
            self.LN_ss1,
            self.LN_ts2,
            self.LN_ss2,
        ):
            res_t = tgt
            res_s = src

            # [Seq, B, D]
            tgt_perm = tgt.permute(1, 0, 2)
            src_perm = src.permute(1, 0, 2)

            tgt_new, _ = multihead_attn_t(tgt_perm, src_perm, src_perm)
            src_new, _ = multihead_attn_s(src_perm, tgt_perm, tgt_perm)

            # [B, Seq, D]
            tgt_new = tgt_new.permute(1, 0, 2)
            src_new = src_new.permute(1, 0, 2)

            tgt_new = LN_t1(tgt_new + res_t)
            src_new = LN_s1(src_new + res_s)

            res_t = tgt_new
            res_s = src_new

            tgt_new = ff_t(tgt_new)
            src_new = ff_s(src_new)

            tgt = LN_t2(tgt_new + res_t)
            src = LN_s2(src_new + res_s)

        return tgt, src


class C3N(nn.Module):
    def __init__(self, args):
        super(C3N, self).__init__()

        self.is_weibo = (args.dataset == "weibo")
        self.device = args.device
        self.crop_num = args.crop_num

        # ------------------------------------------------------
        # 1. Load HuggingFace CLIP (replaces cn_clip + openai/clip)
        # ------------------------------------------------------
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch16"
        ).to(self.device)

        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch16"
        )

        # HF CLIP ViT-B/16 produces 512-d embeddings
        self.fc_768 = nn.Linear(512, 512)

        # ------------------------------------------------------
        # 2. Rest of original architecture
        # ------------------------------------------------------
        self.finetune = args.finetune
        Ks_word = args.conv_kernel  # we still use its *length* for Conv_out

        # We keep multiple convs (one per K) but use kernel height 1 for stability.
        # Each conv: kernel over [T, K=crop_num] where T will be 1 here.
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=args.conv_out,
                    kernel_size=(1, self.crop_num),
                )
                for _ in Ks_word
            ]
        )

        self.logit_scale = 100
        Conv_out = len(Ks_word) * args.conv_out

        self.transformer = TransformerEncoder(
            model_dim=512,
            layer_num=args.layer_num,
            head=8,
            tgt_seq=args.st_num,
            src_seq=args.crop_num,
        )

        self.fc_st1 = nn.Sequential(
            nn.Linear(CNNSim2_param["st_fc1_in"], CNNSim2_param["st_fc1_out"]),
            nn.ReLU(),
            nn.Linear(CNNSim2_param["st_fc1_out"], CNNSim2_param["st_fc2_out"]),
            nn.ReLU(),
            nn.Linear(CNNSim2_param["st_fc2_out"], CNNSim2_param["st_fc3_out"]),
            nn.ReLU(),
        )

        self.fc_consis1 = nn.Sequential(
            nn.Linear(Conv_out, CNNSim2_param["consis_fc1_out"]),
            nn.ReLU(),
            nn.Linear(CNNSim2_param["consis_fc1_out"], CNNSim2_param["consis_fc2_out"]),
            nn.ReLU(),
        )

        self.fc_ob1 = nn.Sequential(
            nn.Linear(CNNSim2_param["st_fc1_in"], CNNSim2_param["st_fc1_out"]),
            nn.ReLU(),
            nn.Linear(CNNSim2_param["st_fc1_out"], CNNSim2_param["st_fc2_out"]),
            nn.ReLU(),
            nn.Linear(CNNSim2_param["st_fc2_out"], CNNSim2_param["st_fc3_out"]),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                CNNSim2_param["consis_fc2_out"] + CNNSim2_param["st_fc3_out"] * 2,
                128,
            ),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # fusion output is 32
        self.fc = nn.Linear(32, 2)
        self.dropout = nn.Dropout(args.dropout_p)

        # ------------------------------------------------------
        # 3. Freeze CLIP if not finetuning
        # ------------------------------------------------------
        if not self.finetune:
            for _, param in self.clip_model.named_parameters():
                param.requires_grad = False

    def similarity_weight(self, txt_fea, img_fea):
        """
        Compute similarity-based consistency features.
        txt_fea: [B, T, D]
        img_fea: [B, K, D]
        """
        txt_fea_ = txt_fea / txt_fea.norm(dim=-1, keepdim=True)
        img_fea_ = img_fea / img_fea.norm(dim=-1, keepdim=True)

        img_fea_T = img_fea_.transpose(1, 2)  # [B, D, K]
        sim = torch.matmul(txt_fea_, img_fea_T)  # [B, T, K]
        sim = (self.logit_scale * sim).unsqueeze(1)  # [B, 1, T, K]

        # Conv over (T, K=crop_num)
        fea_maps = [F.relu(conv(sim)).squeeze(3) for conv in self.convs]  # [B, out, T']
        consis_fea_avg = [F.avg_pool1d(m, m.size(2)).squeeze(2) for m in fea_maps]
        consis_fea_avg = torch.cat(consis_fea_avg, dim=1)  # [B, Conv_out]

        # Use first positions as "sentence" / "object" embedding
        st_embed_pos = txt_fea[:, 0, :]  # [B, D]
        ob_embed_pos = img_fea[:, 0, :]  # [B, D]

        return st_embed_pos, ob_embed_pos, consis_fea_avg

    def forward(self, data):
        """
        data:
          - data["text_input"]: list[str] (B elements)  <-- must be raw text now
          - data["crop_input"]: tensor [B, crop_num, 3, 224, 224]
          - data["label"]:      tensor [B]
          - data["n_word_input"]: still present but unused with HF CLIP
        """
        raw_text = data["text_input"]
        crop_tensor = data["crop_input"]  # [B, crop_num, 3, 224, 224]

        B, CNUM, _, _, _ = crop_tensor.shape
        assert CNUM == self.crop_num, f"Got {CNUM} crops, expected {self.crop_num}"

        # -----------------------------------------------------
        # 1. Encode text with HF CLIP
        # -----------------------------------------------------
        text_inputs = self.clip_processor(
            text=raw_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        text_features = self.clip_model.get_text_features(**text_inputs)  # [B, 512]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = self.fc_768(text_features)  # [B, 512]
        text_features = text_features.unsqueeze(1)  # [B, 1, 512]

        # -----------------------------------------------------
        # 2. Encode each crop independently with HF CLIP
        # -----------------------------------------------------
        crop_features_list = []

        for i in range(CNUM):
            crop_inputs = self.clip_processor(
                images=crop_tensor[:, i],
                return_tensors="pt",
            )
            crop_inputs = {k: v.to(self.device) for k, v in crop_inputs.items()}

            crop_feat = self.clip_model.get_image_features(**crop_inputs)  # [B, 512]
            crop_feat = crop_feat / crop_feat.norm(dim=-1, keepdim=True)
            crop_feat = self.fc_768(crop_feat)  # [B, 512]
            crop_features_list.append(crop_feat)

        crop_features = torch.stack(crop_features_list, dim=1)  # [B, crop_num, 512]

        # -----------------------------------------------------
        # 3. Cross-modal transformer
        # -----------------------------------------------------
        wi_fea, iw_fea = self.transformer(text_features, crop_features)

        # -----------------------------------------------------
        # 4. Similarity + fusion
        # -----------------------------------------------------
        st, ob, consis = self.similarity_weight(wi_fea, iw_fea)

        st = self.dropout(self.fc_st1(st))
        ob = self.dropout(self.fc_ob1(ob))
        consis = self.dropout(self.fc_consis1(consis))

        fused = torch.cat([st, ob, consis], dim=-1)
        fused = self.dropout(self.fusion(fused))

        logits = self.fc(fused)
        log_prob = F.log_softmax(logits, dim=-1)

        return log_prob
