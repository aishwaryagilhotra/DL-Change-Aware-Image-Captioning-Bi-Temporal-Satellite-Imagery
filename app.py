"""
ChangeInsight — AI-Powered Change Detection & Captioning for Remote Sensing
Run:
    pip install streamlit torch torchvision pillow numpy pandas matplotlib
    streamlit run app.py

The app wires the three dissertation models into one inference workflow:
    Model 1: Change Localization (Enhanced Siamese ResNet18 U-Net)
    Model 2: Semantic Change Classification (Siamese ResNet18)
    Model 3: Change Caption Generation (Siamese ResNet18 + GRU)

Update the MODEL_* paths below if your checkpoint filenames differ.
"""

import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"C:\College\Dissertation\Data\FINAL_LEVIR_DATASET")

MODEL1_PATH = BASE_DIR / "CHANGE_DETECTION_MODEL" / "EXPERIMENT_4_RESNET18" / "experiment4_best_model.pth"

MODEL2_DIR = BASE_DIR / "CHANGE_CLASSIFICATION_MODEL"
MODEL2_PATH = MODEL2_DIR / "model2_best_siamese_resnet18.pth"

MODEL3_DIR = BASE_DIR / "CAPTION_GENERATION_MODEL"
MODEL3_PATH = MODEL3_DIR / "model3_best_caption_model.pth"
VOCAB_PATH = MODEL3_DIR / "model3_vocabulary.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output order from the existing 7-class Model 2 checkpoint.
# Keep this list aligned with the label encoding used during training.
MODEL2_CLASSES = [
    "No Change",
    "Building / Construction",
    "Vegetation",
    "Road / Infrastructure",
    "Land / Surface",
    "Water",
    "Other Change",
]

# Matches the visual language of the supplied design.
NAVY = "#12304A"
NAVY_2 = "#183D5C"
ORANGE = "#F26A21"
CREAM = "#F7F4EE"
WHITE = "#FFFFFF"
MUTED = "#64748B"
BORDER = "#DCE3E8"
PALE_BLUE = "#EAF1F5"
PALE_ORANGE = "#FFF1E9"


# ============================================================
# PAGE / CSS
# ============================================================

st.set_page_config(
    page_title="ChangeInsight — Remote Sensing Change Analysis",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
}}
.stApp {{
    background:
        radial-gradient(circle at 92% 3%, rgba(242,106,33,.07), transparent 25%),
        linear-gradient(135deg, #fbfaf7 0%, #f5f3ee 100%);
    color: {NAVY};
}}
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #f5f7f8 0%, #eaf0f2 62%, #173a56 62%, #102d43 100%);
    border-right: 1px solid #dce4e8;
}}
section[data-testid="stSidebar"] > div {{
    padding-top: 1rem;
}}
.brand {{
    padding: 8px 10px 22px 10px;
}}
.brand-name {{
    font-family: 'Playfair Display', serif;
    font-size: 29px;
    font-weight: 700;
    letter-spacing: -.8px;
}}
.brand-name span {{ color: {ORANGE}; }}
.brand-sub {{
    color: {MUTED};
    font-size: 11px;
    margin-top: -3px;
}}
.nav-note {{
    color: #dbe8ef;
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 22px 10px 8px;
}}
.hero {{
    min-height: 255px;
    border-radius: 22px;
    padding: 35px 42px;
    margin-bottom: 20px;
    background:
      linear-gradient(90deg, rgba(247,244,238,.97) 0%, rgba(247,244,238,.88) 47%, rgba(247,244,238,.32) 100%),
      radial-gradient(circle at 78% 20%, rgba(19,48,74,.12), transparent 30%);
    border: 1px solid #e4e5e2;
    box-shadow: 0 12px 36px rgba(18,48,74,.06);
    position: relative;
    overflow: hidden;
}}
.hero-kicker {{
    color: {NAVY};
    font-size: 11px;
    letter-spacing: 4px;
    text-transform: uppercase;
    font-weight: 600;
}}
.hero h1 {{
    font-family: 'Playfair Display', serif;
    font-size: 52px;
    line-height: .98;
    max-width: 700px;
    margin: 20px 0 12px;
    color: {NAVY};
}}
.hero h1 span {{ color: {ORANGE}; }}
.hero p {{
    max-width: 620px;
    font-size: 16px;
    color: #30465A;
    line-height: 1.55;
}}
.section {{
    background: rgba(255,255,255,.80);
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 7px 24px rgba(18,48,74,.045);
    height: 100%;
}}
.section-title {{
    font-size: 19px;
    font-weight: 700;
    color: {NAVY};
}}
.section-sub {{
    font-size: 12px;
    color: {MUTED};
    margin-bottom: 14px;
}}
.step {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: {NAVY};
    color: white;
    font-weight: 700;
    margin-right: 10px;
}}
.step-row {{ display:flex; align-items:center; }}
.model-card {{
    border: 1px solid #d9e0e5;
    border-radius: 14px;
    padding: 14px;
    min-height: 128px;
    background: #fbfcfc;
}}
.model-card.active {{
    border: 1.5px solid {ORANGE};
    background: #fffaf6;
    box-shadow: 0 4px 16px rgba(242,106,33,.10);
}}
.model-number {{
    color: {ORANGE};
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}}
.model-name {{
    color: {NAVY};
    font-size: 15px;
    font-weight: 700;
    margin-top: 6px;
}}
.model-desc {{
    color: {MUTED};
    font-size: 11px;
    line-height: 1.35;
    margin-top: 3px;
}}
.metric {{
    border: 1px solid {BORDER};
    border-radius: 13px;
    padding: 13px;
    background: #fff;
}}
.metric-label {{ font-size: 11px; color: {MUTED}; }}
.metric-value {{ font-size: 22px; font-weight: 700; color: {NAVY}; margin-top: 5px; }}
.result-card {{
    border: 1px solid {BORDER};
    border-radius: 18px;
    background: rgba(255,255,255,.88);
    padding: 18px;
    box-shadow: 0 7px 24px rgba(18,48,74,.045);
}}
.badge-change {{
    display:inline-block;
    background:#fff0ea;
    color:#c84d0c;
    border:1px solid #ffd4c2;
    border-radius:999px;
    padding:7px 12px;
    font-size:12px;
    font-weight:700;
}}
.badge-nochange {{
    display:inline-block;
    background:#edf7f2;
    color:#287453;
    border:1px solid #c9eadb;
    border-radius:999px;
    padding:7px 12px;
    font-size:12px;
    font-weight:700;
}}
.caption {{
    background:#f2f5f7;
    border:1px solid #dde5ea;
    border-radius:13px;
    padding:17px;
    font-family:'Playfair Display', serif;
    font-size:18px;
    line-height:1.5;
    color:{NAVY};
}}
.small {{
    color:{MUTED};
    font-size:11px;
}}
div[data-testid="stFileUploader"] {{
    background: #fafcfc;
    border-radius: 13px;
}}
.stButton > button {{
    border-radius: 10px;
    border: 0;
    background: {NAVY};
    color: white;
    font-weight: 700;
    min-height: 45px;
}}
.stButton > button:hover {{
    background: {NAVY_2};
    color: white;
}}
div[data-baseweb="select"] > div {{
    border-radius: 10px;
}}
hr {{ border-color: #e1e6e9; }}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# UTILS
# ============================================================

def clean_state_dict(checkpoint):
    """Accept common checkpoint formats."""
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break

    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format.")

    out = {}
    for k, v in checkpoint.items():
        if not torch.is_tensor(v):
            continue
        nk = k
        for prefix in ("module.", "model.", "net."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        out[nk] = v
    return out


def load_weights(model, path):
    """Load all checkpoint tensors strictly; never run with partial weights."""
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    state = clean_state_dict(checkpoint)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Strict checkpoint loading failed for {Path(path).name}. "
            "The model architecture must match the trained checkpoint exactly. "
            f"Original details: {exc}"
        ) from exc


def read_image(uploaded):
    return Image.open(uploaded).convert("RGB")


def make_transform(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def prepare_pair(img_a, img_b, size):
    tfm = make_transform(size)
    a = tfm(img_a).unsqueeze(0).to(DEVICE)
    b = tfm(img_b).unsqueeze(0).to(DEVICE)
    return a, b


def save_image_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def overlay_mask(image, mask, alpha=.45):
    base = np.asarray(image.resize((mask.shape[1], mask.shape[0]))).astype(np.float32)
    overlay = base.copy()
    overlay[mask] = [242, 70, 33]
    out = base * (1-alpha) + overlay * alpha
    out[mask] = base[mask] * (1-alpha) + np.array([242, 70, 33]) * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def make_probability_map(prob):
    arr = (prob * 255).astype(np.uint8)
    return Image.fromarray(arr).convert("L").resize((512, 512))


# ============================================================
# MODEL 1 — EXACT EXPERIMENT 4 CHECKPOINT ARCHITECTURE
# ============================================================

class ConvBNReLU(nn.Module):
    """Two-convolution block used by Experiment 4."""
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FeatureFusion(nn.Module):
    """Checkpoint-compatible temporal feature fusion."""
    def __init__(self, channels):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            ConvBNReLU(channels, channels),
        )

    def forward(self, a, b):
        return self.fusion(torch.cat([a, b, torch.abs(a - b)], dim=1))


class UpBlock(nn.Module):
    def __init__(self, cin, skip_channels, cout):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, kernel_size=2, stride=2)
        self.conv = ConvBNReLU(cout + skip_channels, cout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class EnhancedSiameseResNet18UNet(nn.Module):
    """Experiment 4 architecture; module names match the trained state dict."""
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=None)

        # Keep the original ResNet names: conv1, bn1, layer1 ... layer4.
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.fuse1 = FeatureFusion(64)
        self.fuse2 = FeatureFusion(128)
        self.fuse3 = FeatureFusion(256)
        self.fuse4 = FeatureFusion(512)

        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 32)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def encode(self, x):
        stem = self.relu(self.bn1(self.conv1(x)))  # 128 x 128 skip
        x = self.maxpool(stem)
        x1 = self.layer1(x)                       # 64 x 64
        x2 = self.layer2(x1)                      # 32 x 32
        x3 = self.layer3(x2)                      # 16 x 16
        x4 = self.layer4(x3)                      # 8 x 8
        return stem, x1, x2, x3, x4

    def forward(self, a, b):
        a0, a1, a2, a3, a4 = self.encode(a)
        b0, b1, b2, b3, b4 = self.encode(b)

        f1 = self.fuse1(a1, b1)
        f2 = self.fuse2(a2, b2)
        f3 = self.fuse3(a3, b3)
        f4 = self.fuse4(a4, b4)

        d4 = self.up4(f4, f3)
        d3 = self.up3(d4, f2)
        d2 = self.up2(d3, f1)
        d1 = self.up1(d2, a0 + b0 if False else (a0 + b0) / 2.0)
        out = self.final_up(d1)
        return self.head(out)


# ============================================================
# MODEL 2 — SIAMESE RESNET18 SEMANTIC CLASSIFIER
# ============================================================

class SiameseResNet18Classifier(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        base = models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(base.children())[:-1])

        # Keep module indices aligned with the trained Model 2 checkpoint:
        # 0=Linear, 1=BatchNorm1d, 4=Linear, 7=output Linear.
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(.25),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(.15),
            nn.Linear(128, num_classes),
        )

    def feat(self, x):
        return self.encoder(x).flatten(1)

    def forward(self, a, b):
        fa, fb = self.feat(a), self.feat(b)
        x = torch.cat([fa, fb, torch.abs(fa-fb), fa*fb], dim=1)
        return self.classifier(x)


# ============================================================
# MODEL 3 — SIAMESE RESNET18 + GRU CAPTIONER
# ============================================================

class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256, feat_dim=1536):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.feature_projection = nn.Linear(feat_dim, hidden_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions=None, max_len=40, sos_idx=1):
        h0 = self.feature_projection(features).unsqueeze(0)

        if captions is not None:
            inp = captions[:, :-1]
            emb = self.embedding(inp)
            out, _ = self.gru(emb, h0)
            return self.output(out)

        ids = torch.full(
            (features.size(0), 1), sos_idx,
            dtype=torch.long, device=features.device
        )
        h = h0
        generated = []

        for _ in range(max_len):
            emb = self.embedding(ids[:, -1:])
            out, h = self.gru(emb, h)
            logits = self.output(out[:, -1])
            nxt = logits.argmax(-1, keepdim=True)
            generated.append(nxt)
            ids = torch.cat([ids, nxt], dim=1)

        return torch.cat(generated, dim=1)


class CaptionResNetEncoder(nn.Module):
    """ResNet-18 feature encoder with checkpoint-compatible encoder.encoder.* keys."""
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(base.children())[:-2])

    def forward(self, x):
        return self.encoder(x)


class SiameseResNet18CaptionModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256):
        super().__init__()
        # The trained checkpoint stores weights under encoder.encoder.*.
        self.encoder = CaptionResNetEncoder()
        self.decoder = CaptionDecoder(
            vocab_size=vocab_size,
            embed_dim=256,
            hidden_dim=hidden_dim,
            feat_dim=1536,
        )

    def feat(self, x):
        return F.adaptive_avg_pool2d(self.encoder(x), 1).flatten(1)

    def forward(self, a, b, captions=None, max_len=40, sos_idx=1):
        fa, fb = self.feat(a), self.feat(b)
        temporal = torch.cat([fa, fb, torch.abs(fa-fb)], dim=1)
        return self.decoder(
            temporal, captions=captions,
            max_len=max_len,
            sos_idx=sos_idx,
        )


# ============================================================
# LOADERS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model1():
    model = EnhancedSiameseResNet18UNet().to(DEVICE)
    load_weights(model, MODEL1_PATH)
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def load_model2():
    model = SiameseResNet18Classifier(num_classes=7).to(DEVICE)
    load_weights(model, MODEL2_PATH)
    model.eval()
    return model


def normalize_caption_vocab(raw_vocab):
    """Extract token->index mapping from supported vocabulary JSON layouts."""
    if not isinstance(raw_vocab, dict):
        raise ValueError(
            f"Caption vocabulary must be a JSON object, got {type(raw_vocab).__name__}."
        )

    # Common saved formats use a named nested mapping.
    preferred_keys = (
        "token_to_idx", "token_to_id", "word_to_idx", "word2idx",
        "stoi", "vocab", "vocabulary", "token2idx"
    )
    for key in preferred_keys:
        candidate = raw_vocab.get(key)
        if isinstance(candidate, dict) and candidate and all(
            isinstance(v, int) and not isinstance(v, bool) for v in candidate.values()
        ):
            return {str(token): int(index) for token, index in candidate.items()}

    # Sometimes the JSON stores token->index alongside idx->token and metadata.
    # Find a direct mapping whose values are all integer indices.
    for candidate in raw_vocab.values():
        if isinstance(candidate, dict) and candidate and all(
            isinstance(v, int) and not isinstance(v, bool) for v in candidate.values()
        ):
            return {str(token): int(index) for token, index in candidate.items()}

    # Direct token->index layout.
    if raw_vocab and all(
        isinstance(v, int) and not isinstance(v, bool) for v in raw_vocab.values()
    ):
        return {str(token): int(index) for token, index in raw_vocab.items()}

    # Direct index->token layout, e.g. {"0": "<pad>", "1": "<unk>"}.
    if raw_vocab and all(
        str(k).isdigit() and isinstance(v, str) for k, v in raw_vocab.items()
    ):
        return {token: int(index) for index, token in raw_vocab.items()}

    # A few vocab builders save entries as {token: {"index": N}}.
    for index_key in ("index", "idx", "id"):
        if raw_vocab and all(
            isinstance(v, dict) and isinstance(v.get(index_key), int)
            for v in raw_vocab.values()
        ):
            return {str(token): int(value[index_key]) for token, value in raw_vocab.items()}

    top_keys = list(raw_vocab.keys())[:12]
    raise ValueError(
        "Unrecognized caption vocabulary format in " + str(VOCAB_PATH) + ". "
        f"Top-level keys: {top_keys}. Expected token->index mapping or a nested "
        "token_to_idx/word2idx mapping."
    )


@st.cache_resource(show_spinner=False)
def load_model3():
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    token_to_idx = normalize_caption_vocab(vocab)
    model = SiameseResNet18CaptionModel(len(token_to_idx)).to(DEVICE)
    load_weights(model, MODEL3_PATH)
    model.eval()

    idx_to_token = {int(index): token for token, index in token_to_idx.items()}
    return model, token_to_idx, idx_to_token


# ============================================================
# INFERENCE
# ============================================================

@torch.inference_mode()
def run_model1(model, img_a, img_b, threshold):
    a, b = prepare_pair(img_a, img_b, 256)
    logits = model(a, b)
    logits = F.interpolate(
        logits, size=(256, 256), mode="bilinear", align_corners=False
    )
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = prob >= threshold
    return prob, mask


@torch.inference_mode()
def run_model2(model, img_a, img_b):
    a, b = prepare_pair(img_a, img_b, 224)
    logits = model(a, b)
    probs = torch.softmax(logits, dim=1)[0]
    idx = int(probs.argmax())
    return MODEL2_CLASSES[idx], float(probs[idx]), probs.cpu().numpy()


def detokenize(ids, idx_to_token):
    special = {"<pad>", "<sos>", "<bos>", "<eos>", "<unk>", "[PAD]", "[SOS]", "[EOS]"}
    words = []
    for idx in ids:
        tok = idx_to_token.get(int(idx), "<unk>")
        if tok in ("<eos>", "[EOS]"):
            break
        if tok in special:
            continue
        words.append(tok)
    text = " ".join(words)
    return text.replace(" ,", ",").replace(" .", ".").replace(" '", "'").strip()


@st.cache_data(max_entries=64, show_spinner=False)
def caption_from_ids(ids_tuple, vocab_items):
    idx_to_token = {int(k): v for k, v in vocab_items}
    return detokenize(ids_tuple, idx_to_token)


@torch.inference_mode()
def run_model3(model, token_to_idx, idx_to_token, img_a, img_b):
    a, b = prepare_pair(img_a, img_b, 224)
    sos = (
        token_to_idx.get("<sos>")
        or token_to_idx.get("<bos>")
        or token_to_idx.get("[SOS]")
        or 1
    )
    ids = model(a, b, captions=None, max_len=40, sos_idx=int(sos))[0].cpu().tolist()
    return detokenize(ids, idx_to_token)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-name">Change<span>Insight</span></div>
            <div class="brand-sub">Remote Sensing for a Changing World</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav = st.radio(
        "Navigation",
        [
            "⌂  Home",
            "▷  Run Inference",
            "◇  Model Information",
            "▤  Dataset Information",
            "▧  Results Gallery",
            "≋  Batch Processing",
            "⇩  Export & Download",
            "ⓘ  About",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        """
        <div style="margin-top:80px; padding:14px 10px;">
            <div style="color:#f0f5f8;font-family:'Playfair Display',serif;
                        font-size:28px;line-height:1.05;">
                Different<br>Times.<br>Same Place.<br><span style="color:#F26A21;">New Stories.</span>
            </div>
            <div style="margin-top:20px;color:#b9ccd8;font-size:9px;letter-spacing:3px;">
                OBSERVE · ANALYZE · UNDERSTAND
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">→ &nbsp; AI-POWERED CHANGE DETECTION AND CAPTIONING FOR REMOTE SENSING</div>
        <h1>See What <span>Changed.</span><br>Understand the Story.</h1>
        <p>
            Analyze bi-temporal satellite images to detect spatial changes,
            classify what changed, and generate natural-language descriptions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MAIN WORKFLOW
# ============================================================

if nav in ("⌂  Home", "▷  Run Inference"):

    # ---- STEP 1 / STEP 2 ----
    left, right = st.columns([1.45, 1.0], gap="large")

    with left:
        st.markdown(
            """
            <div class="section">
                <div class="step-row">
                    <span class="step">1</span>
                    <div>
                        <div class="section-title">Upload Bi-Temporal Images</div>
                        <div class="section-sub">Select two satellite images of the same area taken at different times</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        ca, cb = st.columns(2)

        with ca:
            st.markdown("**Image A — Before**")
            st.caption("Earlier time")
            file_a = st.file_uploader(
                "Upload Image A",
                type=["png", "jpg", "jpeg", "tif", "tiff"],
                key="image_a",
                label_visibility="collapsed",
            )
            if file_a:
                img_a = read_image(file_a)
                st.image(img_a, use_container_width=True)

        with cb:
            st.markdown("**Image B — After**")
            st.caption("Later time")
            file_b = st.file_uploader(
                "Upload Image B",
                type=["png", "jpg", "jpeg", "tif", "tiff"],
                key="image_b",
                label_visibility="collapsed",
            )
            if file_b:
                img_b = read_image(file_b)
                st.image(img_b, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            """
            <div class="section">
                <div class="step-row">
                    <span class="step">2</span>
                    <div>
                        <div class="section-title">Model & Settings</div>
                        <div class="section-sub">Run the complete three-stage analysis pipeline</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        model_choice = st.radio(
            "Pipeline",
            [
                "Complete Pipeline — Models 1 → 2 → 3",
                "Model 1 — Change Localization",
                "Model 2 — Semantic Classification",
                "Model 3 — Caption Generation",
            ],
            index=0,
            label_visibility="collapsed",
        )

        threshold = st.slider(
            "Prediction threshold",
            min_value=0.10,
            max_value=0.95,
            value=0.55,
            step=0.01,
            help="Model 1 threshold. The dissertation validation-selected threshold is 0.55.",
        )

        with st.expander("⚙ Advanced Options"):
            show_probability = st.checkbox("Show probability map", True)
            overlay_alpha = st.slider("Overlay opacity", .15, .75, .45, .05)

        run = st.button("▶  Run Analysis  →", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ---- INFERENCE ----
    if run:
        if not file_a or not file_b:
            st.error("Please upload both Image A and Image B before running the analysis.")
            st.stop()

        with st.spinner("Running the ChangeInsight pipeline..."):
            start = time.perf_counter()

            img_a = read_image(file_a)
            img_b = read_image(file_b)

            result = {
                "prob": None,
                "mask": None,
                "classification": None,
                "class_confidence": None,
                "caption": None,
            }

            if model_choice in (
                "Complete Pipeline — Models 1 → 2 → 3",
                "Model 1 — Change Localization",
            ):
                m1 = load_model1()
                result["prob"], result["mask"] = run_model1(
                    m1, img_a, img_b, threshold
                )

            if model_choice in (
                "Complete Pipeline — Models 1 → 2 → 3",
                "Model 2 — Semantic Classification",
            ):
                m2 = load_model2()
                cls, conf, _ = run_model2(m2, img_a, img_b)
                result["classification"] = cls
                result["class_confidence"] = conf

            if model_choice in (
                "Complete Pipeline — Models 1 → 2 → 3",
                "Model 3 — Caption Generation",
            ):
                m3, token_to_idx, idx_to_token = load_model3()
                result["caption"] = run_model3(
                    m3, token_to_idx, idx_to_token, img_a, img_b
                )

            elapsed = time.perf_counter() - start
            result["elapsed"] = elapsed

            st.session_state["result"] = result
            st.session_state["img_a"] = img_a
            st.session_state["img_b"] = img_b

    # ---- RESULTS ----
    if "result" in st.session_state:
        result = st.session_state["result"]
        img_a = st.session_state["img_a"]
        img_b = st.session_state["img_b"]

        st.markdown("<br>", unsafe_allow_html=True)
        rleft, rright = st.columns([1.45, 1.0], gap="large")

        with rleft:
            st.markdown(
                f"""
                <div class="section">
                    <div class="step-row">
                        <span class="step">3</span>
                        <div>
                            <div class="section-title">Analysis Results</div>
                            <div class="section-sub">Change localization and visualization results</div>
                        </div>
                        <div style="margin-left:auto;" class="small">◷ {result.get("elapsed", 0):.2f} seconds</div>
                    </div>
                """,
                unsafe_allow_html=True,
            )

            if result["mask"] is not None:
                tabs = st.tabs(["Predicted Mask", "Overlay", "Probability Map"])

                with tabs[0]:
                    st.image(
                        Image.fromarray((result["mask"] * 255).astype(np.uint8)),
                        caption="Predicted Change Mask",
                        use_container_width=True,
                    )

                with tabs[1]:
                    over = overlay_mask(
                        img_b.resize((256, 256)),
                        result["mask"],
                        overlay_alpha,
                    )
                    st.image(over, caption="Change Overlay — Orange = Change", use_container_width=True)

                with tabs[2]:
                    if show_probability:
                        st.image(
                            make_probability_map(result["prob"]),
                            caption="Model 1 Change Probability",
                            use_container_width=True,
                        )
                    else:
                        st.info("Enable the probability map under Advanced Options.")

                changed_px = int(result["mask"].sum())
                total_px = int(result["mask"].size)
                changed_pct = 100 * changed_px / max(total_px, 1)

                m1a, m1b, m1c = st.columns(3)
                with m1a:
                    st.metric("Changed Area", f"{changed_pct:.2f}%")
                with m1b:
                    st.metric("Changed Pixels", f"{changed_px:,}")
                with m1c:
                    st.metric("Threshold", f"{threshold:.2f}")
            else:
                st.info("Model 1 was not selected, so no spatial change mask was generated.")

            st.markdown("</div>", unsafe_allow_html=True)

        with rright:
            st.markdown(
                """
                <div class="section">
                    <div class="step-row">
                        <span class="step">4</span>
                        <div>
                            <div class="section-title">Change Information</div>
                            <div class="section-sub">Classification, confidence and generated description</div>
                        </div>
                    </div>
                """,
                unsafe_allow_html=True,
            )

            if result["mask"] is not None:
                changed = bool(result["mask"].any())
                badge = "badge-change" if changed else "badge-nochange"
                label = "Change Detected" if changed else "No Change Detected"
                st.markdown(
                    f'<span class="{badge}">● &nbsp; {label}</span>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            if result["classification"]:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<div class="metric"><div class="metric-label">Semantic Class</div>'
                                f'<div class="metric-value" style="font-size:17px;">{result["classification"]}</div></div>',
                                unsafe_allow_html=True)
                with c2:
                    st.markdown('<div class="metric"><div class="metric-label">Classification Confidence</div>'
                                f'<div class="metric-value">{result["class_confidence"]:.3f}</div></div>',
                                unsafe_allow_html=True)
            else:
                st.info("Model 2 was not selected.")

            st.markdown("<br>", unsafe_allow_html=True)

            if result["caption"]:
                st.markdown(
                    f"""
                    <div class="metric-label">Generated Caption</div>
                    <div class="caption">{result["caption"]}</div>
                    """,
                    unsafe_allow_html=True,
                )
                st.button("⧉  Copy caption", key="copy_caption")
            else:
                st.info("Model 3 was not selected.")

            st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# MODEL INFORMATION
# ============================================================

elif nav == "◇  Model Information":
    st.subheader("Three-Stage ChangeInsight Architecture")
    st.caption("The three models answer three different questions.")

    cols = st.columns(3)
    cards = [
        ("MODEL 1", "Where did change occur?", "Enhanced Siamese ResNet18 U-Net", 
         "Pixel-level change localization using shared temporal encoders, multi-scale feature fusion and a U-Net decoder.", 
         "IoU 0.8128  ·  Dice 0.8967"),
        ("MODEL 2", "What changed?", "Siamese ResNet18 Classifier", 
         "Weakly supervised semantic classification using the seven output classes of the existing trained checkpoint: No Change, Building / Construction, Vegetation, Road / Infrastructure, Land / Surface, Water, and Other Change.", 
         "4 semantic classes"),
        ("MODEL 3", "How do we describe it?", "Siamese ResNet18 + GRU", 
         "Temporal feature fusion followed by autoregressive natural-language caption generation.", 
         "BLEU-4 0.3920  ·  METEOR 0.6472"),
    ]

    for col, (num, q, name, desc, score) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="model-number">{num}</div>
                    <h3 style="font-family:'Playfair Display',serif;color:{NAVY};">{q}</h3>
                    <b>{name}</b>
                    <p class="small">{desc}</p>
                    <hr>
                    <span style="color:{ORANGE};font-weight:700;">{score}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Model 1 — Dissertation Evaluation")
    st.dataframe(
        pd.DataFrame({
            "Metric": ["IoU", "Dice / F1", "Precision", "Recall", "Pixel Accuracy"],
            "Score": [0.812765, 0.896713, 0.899500, 0.893943, 0.989509],
        }),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("### Model 3 — Dissertation Evaluation")
    st.dataframe(
        pd.DataFrame({
            "Metric": ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L"],
            "Score": [0.6959, 0.5823, 0.4853, 0.3920, 0.6472, 0.6607],
        }),
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# DATASET INFORMATION
# ============================================================

elif nav == "▤  Dataset Information":
    st.subheader("LEVIR Bi-Temporal Dataset")
    st.write(
        "ChangeInsight uses complementary LEVIR-CC and LEVIR-CD supervision. "
        "The unified working dataset contains paired 256×256 samples with "
        "change masks, change labels and natural-language captions."
    )

    a, b, c, d = st.columns(4)
    for col, value, label in [
        (a, "7,264", "Unified samples"),
        (b, "6,345", "Training samples"),
        (c, "919", "Validation samples"),
        (d, "5", "Captions / pair"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### Dataset Role in the Pipeline")
    st.dataframe(
        pd.DataFrame({
            "Stage": ["Model 1", "Model 2", "Model 3"],
            "Task": ["Change localization", "Semantic change classification", "Caption generation"],
            "Supervision": ["LEVIR-CD masks", "Caption-derived weak labels", "LEVIR-CC captions"],
        }),
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# RESULTS GALLERY
# ============================================================

elif nav == "▧  Results Gallery":
    st.subheader("Results Gallery")
    st.info(
        "Run an analysis from the Home / Run Inference page. "
        "The latest prediction remains available during this Streamlit session."
    )
    if "result" in st.session_state:
        result = st.session_state["result"]
        img_a = st.session_state["img_a"]
        img_b = st.session_state["img_b"]

        g1, g2 = st.columns(2)
        with g1:
            st.image(img_a, caption="Before — Image A", use_container_width=True)
        with g2:
            st.image(img_b, caption="After — Image B", use_container_width=True)

        if result["mask"] is not None:
            g1, g2 = st.columns(2)
            with g1:
                st.image(
                    Image.fromarray((result["mask"] * 255).astype(np.uint8)),
                    caption="Predicted Change Mask",
                    use_container_width=True,
                )
            with g2:
                st.image(
                    overlay_mask(img_b, result["mask"]),
                    caption="Change Overlay",
                    use_container_width=True,
                )

        if result["caption"]:
            st.markdown(f'<div class="caption">{result["caption"]}</div>', unsafe_allow_html=True)


# ============================================================
# BATCH PROCESSING
# ============================================================

elif nav == "≋  Batch Processing":
    st.subheader("Batch Processing")
    st.write("Upload paired images and process them using the same three-stage inference pipeline.")
    st.warning(
        "For the dissertation demo, single-pair inference is recommended. "
        "Batch processing is provided as an extension for multiple paired inputs."
    )
    st.file_uploader(
        "Upload a ZIP containing paired before/after images",
        type=["zip"],
        disabled=True,
        help="Enable after defining your project-specific filename pairing convention.",
    )


# ============================================================
# EXPORT
# ============================================================

elif nav == "⇩  Export & Download":
    st.subheader("Export & Download")
    if "result" not in st.session_state:
        st.info("Run an analysis first.")
    else:
        result = st.session_state["result"]

        export = {
            "pipeline": "ChangeInsight",
            "model1_threshold": threshold if "threshold" in locals() else None,
            "changed_area_percent": (
                float(result["mask"].mean() * 100)
                if result["mask"] is not None else None
            ),
            "semantic_class": result["classification"],
            "classification_confidence": result["class_confidence"],
            "generated_caption": result["caption"],
            "inference_time_seconds": result["elapsed"],
        }

        st.download_button(
            "⇩ Download Results JSON",
            data=json.dumps(export, indent=2),
            file_name="changeinsight_results.json",
            mime="application/json",
            use_container_width=True,
        )

        if result["mask"] is not None:
            mask_bytes = save_image_bytes(
                Image.fromarray((result["mask"] * 255).astype(np.uint8))
            )
            st.download_button(
                "⇩ Download Predicted Mask",
                data=mask_bytes,
                file_name="change_mask.png",
                mime="image/png",
                use_container_width=True,
            )


# ============================================================
# ABOUT
# ============================================================

elif nav == "ⓘ  About":
    st.subheader("About ChangeInsight")
    st.markdown(
        """
        **ChangeInsight** is an AI-powered remote-sensing analysis system
        designed around a simple progression:

        **Where? → What? → How?**

        1. **Where?** — pixel-level change localization.
        2. **What?** — semantic classification of the detected change.
        3. **How?** — natural-language description of the observed temporal change.

        The system is designed as a research prototype for bi-temporal satellite
        imagery and demonstrates how spatial detection, semantic interpretation
        and language generation can be combined into a single analysis workflow.
        """
    )

    st.markdown("### Hardware")
    st.write(
        f"Inference device: **{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}**"
    )
    st.write(f"PyTorch device: **{DEVICE}**")

    st.markdown("### Model 1 validation-selected operating point")
    st.write("Threshold: **0.55**")


# Footer
st.markdown(
    """
    <div style="text-align:center;padding:35px 0 10px;color:#8795a0;font-size:10px;letter-spacing:1.5px;">
        CHANGEINSIGHT · BI-TEMPORAL REMOTE SENSING · AI FOR A CHANGING WORLD
    </div>
    """,
    unsafe_allow_html=True,
)
