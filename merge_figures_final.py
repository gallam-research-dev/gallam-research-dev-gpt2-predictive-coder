# ================================================================
# merge_figures.py
# 개별 모델 그림을 논문용 패널로 합치기
# 출력:
#   directional_comparison.png  → 논문 Figure 1 (GPT-2 3개)
#   directional_appendix.png    → Appendix (GPT-Neo, Pythia 3개)
# ================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

FILES = {
    "gpt2":        "directional_gpt2.png",
    "gpt2-medium": "directional_gpt2-medium.png",
    "gpt2-large":  "directional_gpt2-large.png",
    "gpt-neo":     "directional_gpt-neo-125M.png",
    "pythia-160m": "directional_pythia-160m.png",
    "pythia-410m": "directional_pythia-410m.png",
}

LABELS = {
    "gpt2":        "GPT-2 Small (117M)",
    "gpt2-medium": "GPT-2 Med (345M)",
    "gpt2-large":  "GPT-2 Large (774M)",
    "gpt-neo":     "GPT-Neo (125M)",
    "pythia-160m": "Pythia (160M)",
    "pythia-410m": "Pythia (410M)",
}

def crop_top_half(img_array):
    h = img_array.shape[0]
    return img_array[:h//2, :, :]

def make_panel(keys, output_fname, suptitle):
    fig, axes = plt.subplots(1, len(keys), figsize=(6*len(keys), 5))
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    for ax, key in zip(axes, keys):
        img = np.array(Image.open(FILES[key]))
        ax.imshow(crop_top_half(img))
        ax.set_title(LABELS[key], fontsize=12, fontweight="bold")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_fname, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  {output_fname} 저장 완료")
    plt.close()

make_panel(
    ["gpt2", "gpt2-medium", "gpt2-large"],
    "directional_comparison.png",
    "Directional Asymmetry Index: GPT-2 Family\n"
    "Positive = Backward (top-down) stronger / "
    "Negative = Forward (bottom-up) stronger"
)

make_panel(
    ["gpt-neo", "pythia-160m", "pythia-410m"],
    "directional_appendix.png",
    "Directional Asymmetry Index: GPT-Neo and Pythia (Control Models)\n"
    "Positive = Backward (top-down) stronger / "
    "Negative = Forward (bottom-up) stronger"
)

try:
    from google.colab import files
    files.download("directional_comparison.png")
    files.download("directional_appendix.png")
except ImportError:
    pass

print("\n✅ 완료!")
