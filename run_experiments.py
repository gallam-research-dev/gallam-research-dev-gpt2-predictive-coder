# ================================================================
# run_experiments.py
# "How GPT-2 Learns to Be a Predictive Coder"
# 2편 전체 실험 통합 스크립트
#
# 사용법:
#   1. RUN_MODEL 설정 후 실행 (모델당 한 번씩, 런타임 초기화 필요)
#   2. 6개 모델 완료 후 gpt2_random 실행
#   3. analyze_results.py 실행 (통계 분석)
#   4. merge_figures.py 실행 (그림 합치기)
#
# 실행 순서:
#   RUN_MODEL = "gpt2"                      → 1번
#   RUN_MODEL = "gpt2-medium"               → 2번
#   RUN_MODEL = "gpt2-large"               → 3번
#   RUN_MODEL = "EleutherAI/gpt-neo-125M"  → 4번
#   RUN_MODEL = "EleutherAI/pythia-160m"   → 5번
#   RUN_MODEL = "EleutherAI/pythia-410m"   → 6번
#   RANDOM_INIT = True                      → 7번 (대조군)
# ================================================================

import subprocess, sys
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
install("transformer_lens")

import os, json, gc, warnings
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── ★ 설정 ★ ─────────────────────────────────────────────────
RUN_MODEL   = "gpt2"
# "gpt2" / "gpt2-medium" / "gpt2-large"
# "EleutherAI/gpt-neo-125M"
# "EleutherAI/pythia-160m" / "EleutherAI/pythia-410m"

RANDOM_INIT = False   # True = 무작위 초기화 대조군 실험
K_STEP      = 3       # 레이어 쌍 간격
OUTPUT_JSON = "directional_results.json"
# ─────────────────────────────────────────────────────────────

ML = {
    "gpt2":                    "GPT-2 (117M)",
    "gpt2-medium":             "GPT-2 Med (345M)",
    "gpt2-large":              "GPT-2 Large (774M)",
    "EleutherAI/gpt-neo-125M": "GPT-Neo (125M)",
    "EleutherAI/pythia-160m":  "Pythia (160M)",
    "EleutherAI/pythia-410m":  "Pythia (410M)",
}

# ── 100개 의미쌍 ──────────────────────────────────────────────
PAIRS = [
    # 수도 (20개)
    ("The capital of France is",   " Paris",     "The capital of Spain is",    " Madrid",    "capital"),
    ("The capital of Japan is",    " Tokyo",     "The capital of China is",    " Beijing",   "capital"),
    ("The capital of Italy is",    " Rome",      "The capital of Greece is",   " Athens",    "capital"),
    ("The capital of Germany is",  " Berlin",    "The capital of France is",   " Paris",     "capital"),
    ("The capital of Russia is",   " Moscow",    "The capital of Germany is",  " Berlin",    "capital"),
    ("The capital of Egypt is",    " Cairo",     "The capital of Russia is",   " Moscow",    "capital"),
    ("The capital of Greece is",   " Athens",    "The capital of Italy is",    " Rome",      "capital"),
    ("The capital of Spain is",    " Madrid",    "The capital of Japan is",    " Tokyo",     "capital"),
    ("The capital of China is",    " Beijing",   "The capital of Egypt is",    " Cairo",     "capital"),
    ("The capital of India is",    " Delhi",     "The capital of China is",    " Beijing",   "capital"),
    ("The capital of Norway is",   " Oslo",      "The capital of India is",    " Delhi",     "capital"),
    ("The capital of Poland is",   " Warsaw",    "The capital of Norway is",   " Oslo",      "capital"),
    ("The capital of Portugal is", " Lisbon",    "The capital of Poland is",   " Warsaw",    "capital"),
    ("The capital of Iraq is",     " Baghdad",   "The capital of Portugal is", " Lisbon",    "capital"),
    ("The capital of Iran is",     " Tehran",    "The capital of Iraq is",     " Baghdad",   "capital"),
    ("The capital of Cuba is",     " Havana",    "The capital of Iran is",     " Tehran",    "capital"),
    ("The capital of Peru is",     " Lima",      "The capital of Cuba is",     " Havana",    "capital"),
    ("The capital of Kenya is",    " Nairobi",   "The capital of Peru is",     " Lima",      "capital"),
    ("The capital of Israel is",   " Jerusalem", "The capital of Kenya is",    " Nairobi",   "capital"),
    ("The capital of Jordan is",   " Amman",     "The capital of Israel is",   " Jerusalem", "capital"),
    # 통화 (10개)
    ("The currency of Japan is the",   " yen",    "The currency of China is the",    " yuan",   "currency"),
    ("The currency of UK is the",      " pound",  "The currency of Japan is the",    " yen",    "currency"),
    ("The currency of China is the",   " yuan",   "The currency of UK is the",       " pound",  "currency"),
    ("The currency of Korea is the",   " won",    "The currency of USA is the",      " dollar", "currency"),
    ("The currency of USA is the",     " dollar", "The currency of Mexico is the",   " peso",   "currency"),
    ("The currency of Mexico is the",  " peso",   "The currency of Korea is the",    " won",    "currency"),
    ("The currency of Russia is the",  " ruble",  "The currency of Mexico is the",   " peso",   "currency"),
    ("The currency of India is the",   " rupee",  "The currency of Russia is the",   " ruble",  "currency"),
    ("The currency of Brazil is the",  " real",   "The currency of India is the",    " rupee",  "currency"),
    ("The currency of Turkey is the",  " lira",   "The currency of Brazil is the",   " real",   "currency"),
    # 언어 (15개)
    ("People in France speak",   " French",      "People in Germany speak",    " German",      "language"),
    ("People in Germany speak",  " German",      "People in France speak",     " French",      "language"),
    ("People in Japan speak",    " Japanese",    "People in China speak",      " Chinese",     "language"),
    ("People in China speak",    " Chinese",     "People in Japan speak",      " Japanese",    "language"),
    ("People in Italy speak",    " Italian",     "People in France speak",     " French",      "language"),
    ("People in Russia speak",   " Russian",     "People in Italy speak",      " Italian",     "language"),
    ("People in Spain speak",    " Spanish",     "People in Russia speak",     " Russian",     "language"),
    ("People in Greece speak",   " Greek",       "People in Spain speak",      " Spanish",     "language"),
    ("People in Portugal speak", " Portuguese",  "People in Greece speak",     " Greek",       "language"),
    ("People in Korea speak",    " Korean",      "People in Portugal speak",   " Portuguese",  "language"),
    ("People in Turkey speak",   " Turkish",     "People in Korea speak",      " Korean",      "language"),
    ("People in Poland speak",   " Polish",      "People in Turkey speak",     " Turkish",     "language"),
    ("People in Sweden speak",   " Swedish",     "People in Poland speak",     " Polish",      "language"),
    ("People in Norway speak",   " Norwegian",   "People in Sweden speak",     " Swedish",     "language"),
    ("People in Arabic speak",   " Arabic",      "People in Norway speak",     " Norwegian",   "language"),
    # 색깔 (12개)
    ("The color of grass is",       " green",    "The color of the sky is",       " blue",    "color"),
    ("The color of the sky is",     " blue",     "The color of grass is",         " green",   "color"),
    ("The color of snow is",        " white",    "The color of coal is",          " black",   "color"),
    ("The color of coal is",        " black",    "The color of snow is",          " white",   "color"),
    ("The color of blood is",       " red",      "The color of snow is",          " white",   "color"),
    ("The color of the sun is",     " yellow",   "The color of blood is",         " red",     "color"),
    ("The color of a banana is",    " yellow",   "The color of the ocean is",     " blue",    "color"),
    ("The color of the ocean is",   " blue",     "The color of a banana is",      " yellow",  "color"),
    ("The color of chocolate is",   " brown",    "The color of the sky is",       " blue",    "color"),
    ("The color of a carrot is",    " orange",   "The color of chocolate is",     " brown",   "color"),
    ("The color of milk is",        " white",    "The color of a carrot is",      " orange",  "color"),
    ("The color of a grape is",     " purple",   "The color of milk is",          " white",   "color"),
    # 반의어 (15개)
    ("The opposite of hot is",    " cold",   "The opposite of big is",     " small",  "antonym"),
    ("The opposite of big is",    " small",  "The opposite of hot is",     " cold",   "antonym"),
    ("The opposite of day is",    " night",  "The opposite of hot is",     " cold",   "antonym"),
    ("The opposite of night is",  " day",    "The opposite of big is",     " small",  "antonym"),
    ("The opposite of fast is",   " slow",   "The opposite of night is",   " day",    "antonym"),
    ("The opposite of slow is",   " fast",   "The opposite of fast is",    " slow",   "antonym"),
    ("The opposite of tall is",   " short",  "The opposite of slow is",    " fast",   "antonym"),
    ("The opposite of short is",  " tall",   "The opposite of tall is",    " short",  "antonym"),
    ("The opposite of dark is",   " light",  "The opposite of short is",   " tall",   "antonym"),
    ("The opposite of light is",  " dark",   "The opposite of dark is",    " light",  "antonym"),
    ("The opposite of hard is",   " soft",   "The opposite of light is",   " dark",   "antonym"),
    ("The opposite of soft is",   " hard",   "The opposite of hard is",    " soft",   "antonym"),
    ("The opposite of old is",    " young",  "The opposite of soft is",    " hard",   "antonym"),
    ("The opposite of young is",  " old",    "The opposite of old is",     " young",  "antonym"),
    ("The opposite of loud is",   " quiet",  "The opposite of young is",   " old",    "antonym"),
    # 과학/사실 (15개)
    ("Water freezes into",              " ice",      "Ice melts into",                   " water",    "science"),
    ("Ice melts into",                  " water",    "Water freezes into",               " ice",      "science"),
    ("The sun rises in the",            " east",     "The sun sets in the",              " west",     "science"),
    ("The sun sets in the",             " west",     "The sun rises in the",             " east",     "science"),
    ("Humans breathe in",               " oxygen",   "Plants absorb",                    " carbon",   "science"),
    ("The Earth orbits the",            " sun",      "The moon orbits the",              " Earth",    "science"),
    ("Water boils at one hundred",      " degrees",  "Water freezes at zero",            " degrees",  "science"),
    ("A triangle has",                  " three",    "A square has",                     " four",     "science"),
    ("A square has",                    " four",     "A triangle has",                   " three",    "science"),
    ("Two plus two equals",             " four",     "Three plus three equals",          " six",      "science"),
    ("Three plus three equals",         " six",      "Two plus two equals",              " four",     "science"),
    ("The fastest land animal is the",  " cheetah",  "The largest land animal is the",   " elephant", "science"),
    ("Bees produce",                    " honey",    "Cows produce",                     " milk",     "science"),
    ("Cows produce",                    " milk",     "Bees produce",                     " honey",    "science"),
    ("The planet closest to the sun is"," Mercury",  "The planet farthest from the sun is"," Neptune","science"),
    # 직업/장소 (13개)
    ("A doctor works in a",     " hospital",   "A teacher works in a",    " school",     "place"),
    ("A teacher works in a",    " school",     "A doctor works in a",     " hospital",   "place"),
    ("A chef works in a",       " kitchen",    "A doctor works in a",     " hospital",   "place"),
    ("A judge works in a",      " court",      "A chef works in a",       " kitchen",    "place"),
    ("A priest works in a",     " church",     "A judge works in a",      " court",      "place"),
    ("A soldier serves in the", " army",       "A priest works in a",     " church",     "place"),
    ("A farmer works on a",     " farm",       "A soldier serves in the", " army",       "place"),
    ("An actor performs on a",  " stage",      "A farmer works on a",     " farm",       "place"),
    ("A prisoner lives in a",   " prison",     "An actor performs on a",  " stage",      "place"),
    ("A student studies in a",  " school",     "A prisoner lives in a",   " prison",     "place"),
    ("A patient stays in a",    " hospital",   "A student studies in a",  " school",     "place"),
    ("A monk lives in a",       " monastery",  "A patient stays in a",    " hospital",   "place"),
    ("A sailor works on a",     " ship",       "A monk lives in a",       " monastery",  "place"),
]

# ── 유틸리티 ──────────────────────────────────────────────────
def clear():
    torch.cuda.empty_cache()
    gc.collect()

def is_single_token(model, word):
    try:
        model.to_single_token(word)
        return True
    except:
        return False

def get_zone(n_layers, norm_pos):
    if 0.10 <= norm_pos <= 0.20:   return "early_spike"
    elif 0.25 <= norm_pos <= 0.75: return "convergence"
    elif 0.80 <= norm_pos <= 0.95: return "late_spike"
    return "other"

# ── 방향성 패칭 실험 ──────────────────────────────────────────
def run_directional_patching(model, n_layers):
    print(f"  [방향성 패칭] K_STEP={K_STEP}, 레이어={n_layers}, 의미쌍={len(PAIRS)}개")

    layer_pairs  = [(l, l + K_STEP) for l in range(n_layers - K_STEP)]
    all_forward  = {(l, l+K_STEP): [] for l in range(n_layers - K_STEP)}
    all_backward = {(l, l+K_STEP): [] for l in range(n_layers - K_STEP)}
    valid_pairs  = 0

    for pair in PAIRS:
        clean_prompt, clean_token, corrupt_prompt, corrupt_token, _ = pair
        if not (is_single_token(model, clean_token) and
                is_single_token(model, corrupt_token)):
            continue
        try:
            ct    = model.to_single_token(clean_token)
            cot   = model.to_single_token(corrupt_token)
            ctoks = model.to_tokens(clean_prompt)[:,1:].to(device)
            xtoks = model.to_tokens(corrupt_prompt)[:,1:].to(device)
            sl    = min(ctoks.shape[1], xtoks.shape[1])
            ctoks, xtoks = ctoks[:,:sl], xtoks[:,:sl]

            hook_names = [f"blocks.{l}.hook_resid_post" for l in range(n_layers)]
            with torch.no_grad():
                _, clean_cache = model.run_with_cache(
                    ctoks, names_filter=lambda n: n in hook_names)

            with torch.no_grad():
                corrupt_out = model(xtoks)
            baseline = (corrupt_out[0,-1,ct] - corrupt_out[0,-1,cot]).item()
            del corrupt_out; clear()

            for (src_l, tgt_l) in layer_pairs:
                src_name = f"blocks.{src_l}.hook_resid_post"
                tgt_name = f"blocks.{tgt_l}.hook_resid_post"
                pos = xtoks.shape[1] - 1

                cv_src = clean_cache[src_name][0, pos, :].detach().clone()
                def make_fwd_hook(c, p):
                    def fn(v, hook): v = v.clone(); v[0,p,:] = c; return v
                    return fn
                with torch.no_grad():
                    fwd_out = model.run_with_hooks(
                        xtoks, fwd_hooks=[(tgt_name, make_fwd_hook(cv_src, pos))])
                fwd_effect = (fwd_out[0,-1,ct] - fwd_out[0,-1,cot]).item() - baseline
                del fwd_out; clear()

                cv_tgt = clean_cache[tgt_name][0, pos, :].detach().clone()
                def make_bwd_hook(c, p):
                    def fn(v, hook): v = v.clone(); v[0,p,:] = c; return v
                    return fn
                with torch.no_grad():
                    bwd_out = model.run_with_hooks(
                        xtoks, fwd_hooks=[(src_name, make_bwd_hook(cv_tgt, pos))])
                bwd_effect = (bwd_out[0,-1,ct] - bwd_out[0,-1,cot]).item() - baseline
                del bwd_out; clear()

                all_forward[(src_l, tgt_l)].append(fwd_effect)
                all_backward[(src_l, tgt_l)].append(bwd_effect)

            del clean_cache; clear()
            valid_pairs += 1
            if valid_pairs % 10 == 0:
                print(f"    진행: {valid_pairs}개 완료")

        except Exception as e:
            print(f"  스킵: {e}")

    print(f"  유효 의미쌍: {valid_pairs}개")

    results = []
    for (src_l, tgt_l) in layer_pairs:
        fwd_vals = all_forward[(src_l, tgt_l)]
        if not fwd_vals: continue
        fwd_mean  = float(np.mean(fwd_vals))
        bwd_mean  = float(np.mean(all_backward[(src_l, tgt_l)]))
        asymmetry = bwd_mean - fwd_mean
        norm_mid  = (src_l/n_layers + tgt_l/n_layers) / 2
        results.append({
            "src_layer": src_l, "tgt_layer": tgt_l,
            "norm_src":  src_l/n_layers, "norm_tgt": tgt_l/n_layers,
            "zone":      get_zone(n_layers, norm_mid),
            "fwd_mean":  fwd_mean, "bwd_mean": bwd_mean,
            "asymmetry": asymmetry, "n_pairs":  len(fwd_vals),
            "fwd_vals":  [float(v) for v in fwd_vals],
            "bwd_vals":  [float(v) for v in all_backward[(src_l, tgt_l)]],
            "asym_vals": [float(b-f) for b, f
                          in zip(all_backward[(src_l, tgt_l)], fwd_vals)],
        })
    return results, valid_pairs

def summarize_by_zone(results):
    summary = {}
    for z in ["early_spike", "convergence", "late_spike", "other"]:
        zr = [r for r in results if r["zone"] == z]
        if not zr: continue
        summary[z] = {
            "mean_asymmetry": float(np.mean([r["asymmetry"] for r in zr])),
            "mean_fwd":       float(np.mean([r["fwd_mean"]  for r in zr])),
            "mean_bwd":       float(np.mean([r["bwd_mean"]  for r in zr])),
        }
        print(f"  [{z}] 비대칭={summary[z]['mean_asymmetry']:+.4f}")
    return summary

def plot_results(results, model_name, n_layers):
    if not results: return
    norm_pos  = [(r["norm_src"]+r["norm_tgt"])/2 for r in results]
    asymmetry = [r["asymmetry"] for r in results]
    fwd_vals  = [r["fwd_mean"]  for r in results]
    bwd_vals  = [r["bwd_mean"]  for r in results]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    title = ML.get(model_name, model_name)
    if model_name == "gpt2_random":
        title = "GPT-2 Small (Random Init)"

    ax = axes[0]
    colors = ["#e74c3c" if r["zone"]=="early_spike"
              else "#3498db" if r["zone"]=="convergence"
              else "#e67e22" if r["zone"]=="late_spike"
              else "#95a5a6" for r in results]
    ax.bar(norm_pos, asymmetry, width=0.02, color=colors, alpha=0.8)
    ax.axhline(0, color="black", lw=1.2, ls="--", alpha=0.7)
    ax.axvspan(0.10, 0.20, alpha=0.08, color="red",    label="Early Spike")
    ax.axvspan(0.25, 0.75, alpha=0.08, color="blue",   label="Convergence")
    ax.axvspan(0.80, 0.95, alpha=0.08, color="orange", label="Late Spike")
    ax.set_xlabel("Normalized Layer Position", fontsize=11)
    ax.set_ylabel("Asymmetry Index (Backward - Forward)", fontsize=11)
    ax.set_title(f"{title}: Directional Asymmetry Index\n"
                 "Positive = Backward(top-down) stronger / "
                 "Negative = Forward(bottom-up) stronger",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, ls="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax2 = axes[1]
    ax2.plot(norm_pos, fwd_vals, "o-", color="#2ecc71", lw=2,
             markersize=4, label="Forward (l->l+k)", alpha=0.85)
    ax2.plot(norm_pos, bwd_vals, "s-", color="#e74c3c", lw=2,
             markersize=4, label="Backward (l+k->l)", alpha=0.85)
    ax2.axhline(0, color="black", lw=1, ls="--", alpha=0.5)
    ax2.axvspan(0.10, 0.20, alpha=0.08, color="red")
    ax2.axvspan(0.25, 0.75, alpha=0.08, color="blue")
    ax2.axvspan(0.80, 0.95, alpha=0.08, color="orange")
    ax2.set_xlabel("Normalized Layer Position", fontsize=11)
    ax2.set_ylabel("Patching Effect (logit diff)", fontsize=11)
    ax2.set_title("Forward vs Backward Patching Effect",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, ls="--")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    key = "gpt2_random" if RANDOM_INIT else model_name.replace("EleutherAI/", "")
    fname = f"directional_{key}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  {fname} 저장 완료")
    plt.close()
    return fname

# ── 메인 실행 ─────────────────────────────────────────────────
print(f"\n{'='*60}")
if RANDOM_INIT:
    print("무작위 초기화 대조군 실험")
else:
    print(f"모델: {RUN_MODEL}")
print(f"{'='*60}\n")

model = HookedTransformer.from_pretrained(RUN_MODEL)

if RANDOM_INIT:
    torch.manual_seed(42)
    for param in model.parameters():
        torch.nn.init.normal_(param, mean=0.0, std=0.02)
    print("무작위 초기화 완료 (seed=42, std=0.02)")

model.eval()
n_layers = model.cfg.n_layers
print(f"레이어 수: {n_layers}, K_STEP: {K_STEP}\n")

results, n_valid = run_directional_patching(model, n_layers)

print("\n── 구간별 비대칭 요약 ──")
zone_summary = summarize_by_zone(results)

fname = plot_results(results, RUN_MODEL, n_layers)

if os.path.exists(OUTPUT_JSON):
    with open(OUTPUT_JSON) as f:
        all_data = json.load(f)
else:
    all_data = {}

model_key = "gpt2_random" if RANDOM_INIT else RUN_MODEL.replace("EleutherAI/", "")
all_data[model_key] = {
    "n_layers":      n_layers,
    "k_step":        K_STEP,
    "n_valid_pairs": n_valid,
    "layer_results": results,
    "zone_summary":  zone_summary,
    "random_init":   RANDOM_INIT,
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(all_data, f, indent=2)
print(f"\n결과 저장: {OUTPUT_JSON}")

try:
    from google.colab import files
    files.download(OUTPUT_JSON)
    if fname and os.path.exists(fname):
        files.download(fname)
    print("다운로드 완료")
except ImportError:
    pass

del model; clear()
print("\n✅ 완료!")
