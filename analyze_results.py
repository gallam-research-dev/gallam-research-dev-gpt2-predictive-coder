# ================================================================
# analyze_results.py
# run_experiments.py 완료 후 실행
# 의미쌍 단위 통계 분석 + Bootstrap CI
# ================================================================

import json
import numpy as np
from scipy import stats

JSON_FILE   = "directional_results.json"
N_BOOTSTRAP = 10000
SEED        = 42
np.random.seed(SEED)

with open(JSON_FILE) as f:
    data = json.load(f)

ML = {
    "gpt2":         "GPT-2 Small (117M)",
    "gpt2-medium":  "GPT-2 Med (345M)",
    "gpt2-large":   "GPT-2 Large (774M)",
    "gpt-neo-125M": "GPT-Neo (125M)",
    "pythia-160m":  "Pythia (160M)",
    "pythia-410m":  "Pythia (410M)",
    "gpt2_random":  "GPT-2 Random Init",
}

MODEL_ORDER = ["gpt2", "gpt2-medium", "gpt2-large",
               "gpt-neo-125M", "pythia-160m", "pythia-410m", "gpt2_random"]

def get_pair_level_asymmetry(results, zone="convergence"):
    zone_results = [r for r in results if r["zone"] == zone]
    if not zone_results or "asym_vals" not in zone_results[0]:
        return []
    n_pairs = len(zone_results[0]["asym_vals"])
    pair_asym = []
    for idx in range(n_pairs):
        vals = [r["asym_vals"][idx] for r in zone_results
                if idx < len(r["asym_vals"])]
        if vals:
            pair_asym.append(float(np.mean(vals)))
    return pair_asym

def bootstrap_ci(vals, n_boot=N_BOOTSTRAP, ci=95):
    vals = np.array(vals)
    boot = [np.mean(np.random.choice(vals, len(vals), replace=True))
            for _ in range(n_boot)]
    lo = np.percentile(boot, (100-ci)/2)
    hi = np.percentile(boot, 100-(100-ci)/2)
    return float(np.mean(vals)), lo, hi

def bootstrap_diff_ci(a, b, n_boot=N_BOOTSTRAP, ci=95):
    a, b = np.array(a), np.array(b)
    boot = [np.mean(np.random.choice(a, len(a), replace=True)) -
            np.mean(np.random.choice(b, len(b), replace=True))
            for _ in range(n_boot)]
    lo = np.percentile(boot, (100-ci)/2)
    hi = np.percentile(boot, 100-(100-ci)/2)
    diff = float(np.mean(a) - np.mean(b))
    sig  = "*" if (lo > 0 or hi < 0) else "ns"
    return diff, lo, hi, sig

# ── 1. 구간별 t-test ──────────────────────────────────────────
print("=" * 70)
print("수렴 구간 비대칭 지수 — 의미쌍 단위 분석")
print("=" * 70)
print(f"{'Model':<25} {'Mean':>8} {'95% CI':>22} {'t':>7} {'p':>8} {'Sig':>5} {'n':>5}")
print("-" * 70)

gpt2_all_pairs = []

for model_key in MODEL_ORDER:
    if model_key not in data:
        continue
    results = data[model_key].get("layer_results", [])
    if not results or "asym_vals" not in results[0]:
        print(f"{ML.get(model_key):<25} {'데이터 없음'}")
        continue

    pair_vals = get_pair_level_asymmetry(results, zone="convergence")
    if len(pair_vals) < 2:
        continue

    mean, lo, hi = bootstrap_ci(pair_vals)
    t, p = stats.ttest_1samp(pair_vals, 0)
    n    = len(pair_vals)
    sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    boot_sig = "*" if (lo > 0 or hi < 0) else "ns"
    final_sig = sig if sig != "ns" else boot_sig

    print(f"{ML.get(model_key, model_key):<25} {mean:>+8.3f} "
          f"[{lo:>+8.3f}, {hi:>+8.3f}] {t:>+7.3f} {p:>8.4f} {final_sig:>5} {n:>5}")

    if "gpt2" in model_key and model_key != "gpt2_random":
        gpt2_all_pairs += pair_vals

print()

# ── 2. 훈련 vs 무작위 비교 ────────────────────────────────────
print("=" * 70)
print("훈련된 GPT-2 vs 무작위 초기화 비교 (수렴 구간)")
print("=" * 70)

random_results = data.get("gpt2_random", {}).get("layer_results", [])
if random_results and "asym_vals" in random_results[0]:
    random_pairs = get_pair_level_asymmetry(random_results, zone="convergence")
    for model_key in ["gpt2", "gpt2-medium", "gpt2-large"]:
        if model_key not in data: continue
        results = data[model_key].get("layer_results", [])
        if not results or "asym_vals" not in results[0]: continue
        trained_pairs = get_pair_level_asymmetry(results, zone="convergence")
        if not trained_pairs: continue
        diff, lo, hi, sig = bootstrap_diff_ci(trained_pairs, random_pairs)
        cohen_d = (np.mean(trained_pairs) - np.mean(random_pairs)) / np.sqrt(
            (np.std(trained_pairs, ddof=1)**2 + np.std(random_pairs, ddof=1)**2) / 2)
        print(f"{ML[model_key]:<25} diff={diff:>+.3f} "
              f"[{lo:>+.3f}, {hi:>+.3f}] {sig}  d={cohen_d:.2f}")
else:
    print("gpt2_random 데이터 없음 — RANDOM_INIT=True로 재실행 필요")

print()

# ── 3. GPT-2 vs GPT-Neo/Pythia ────────────────────────────────
print("=" * 70)
print("GPT-2 계열 vs GPT-Neo/Pythia 비교 (수렴 구간)")
print("=" * 70)

if gpt2_all_pairs:
    for model_key in ["gpt-neo-125M", "pythia-160m", "pythia-410m"]:
        if model_key not in data: continue
        results = data[model_key].get("layer_results", [])
        if not results or "asym_vals" not in results[0]: continue
        other_pairs = get_pair_level_asymmetry(results, zone="convergence")
        if not other_pairs: continue
        diff, lo, hi, sig = bootstrap_diff_ci(gpt2_all_pairs, other_pairs)
        cohen_d = (np.mean(gpt2_all_pairs) - np.mean(other_pairs)) / np.sqrt(
            (np.std(gpt2_all_pairs, ddof=1)**2 + np.std(other_pairs, ddof=1)**2) / 2)
        label = f"GPT-2 family vs {ML[model_key]}"
        print(f"{label:<40} diff={diff:>+.3f} "
              f"[{lo:>+.3f}, {hi:>+.3f}] {sig}  d={cohen_d:.2f}")

print()
print("* p<.05  ** p<.01  *** p<.001  ns = not significant")
print(f"Bootstrap: n={N_BOOTSTRAP}, Seed={SEED}")
print("\n✅ 완료!")
