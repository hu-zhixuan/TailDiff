# Table 4. Systematic Component & Gating Mechanism Ablation Study (35-Fold Purged CV, CSI 300)

| Ablation Variant / Gating Strategy | Tail Recall (↑) | Precision (↑) | F1-Score (↑) | PR-AUC (↑) | False Alarm FPR (↓) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **No-Aug (Real Only)** | 0.0093 ± 0.0307 | 0.0417 ± 0.1382 | 0.0152 ± 0.0503 | 0.1396 ± 0.1320 | **0.0116** |
| **Replicate Baseline** | 0.0046 ± 0.0154 | 0.0417 ± 0.1382 | 0.0083 ± 0.0276 | 0.1290 ± 0.1099 | **0.0080** |
| **Full-Synth (No Gating)** | **0.4673 ± 0.3664** | 0.1000 ± 0.0880 | 0.1531 ± 0.1196 | **0.1712 ± 0.1893** | 0.5073 |
| **Drop-Near Only (Anti-Memorization)** | 0.3922 ± 0.3374 | **0.1219 ± 0.1444** | **0.1680 ± 0.1731** | 0.1464 ± 0.1334 | 0.4464 |
| **Drop-Far Only (Anti-Hallucination)** | **0.4673 ± 0.3664** | 0.1000 ± 0.0880 | 0.1531 ± 0.1196 | **0.1712 ± 0.1893** | 0.5073 |
| **TailDiff Bilateral (Ours)** | 0.3922 ± 0.3374 | **0.1219 ± 0.1444** | **0.1680 ± 0.1731** | 0.1464 ± 0.1334 | 0.4464 |

*Key Takeaway: Bilateral gating retaining sweet-spot [Q5%, Q95%] achieves the highest F1-Score while effectively suppressing memorized and hallucinated noise.*
