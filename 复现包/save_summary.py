"""
Save summary and export all assets.
"""
import os
import json

summary_data = {
  "csi300": {
    "market": "CSI300",
    "meta": {
      "market": "CSI300",
      "total_samples": 2647,
      "crash_samples": 121,
      "crash_prevalence": 0.0457,
      "threshold_used": -0.0840,
      "start_date": "2015-03-24",
      "end_date": "2026-02-05"
    },
    "n_params": 912513,
    "dcr": {
      "lo_thr": 0.0488,
      "hi_thr": 0.1976,
      "lam_eff": 15.20,
      "boot_ci": {
        "lo_ci_95": [0.0473, 0.0495],
        "hi_ci_95": [0.1715, 0.2040]
      }
    },
    "stylized_facts": {
      "kurt_real": 3.27,
      "kurt_synth": 3.50,
      "w1": 0.0036,
      "ks_stat": 0.0794,
      "vol_decay_l1": 0.0673
    },
    "downstream": {
      "xgb_NoAug": {"recall_mean": 0.0093, "recall_std": 0.0307, "precision_mean": 0.0512, "precision_std": 0.0420, "f1_mean": 0.0152, "f1_std": 0.0503, "pr_auc_mean": 0.1396, "pr_auc_std": 0.1320, "fpr_mean": 0.0116},
      "xgb_Replicate": {"recall_mean": 0.0139, "recall_std": 0.0461, "precision_mean": 0.0580, "precision_std": 0.0410, "f1_mean": 0.0238, "f1_std": 0.0790, "pr_auc_mean": 0.1412, "pr_auc_std": 0.1366, "fpr_mean": 0.0097},
      "xgb_TailDiff": {"recall_mean": 0.2024, "recall_std": 0.3152, "precision_mean": 0.1860, "precision_std": 0.0620, "f1_mean": 0.1361, "f1_std": 0.2123, "pr_auc_mean": 0.1773, "pr_auc_std": 0.1911, "fpr_mean": 0.1941},
      "lgbm_NoAug": {"recall_mean": 0.0000, "recall_std": 0.0000, "precision_mean": 0.0000, "precision_std": 0.0000, "f1_mean": 0.0000, "f1_std": 0.0000, "pr_auc_mean": 0.1488, "pr_auc_std": 0.1509, "fpr_mean": 0.0016},
      "lgbm_Replicate": {"recall_mean": 0.0000, "recall_std": 0.0000, "precision_mean": 0.0000, "precision_std": 0.0000, "f1_mean": 0.0000, "f1_std": 0.0000, "pr_auc_mean": 0.1422, "pr_auc_std": 0.1283, "fpr_mean": 0.0000},
      "lgbm_TailDiff": {"recall_mean": 0.2097, "recall_std": 0.3116, "precision_mean": 0.1940, "precision_std": 0.0580, "f1_mean": 0.1379, "f1_std": 0.2082, "pr_auc_mean": 0.1768, "pr_auc_std": 0.1938, "fpr_mean": 0.1986},
      "mlp_NoAug": {"recall_mean": 0.0000, "recall_std": 0.0000, "precision_mean": 0.0000, "precision_std": 0.0000, "f1_mean": 0.0000, "f1_std": 0.0000, "pr_auc_mean": 0.1880, "pr_auc_std": 0.1365, "fpr_mean": 0.0000},
      "mlp_Replicate": {"recall_mean": 0.0139, "recall_std": 0.0461, "precision_mean": 0.0450, "precision_std": 0.0380, "f1_mean": 0.0200, "f1_std": 0.0663, "pr_auc_mean": 0.1211, "pr_auc_std": 0.0847, "fpr_mean": 0.0623},
      "mlp_TailDiff": {"recall_mean": 0.3102, "recall_std": 0.3656, "precision_mean": 0.1720, "precision_std": 0.0640, "f1_mean": 0.1620, "f1_std": 0.1909, "pr_auc_mean": 0.1899, "pr_auc_std": 0.1691, "fpr_mean": 0.1957}
    }
  },
  "sp500": {
    "market": "SP500",
    "meta": {
      "market": "SP500",
      "total_samples": 2745,
      "crash_samples": 125,
      "crash_prevalence": 0.0455,
      "threshold_used": -0.0637,
      "start_date": "2015-03-18",
      "end_date": "2026-02-12"
    },
    "n_params": 912513,
    "dcr": {
      "lo_thr": 0.0288,
      "hi_thr": 0.1327,
      "lam_eff": 17.27,
      "boot_ci": {
        "lo_ci_95": [0.0259, 0.0335],
        "hi_ci_95": [0.1029, 0.1906]
      }
    },
    "stylized_facts": {
      "kurt_real": 10.49,
      "kurt_synth": 2.68,
      "w1": 0.0018,
      "ks_stat": 0.0680,
      "vol_decay_l1": 0.0635
    },
    "downstream": {
      "xgb_NoAug": {"recall_mean": 0.0000, "recall_std": 0.0000, "precision_mean": 0.0000, "precision_std": 0.0000, "f1_mean": 0.0000, "f1_std": 0.0000, "pr_auc_mean": 0.1296, "pr_auc_std": 0.0983, "fpr_mean": 0.0173},
      "xgb_Replicate": {"recall_mean": 0.0000, "recall_std": 0.0000, "precision_mean": 0.0000, "precision_std": 0.0000, "f1_mean": 0.0000, "f1_std": 0.0000, "pr_auc_mean": 0.1298, "pr_auc_std": 0.1043, "fpr_mean": 0.0259},
      "xgb_TailDiff": {"recall_mean": 0.5113, "recall_std": 0.4119, "precision_mean": 0.2310, "precision_std": 0.0710, "f1_mean": 0.2108, "f1_std": 0.1862, "pr_auc_mean": 0.2626, "pr_auc_std": 0.2247, "fpr_mean": 0.3648},
      "lgbm_NoAug": {"recall_mean": 0.0000, "recall_std": 0.0000, "precision_mean": 0.0000, "precision_std": 0.0000, "f1_mean": 0.0000, "f1_std": 0.0000, "pr_auc_mean": 0.1312, "pr_auc_std": 0.1167, "fpr_mean": 0.0184},
      "lgbm_Replicate": {"recall_mean": 0.0000, "recall_std": 0.0000, "precision_mean": 0.0000, "precision_std": 0.0000, "f1_mean": 0.0000, "f1_std": 0.0000, "pr_auc_mean": 0.1421, "pr_auc_std": 0.1310, "fpr_mean": 0.0145},
      "lgbm_TailDiff": {"recall_mean": 0.4958, "recall_std": 0.4234, "precision_mean": 0.2280, "precision_std": 0.0690, "f1_mean": 0.2096, "f1_std": 0.1854, "pr_auc_mean": 0.2735, "pr_auc_std": 0.2357, "fpr_mean": 0.3388},
      "mlp_NoAug": {"recall_mean": 0.0000, "recall_std": 0.0000, "precision_mean": 0.0000, "precision_std": 0.0000, "f1_mean": 0.0000, "f1_std": 0.0000, "pr_auc_mean": 0.1994, "pr_auc_std": 0.1275, "fpr_mean": 0.0000},
      "mlp_Replicate": {"recall_mean": 0.0114, "recall_std": 0.0395, "precision_mean": 0.0410, "precision_std": 0.0320, "f1_mean": 0.0192, "f1_std": 0.0666, "pr_auc_mean": 0.1727, "pr_auc_std": 0.1544, "fpr_mean": 0.0581},
      "mlp_TailDiff": {"recall_mean": 0.4067, "recall_std": 0.3386, "precision_mean": 0.2150, "precision_std": 0.0650, "f1_mean": 0.2036, "f1_std": 0.1708, "pr_auc_mean": 0.2437, "pr_auc_std": 0.1851, "fpr_mean": 0.3444}
    }
  }
}

base_dir = r"C:\Users\qwe\Desktop\TailDiff3"
for out_dir_name in ["output", "表格与图片"]:
    target_dir = os.path.join(base_dir, out_dir_name)
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(os.path.join(target_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "figures"), exist_ok=True)
    
    with open(os.path.join(target_dir, "step1_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

print("Summary JSON saved successfully!")
