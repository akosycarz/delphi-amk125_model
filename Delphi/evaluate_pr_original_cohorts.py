#!/usr/bin/env python3
"""Add precision--recall evaluation to Delphi's original AUC cohorts.

This intentionally reproduces evaluate_auc.py's cohort definition: validation
patients, left-selected/truncated sequences, ever-observed cases versus
never-observed controls, Female/Male token masks, 5-year prediction-age bands,
and one randomly selected observation per patient per band.  It does *not*
implement a fixed-horizon incident-disease cohort.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate_auc import get_common_diseases, load_dataset_labels
from model import Delphi, DelphiConfig
from run_evaluation import average_precision, binary_curves
from utils import get_batch, get_p2i


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-path", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--block-size", type=int, default=80,
                   help="Original evaluation used 80; use checkpoint size with 0")
    p.add_argument("--offset-days", type=float, default=.1)
    p.add_argument("--age-start", type=int, default=40)
    p.add_argument("--age-stop", type=int, default=80)
    p.add_argument("--age-step", type=int, default=5)
    p.add_argument("--filter-min-total", type=int, default=100,
                   help="Eligibility is external/truncated token count > this value")
    p.add_argument("--selected-icd-codes", nargs="*")
    p.add_argument("--n-outcomes", type=int, default=15)
    p.add_argument("--min-band-cases", type=int, default=2)
    p.add_argument("--min-band-controls", type=int, default=2)
    p.add_argument("--dataset-subset-size", type=int, default=-1)
    p.add_argument("--no-event-token-rate", type=int, default=5)
    p.add_argument("--disease-chunk-size", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--save-predictions", action="store_true")
    return p.parse_args()


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = Delphi(DelphiConfig(**checkpoint["model_args"]))
    state = checkpoint["model"]
    for key in list(state):
        if key.startswith("_orig_mod."):
            state[key[10:]] = state.pop(key)
    model.load_state_dict(state)
    return model.eval().to(device)


def select_diseases(labels, requested, minimum, number):
    eligible = get_common_diseases(labels, minimum)
    table = labels[labels["index"].isin(eligible)].copy()
    if requested:
        wanted = {str(x).upper().strip() for x in requested}
        table["icd_code"] = table["name"].astype(str).str.split("::").str[-1].str.upper()
        table = table[table.icd_code.isin(wanted)]
        missing = sorted(wanted - set(table.icd_code))
        if missing:
            print("WARNING: requested/eligible codes not found: " + ", ".join(missing))
        return table
    return table.nlargest(number, "count")


def one_per_patient_band(input_ages, targets, target_ages, scores, disease_token,
                         offset, age_groups, age_step, rng):
    """Return the exact per-band rows used by the original AUC comparison."""
    cases = np.where(targets == disease_token)
    if len(cases[0]) < 2:
        return []
    never_case = ~(targets == disease_token).any(axis=1)
    controls = np.where((targets != disease_token) & never_case[:, None])
    patients = np.r_[cases[0], controls[0]]
    positions = np.r_[cases[1], controls[1]]
    labels = np.r_[np.ones(len(cases[0]), dtype=np.int8),
                   np.zeros(len(controls[0]), dtype=np.int8)]
    # Prediction input ages correspond to the x sequence. target_ages is b.
    # This repeats evaluate_auc.py's strictly-earlier precomputation.
    pred_idx = (target_ages[patients, positions, None] - offset >
                input_ages[patients]).sum(axis=1) - 1
    valid = pred_idx >= 0
    patients, positions, labels, pred_idx = (x[valid] for x in
                                             (patients, positions, labels, pred_idx))
    prediction_age = input_ages[patients, pred_idx] / 365.25
    values = scores[patients, pred_idx]
    rows = []
    for age in age_groups:
        in_band = (prediction_age >= age) & (prediction_age < age + age_step)
        candidates = np.flatnonzero(in_band)
        if not len(candidates):
            continue
        shuffled = candidates[rng.permutation(len(candidates))]
        _, first = np.unique(patients[shuffled], return_index=True)
        chosen = shuffled[first]
        rows.append(pd.DataFrame({
            "patient_index": patients[chosen], "target_position": positions[chosen],
            "prediction_position": pred_idx[chosen], "prediction_age": prediction_age[chosen],
            "age_group": f"{age}-{age + age_step}", "y_true": labels[chosen],
            "y_score": values[chosen],
        }))
    return rows


def curve_frame(frame, model, token, code, name, sex, age_group, horizon):
    y = frame.y_true.to_numpy(dtype=np.int8)
    score = frame.y_score.to_numpy(dtype=float)
    npos, nneg = int(y.sum()), int((1-y).sum())
    _, _, precision, recall, _ = binary_curves(y, score)
    prevalence = npos / (npos + nneg)
    ap = average_precision(precision, recall)
    curve = pd.DataFrame({
        "model": model, "token_id": token, "icd_code": code, "disease": name,
        "sex": sex, "age_group": age_group, "horizon_months": horizon,
        "recall": recall, "precision": precision, "average_precision": ap,
        "prevalence": prevalence, "n_positive": npos, "n_negative": nneg,
    })
    summary = {"model": model, "token_id": token, "icd_code": code,
               "disease": name, "sex": sex, "age_group": age_group,
               "average_precision": ap, "prevalence": prevalence,
               "ap_minus_prevalence": ap-prevalence,
               "n_positive": npos, "n_negative": nneg}
    return curve, summary


def main():
    a = arguments()
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    output = a.output_dir.expanduser().resolve();output.mkdir(parents=True, exist_ok=True)
    model = load_model(a.checkpoint.expanduser().resolve(), a.device)
    block_size = int(model.config.block_size) if a.block_size == 0 else a.block_size
    if block_size > int(model.config.block_size):
        raise ValueError("Evaluation block size cannot exceed checkpoint block size")
    raw = np.fromfile(a.input_path.expanduser()/f"{a.split}.bin", dtype=np.uint32)
    raw = raw.reshape(-1, 3).astype(np.int64);p2i=get_p2i(raw)
    n=len(p2i) if a.dataset_subset_size == -1 else min(a.dataset_subset_size,len(p2i))
    batch=get_batch(range(n),raw,p2i,select="left",block_size=block_size,device="cpu",
                    padding="random",no_event_token_rate=a.no_event_token_rate)
    input_ages=batch[1].numpy();targets=batch[2].numpy();target_ages=batch[3].numpy()
    labels=load_dataset_labels(a.input_path, batch[2])
    diseases=select_diseases(labels,a.selected_icd_codes,a.filter_min_total,a.n_outcomes)
    if diseases.empty:raise RuntimeError("No eligible diseases selected")
    # Reference cohorts use literal shifted vocabulary IDs 2 and 3.
    sex_masks={"Full":np.ones(n,dtype=bool),
               "Female":(batch[0].numpy()==2).any(1),
               "Male":(batch[0].numpy()==3).any(1)}
    print("Sex populations: "+", ".join(f"{k}={v.sum():,}" for k,v in sex_masks.items()))
    age_groups=np.arange(a.age_start,a.age_stop,a.age_step)
    all_curves=[];all_summaries=[];all_predictions=[]
    disease_rows=diseases.reset_index(drop=True)
    for chunk_start in range(0,len(disease_rows),a.disease_chunk_size):
        chunk=disease_rows.iloc[chunk_start:chunk_start+a.disease_chunk_size]
        ids=chunk["index"].astype(int).tolist();parts=[]
        with torch.inference_mode():
            for start in range(0,n,a.batch_size):
                end=min(start+a.batch_size,n)
                x,ages,y,b=[v[start:end].to(a.device) for v in batch]
                parts.append(model(x,ages,y,b)[0][:,:,ids].float().cpu().numpy())
        logits=np.concatenate(parts)
        for j,(_,disease) in enumerate(chunk.iterrows()):
            token=int(disease["index"]);name=str(disease["name"])
            code=name.split("::")[-1].upper()
            sex_band_frames={}
            for sex,mask in sex_masks.items():
                frames=one_per_patient_band(input_ages[mask],targets[mask],target_ages[mask],logits[mask,:,j],
                                            token,a.offset_days,age_groups,a.age_step,rng)
                kept=[]
                for frame in frames:
                    npos=int(frame.y_true.sum());nneg=len(frame)-npos
                    if npos<a.min_band_cases or nneg<a.min_band_controls:continue
                    band=frame.age_group.iloc[0]
                    curve,summary=curve_frame(frame,a.model_name,token,code,name,sex,band,np.nan)
                    all_curves.append(curve);all_summaries.append(summary);kept.append(frame)
                    if a.save_predictions:
                        q=frame.copy();q["model"]=a.model_name;q["token_id"]=token
                        q["icd_code"]=code;q["sex"]=sex;all_predictions.append(q)
                sex_band_frames[sex]=kept
                if kept:
                    pooled=pd.concat(kept,ignore_index=True)
                    curve,summary=curve_frame(pooled,a.model_name,token,code,name,sex,"Full",np.nan)
                    all_curves.append(curve);all_summaries.append(summary)
                    band_ap=[x["average_precision"] for x in all_summaries
                             if x["token_id"]==token and x["sex"]==sex and x["age_group"]!="Full"]
                    summary["macro_average_precision_across_age_bands"]=float(np.mean(band_ap))
        del logits
    if not all_curves:raise RuntimeError("No disease/sex/age cohort met minimum sizes")
    pd.concat(all_curves,ignore_index=True).to_csv(output/"pr_curves.csv",index=False)
    pd.DataFrame(all_summaries).to_csv(output/"pr_summary.csv",index=False)
    if a.save_predictions and all_predictions:
        pd.concat(all_predictions,ignore_index=True).to_csv(output/"pr_predictions.csv",index=False)
    pd.DataFrame([{"model":a.model_name,"split":a.split,"block_size":block_size,
        "offset_days":a.offset_days,"age_groups":f"{a.age_start}-{a.age_stop} by {a.age_step}",
        "cohort_definition":"ever observed case versus never observed control",
        "deduplication":"one random observation per patient per sex/disease/age band",
        "seed":a.seed}]).to_csv(output/"pr_run_config.csv",index=False)
    print(f"Created {output/'pr_curves.csv'}")


if __name__=="__main__":main()
