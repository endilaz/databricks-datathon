"""Limited architecture comparison using training and validation, never test labels."""
import argparse
import collections
import csv
import hashlib
import json
import random
from pathlib import Path
import torch

from .features import FeatureBuilder
from .io import read_jsonl, write_json
from .model import MatchingMLP, FeatureMLP, SemanticResidualMLP, make_semantic_config
from .train_pairs import direct_pair_loss
from .evaluate_pairs import preference_credit


def prepare_groups(features, path):
    rows = collections.defaultdict(list)
    for row in read_jsonl(path):
        sid = row["scenario_id"]
        if row["split"] != features.scenarios[sid]["split"]:
            raise ValueError("Incorrect split")
        if row["split"] not in ("train", "validation") or row["target_a"] is None:
            continue
        rows[sid].append(row)
    cache = collections.defaultdict(list)
    for sid, labels in rows.items():
        scenario = features.scenarios[sid]
        ids = scenario["candidate_ids"]; index = {fid: i for i, fid in enumerate(ids)}
        cache[scenario["split"]].append({"sid": sid,
            "vectors": features.group(sid, ids),
            "a": torch.tensor([index[r["facility_a"]] for r in labels]),
            "b": torch.tensor([index[r["facility_b"]] for r in labels]),
            "target": torch.tensor([r["target_a"] for r in labels])})
    return cache


def measure(model, groups, tie_weight=0.25, zero_patient=False, zero_match=False):
    model.eval(); losses=[]; credits=[]; macro=[]; tie_dev=[]
    with torch.no_grad():
        for group in groups:
            patient, hospital, match = group["vectors"]
            if zero_patient: patient = torch.zeros_like(patient)
            if zero_match: match = torch.zeros_like(match)
            s = model(patient, hospital, match)
            a, b, t = group["a"], group["b"], group["target"]
            losses.append(direct_pair_loss(s[a], s[b], t, tie_weight).item())
            gaps = s[a]-s[b]; query_credits=[]
            for gap, target in zip(gaps.tolist(), t.tolist()):
                if target == 0.5:
                    tie_dev.append(abs(torch.sigmoid(torch.tensor(gap)).item()-0.5))
                else:
                    credit = preference_credit(gap, target)
                    credits.append(credit);query_credits.append(credit)
            if query_credits:macro.append(sum(query_credits)/len(query_credits))
    return {"loss": sum(losses)/len(losses),
            "strict_agreement": sum(credits)/len(credits),
            "macro_agreement": sum(macro)/len(macro),
            "tie_preference_deviation": sum(tie_dev)/len(tie_dev) if tie_dev else None,
            "strict_pairs": len(credits), "groups": len(groups)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/pairwise_v2")
    parser.add_argument("--original-checkpoint", required=True)
    parser.add_argument("--out", default="runs/architecture_refinement")
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42,43,44])
    args=parser.parse_args()
    if args.max_epochs < 10 or args.patience < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use max-epochs >= 10, patience >= 1, and distinct seeds")
    torch.set_num_threads(1)
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise ValueError("Choose an empty output directory")
    out.mkdir(parents=True,exist_ok=True)
    root=Path(args.root);features=FeatureBuilder(root/"prepared")
    groups=prepare_groups(features,root/"labeled/pair_labels.jsonl")
    semantic_config=make_semantic_config(features)
    runs=[]
    for kind in ("original","features_only","semantic_residual"):
        for seed in args.seeds:
            random.seed(seed);torch.manual_seed(seed)
            if kind=="original":model=MatchingMLP(**features.dimensions)
            elif kind=="features_only":model=FeatureMLP()
            else:model=SemanticResidualMLP(**semantic_config)
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)
            run_out=out/(kind+"_seed_"+str(seed));run_out.mkdir()
            best=float("inf");best_epoch=None;best_state=None;stale=0;history=[]
            training=list(groups["train"])
            for epoch in range(1,args.max_epochs+1):
                model.train();random.shuffle(training);total=0.0
                for group in training:
                    optimizer.zero_grad(set_to_none=True)
                    s=model(*group["vectors"])
                    loss=direct_pair_loss(s[group["a"]],s[group["b"]],group["target"],0.25)
                    if not torch.isfinite(loss):raise ValueError("Nonfinite training loss")
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);optimizer.step()
                    total+=loss.item()
                val=measure(model,groups["validation"])
                history.append({"epoch":epoch,"training_loss":total/len(training),"validation":val})
                if val["loss"]<best-1e-4:
                    best=val["loss"];best_epoch=epoch;stale=0
                    best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
                else:stale+=1
                if epoch>=10 and stale>=args.patience:break
            model.load_state_dict(best_state)
            result={"architecture":kind,"seed":seed,"parameters":sum(p.numel() for p in model.parameters()),
                    "best_epoch":best_epoch,"epochs_run":epoch,"selection":"minimum validation logistic loss",
                    "train":measure(model,groups["train"]),"validation":measure(model,groups["validation"]),
                    "history":history,"test_evaluated":False}
            write_json(run_out/"training_run.json",{**result,"tie_weight":0.25})
            torch.save({"architecture":kind,"model_state":best_state,"config":model.config,
                        "vocabulary":features.vocab,"seed":seed,"loss":"direct_pair_logistic"},run_out/"model.pt")
            runs.append({k:v for k,v in result.items() if k!="history"})
            print(json.dumps(runs[-1]),flush=True)
    original_path=Path(args.original_checkpoint)
    original_hash=hashlib.sha256(original_path.read_bytes()).hexdigest()
    checkpoint=torch.load(original_path,map_location="cpu",weights_only=True)
    original=MatchingMLP(**checkpoint["config"]);original.load_state_dict(checkpoint["model_state"])
    diagnostic={"full":measure(original,groups["validation"]),
                "patient_id_zeroed":measure(original,groups["validation"],zero_patient=True),
                "match_features_zeroed":measure(original,groups["validation"],zero_match=True),
                "limit":"Frozen-input removal is out of distribution; diagnostic sensitivity, not causal attribution."}
    assert hashlib.sha256(original_path.read_bytes()).hexdigest()==original_hash
    write_json(out/"comparison.json",{"runs":runs,"original_checkpoint_sha256":original_hash,
        "frozen_input_diagnostics":diagnostic,"test_evaluated":False,
        "limits":["Validation is reused for model/epoch selection, not an independent final test.",
                  "Agreement is against provisional AI labels, not clinical accuracy.",
                  "Only single-symptom examples exist; multi-symptom interactions remain untested."]})


if __name__=="__main__":main()
