"""Train LSTM / 1D-CNN / PatchTST with pinball loss for the day-ahead task (see windows.py).
Selection uses validation only; test predictions are saved for the single final evaluation (final_eval.py).

Run:  python src/train.py --model cnn [--weather] [--future|--strict] [--intrahour] [--epochs 50] [--seed 0]
      [--lr 1e-3] [--dropout 0.1] [--hidden 128] [--tag name]
"""
import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import windows as W
from evaluate import metrics, table, to_units
from models import build, pinball_loss


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict(model, X: np.ndarray, F: np.ndarray, dev, bs: int = 256) -> np.ndarray:
    model.eval()
    out = [model(torch.from_numpy(X[i:i + bs]).to(dev), torch.from_numpy(F[i:i + bs]).to(dev)).cpu().numpy()
           for i in range(0, len(X), bs)]
    return np.concatenate(out)


def run_name(args) -> str:
    return (f"{args.model}{'_weather' if args.weather else ''}{'_strict' if args.strict else '_future' if args.future else ''}"
            f"{'_intra' if args.intrahour else ''}{('_' + args.tag) if args.tag else ''}_s{args.seed}")


def feature_sets(args):
    feats = W.BASE + (W.WEATHER if args.weather else []) + (W.INTRAHOUR if args.intrahour else [])
    return feats, (W.STRICT_FUTURE if args.strict else W.FUTURE if args.future else [])


def model_kwargs(args) -> dict:
    hp = {"dropout": args.dropout}
    if args.model == "lstm":
        hp["hidden"] = args.hidden
    elif args.model == "cnn":
        hp["filters"] = args.hidden
    else:
        hp["d_model"] = args.hidden
    return hp


def scaler_path(args):
    tag = f"{'w' if args.weather else 'b'}{'s' if args.strict else 'f' if args.future else ''}{'i' if args.intrahour else ''}"
    return W.ROOT / "models" / f"scaler_{tag}.joblib"


def run(args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = device()
    feats, fut = feature_sets(args)
    name = run_name(args)
    data, scaler = W.prepare(W.load_dataset(), feats, fut, save_scaler=scaler_path(args))
    tr, va, te = data["train"], data["val"], data["test"]
    loader = DataLoader(TensorDataset(torch.from_numpy(tr.X), torch.from_numpy(tr.F), torch.from_numpy(tr.Y)),
                        batch_size=args.batch, shuffle=True)
    model = build(args.model, len(feats), len(fut), **model_kwargs(args)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, best_state, bad, hist = float("inf"), None, 0, []
    Xv, Fv, Yv = (torch.from_numpy(a).to(dev) for a in (va.X, va.F, va.Y))
    print(f"{name}: device={dev}, params={sum(p.numel() for p in model.parameters()):,}, train={len(tr.X)}", flush=True)
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for xb, fb, yb in loader:
            xb, fb, yb = xb.to(dev), fb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = pinball_loss(model(xb, fb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            val = pinball_loss(model(Xv, Fv), Yv).item()
        hist.append((total / len(tr.X), val))
        print(f"ep {ep:02d} train {hist[-1][0]:.4f} val {val:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if val < best - 1e-4:
            best, bad = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop")
                break
    model.load_state_dict(best_state)
    cfg = {"model": args.model, "features": feats, "future": fut, "hp": model_kwargs(args), "lr": args.lr,
           "batch": args.batch, "seed": args.seed, "scaler": scaler_path(args).name}
    torch.save({"state": best_state, **cfg}, W.ROOT / "models" / f"{name}.pt")
    preds = {s: predict(model, data[s].X, data[s].F, dev) for s in ("val", "test")}
    np.savez(W.ROOT / "results" / f"pred_{name}.npz", val=preds["val"], test=preds["test"],
             val_start=data["val"].start.values, test_start=data["test"].start.values)
    cols = feats + [c for c in fut if c not in feats]
    res_val = metrics(to_units(va.Y, scaler, cols), to_units(preds["val"], scaler, cols))
    with open(W.ROOT / "results" / f"val_{name}.json", "w") as f:
        json.dump({"val": res_val, "best_val_loss": best, "epochs": len(hist), "history": hist, **cfg}, f, indent=1)
    print("VALIDATION\n" + table({name: res_val}))
    return res_val


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["lstm", "cnn", "patchtst"], required=True)
    p.add_argument("--weather", action="store_true", help="add DWD/Open-Meteo observations to the inputs")
    p.add_argument("--future", action="store_true", help="add known-future inputs of the target day")
    p.add_argument("--strict", action="store_true", help="known-future inputs = calendar + load forecast only (noon-safe)")
    p.add_argument("--intrahour", action="store_true", help="add the within-hour 15-min price range")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--hidden", type=int, default=128, help="LSTM units / CNN filters / PatchTST d_model (all runs here use 128)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="", help="suffix for tuning runs")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
