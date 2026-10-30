"""Sketch-a-Series, deployed locally: draw a stroke, Timestral continues it.

This is the app half of the request: the same interaction as
https://huggingface.co/spaces/townbox/sketch-a-series (draw with the mouse, the forecast
follows the ink as a quantile fan), served from this machine and pointed at two backends:

    fp32      Mistral's released Timestral-5M, PyTorch, 20.8 MB of weights
    ternary   the same checkpoint quantized by this repository's Zig stack
              (absmean ternarization + 5-trits-per-byte packing + QAT), 1.15 MB

Both answer the identical `/api/forecast` request, so the page can draw the two forecasts
on top of each other for the same stroke. The ternary backend runs in the `timestral-cli`
worker (one process, artifact loaded once, JSON lines over stdin/stdout); nothing in that
path is PyTorch.

    python deploy/app.py                     # http://127.0.0.1:8000
    python deploy/app.py --backends ternary  # quantized model only

Environment overrides: TIMESTRAL_WORKER (timestral-cli path), TIMESTRAL_ARTIFACT,
TIMESTRAL_FP32 (checkpoint directory).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

EX = Path(__file__).resolve().parents[1]        # examples/timestral_sketch
ROOT = EX.parents[1]                           # repository root
sys.path.insert(0, str(EX / "reference"))

WEB = EX / "web" / "sketch-a-series.html"
# The worker is built by `zig build`, which appends .exe on Windows only. Hardcoding
# the suffix made `make sketch` fail on Linux and macOS with "worker not found".
WORKER_NAME = "timestral-cli.exe" if os.name == "nt" else "timestral-cli"
WORKER = Path(os.environ.get("TIMESTRAL_WORKER", ROOT / "zig-out" / "bin" / WORKER_NAME))
ARTIFACT = Path(os.environ.get("TIMESTRAL_ARTIFACT", EX / "modeldata" / "timestral_packed.bin"))
FP32_DIR = Path(os.environ.get("TIMESTRAL_FP32", EX / "modeldata" / "timestral-5m"))
EVAL_FP32 = EX / "modeldata" / "timestral_eval_fp32.json"
EVAL_TERNARY = EX / "modeldata" / "timestral_eval_ternary.json"

H_CHOICES = (96, 192)
LEVELS = [round(0.01 + 0.0, 2)]  # replaced at startup from the checkpoint config
DEFAULT_LEVELS = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
LEVELS = DEFAULT_LEVELS

app = FastAPI(title="Sketch-a-Series (Timestral-5M, ternary)")

# ---------------------------------------------------------------------------
# fp32 backend: PyTorch
# ---------------------------------------------------------------------------


class Fp32Backend:
    """Mistral's released checkpoint through the reference `timestral` package."""

    def __init__(self) -> None:
        self.forecaster = None
        self.lock = threading.Lock()
        self.params = 0
        self.levels = DEFAULT_LEVELS

    def load(self) -> None:
        import torch

        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        from timestral import TimestralForecaster

        self.forecaster = TimestralForecaster.from_pretrained(FP32_DIR)
        self.params = sum(p.numel() for p in self.forecaster.model.parameters())
        self.levels = list(self.forecaster.quantile_levels)

    def forecast(self, y: np.ndarray, h: int) -> tuple[list, float]:
        assert self.forecaster is not None
        t0 = time.perf_counter()
        # symmetric averaging is what the Space does by hand for batching; the forecaster
        # implements the same two passes + reflection internally.
        q = self.forecaster.forecast(y[None, :], prediction_length=h, clip_nonnegative=False).quantiles[0, 0]
        ms = (time.perf_counter() - t0) * 1000.0
        return np.asarray(q, dtype=np.float32).tolist(), ms

    def info(self) -> dict:
        size = (FP32_DIR / "model.safetensors").stat().st_size if (FP32_DIR / "model.safetensors").exists() else 0
        return {"id": "fp32", "label": "Timestral-5M fp32", "params": self.params, "bytes": size}


# ---------------------------------------------------------------------------
# ternary backend: the Zig worker
# ---------------------------------------------------------------------------


class ZigBackend:
    """`timestral-cli serve`: the packed ternary artifact, decoded and run in Zig."""

    def __init__(self, path: Path = WORKER, artifact: Path = ARTIFACT) -> None:
        self.worker_path = path
        self.artifact = artifact
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.params = 0
        self.ternary_params = 0

    def start(self) -> None:
        if not self.worker_path.exists():
            raise FileNotFoundError(f"{self.worker_path} not built (zig build --release=fast)")
        if not self.artifact.exists():
            raise FileNotFoundError(f"{self.artifact} not built (timestral-cli quantize)")
        self.proc = subprocess.Popen(
            [str(self.worker_path), "serve", str(self.artifact)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
            cwd=str(ROOT),
        )
        hello = self.proc.stdout.readline()
        print(f"worker: {hello.strip()}", flush=True)
        # "timestral worker ready: <path> (<params> params, <ternary> ternary)"
        m = re.search(r"\((\d+) params, (\d+) ternary\)", hello)
        if m:
            self.params, self.ternary_params = int(m.group(1)), int(m.group(2))

    def _restart(self) -> None:
        self.stop()
        self.start()

    def forecast(self, y: np.ndarray, h: int) -> tuple[list, float]:
        payload = json.dumps({"y": [float(v) for v in y], "h": int(h)}, separators=(",", ":"))
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self._restart()
            assert self.proc is not None and self.proc.stdin is not None and self.proc.stdout is not None
            try:
                self.proc.stdin.write(payload + "\n")
                self.proc.stdin.flush()
                line = self.proc.stdout.readline()
            except (BrokenPipeError, OSError) as exc:
                self._restart()
                raise HTTPException(502, f"worker failed: {exc}") from exc
        if not line:
            self._restart()
            raise HTTPException(502, "worker closed the connection")
        doc = json.loads(line)
        if "error" in doc:
            raise HTTPException(422, doc["error"])
        return doc["q"], float(doc.get("ms", 0.0))

    def stop(self) -> None:
        if self.proc is not None:
            try:
                self.proc.stdin.close()  # type: ignore[union-attr]
            except Exception:
                pass
            try:
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
            self.proc = None

    def info(self) -> dict:
        size = self.artifact.stat().st_size if self.artifact.exists() else 0
        bits = (size * 8 / self.params) if self.params else 0.0
        return {
            "id": "ternary", "label": "Timestral-5M ternary (QAT)",
            "params": self.params, "ternary_params": self.ternary_params,
            "bytes": size, "bits_per_weight": round(bits, 2),
        }


BACKENDS = {"fp32": Fp32Backend(), "ternary": ZigBackend()}
ENABLED = ["fp32", "ternary"]


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class ForecastRequest(BaseModel):
    y: list[float] = Field(min_length=8, max_length=4096)
    h: int = 96
    backend: str = "both"


def _clean(q: list) -> list:
    out = []
    for row in q:
        vals = []
        for v in row:
            f = float(v)
            if not math.isfinite(f):
                f = 0.0
            vals.append(round(f, 4))
        out.append(vals)
    return out


@app.get("/")
def index() -> FileResponse:
    if not WEB.exists():
        raise HTTPException(500, f"missing page: {WEB}")
    return FileResponse(WEB, headers={"Cache-Control": "no-cache"})


@app.get("/api/info")
def info() -> JSONResponse:
    doc = {
        "model": "Timestral-5M",
        "levels": LEVELS,
        "h_values": list(H_CHOICES),
        "backends": [BACKENDS[b].info() for b in ENABLED],
    }
    evals = {}
    for name, path in (("fp32", EVAL_FP32), ("ternary", EVAL_TERNARY)):
        if path.exists():
            try:
                evals[name] = json.loads(path.read_text())
            except json.JSONDecodeError:
                pass
    if evals:
        doc["eval"] = evals
    return JSONResponse(doc)


@app.post("/api/forecast")
def forecast(req: ForecastRequest) -> JSONResponse:
    y = np.asarray(req.y, dtype=np.float32)
    if not np.isfinite(y).all():
        raise HTTPException(422, "the stroke contains NaN or infinite values")
    if req.h not in H_CHOICES:
        raise HTTPException(422, f"h must be one of {list(H_CHOICES)}")
    wanted = [b for b in ENABLED if req.backend in ("both", "all", b)]
    if not wanted:
        raise HTTPException(422, f"backend must be 'both' or one of {ENABLED}")
    results, errors = {}, {}
    for name in wanted:
        try:
            q, ms = BACKENDS[name].forecast(y, req.h)
            results[name] = {"q": _clean(q), "ms": round(ms, 2)}
        except HTTPException as exc:
            errors[name] = str(exc.detail)
        except Exception as exc:  # a backend must not take the page down
            errors[name] = f"{type(exc).__name__}: {exc}"
    if not results:
        raise HTTPException(502, "; ".join(f"{k}: {v}" for k, v in errors.items()) or "no backend answered")
    body = {"h": req.h, "levels": LEVELS, "results": results}
    if errors:
        body["errors"] = errors
    return JSONResponse(body)


def main() -> None:
    global ENABLED
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--backends", default="fp32,ternary", help="comma separated: fp32,ternary")
    # Smoke-testable without a model or a GPU: prints what the app would load and
    # what is missing, then exits. CI uses it to check the platform-specific worker
    # path on Linux/macOS, which is exactly the thing that was wrong for months.
    ap.add_argument("--check", action="store_true", help="print the resolved paths and exit")
    args = ap.parse_args()

    if args.check:
        # Can the app start at all? The page and the worker are required; the model
        # files are only needed by the backend you enable, and their absence is
        # reported by the backend itself with a clearer message.
        missing = 0
        for label, path, required in (
            ("worker", WORKER, True),
            ("page", WEB, True),
            ("artifact", ARTIFACT, False),
            ("fp32 checkpoint", FP32_DIR / "model.safetensors", False),
        ):
            ok = path.exists()
            if required and not ok:
                missing += 1
            state = "ok " if ok else ("MISSING" if required else "missing (needed only by its backend)")
            print(f"{label:16} {state:45} {path}", flush=True)
        print(f"backends         available: {', '.join(sorted(BACKENDS))}; requested: {args.backends}", flush=True)
        raise SystemExit(1 if missing else 0)

    ENABLED = [b.strip() for b in args.backends.split(",") if b.strip() in BACKENDS]
    if not ENABLED:
        raise SystemExit(f"no known backend in {args.backends!r}")

    if "fp32" in ENABLED:
        print("loading Timestral-5M (fp32, torch)...", flush=True)
        BACKENDS["fp32"].load()
        print(f"  params {BACKENDS['fp32'].params:,}", flush=True)
    if "ternary" in ENABLED:
        print("starting the ternary worker...", flush=True)
        BACKENDS["ternary"].start()
        print(f"  artifact {ARTIFACT.stat().st_size:,} B", flush=True)

    import uvicorn

    print(f"open http://{args.host}:{args.port}/", flush=True)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        BACKENDS["ternary"].stop()


if __name__ == "__main__":
    main()
