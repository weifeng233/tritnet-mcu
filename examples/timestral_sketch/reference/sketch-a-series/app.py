"""Sketch-a-Series: draw a line, and Timestral continues it while you draw.

The page sends the stroke, resampled to a steady step, each time it grows. Requests that arrive together are
forecast in one batch, which keeps a small CPU responsive when many people draw at once. Requests wait for their
batch without holding a server thread, so the page itself still loads at once when the forecaster is busy.

    uvicorn app:app --port 7860
"""
from __future__ import annotations

import asyncio
import gzip
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
import orjson
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from timestral import TimestralForecaster

MODEL = os.environ.get("MODEL", "mistralai/Timestral-5M")
MAX_BATCH = 64
CHUNK = 16  # a batch runs 16 strokes at a time, sorted by length, so short strokes are not padded to the longest
TIMEOUT = 10.0
STATIC = Path(__file__).parent / "static"
log = logging.getLogger("uvicorn.error")


def cpu_limit() -> int:
    """CPUs this container may use. os.cpu_count() reports the host's, far more than a 2-vCPU Space gets, and more
    threads than the quota makes every forecast wait on throttling (2.2x slower for a batch of 16)."""
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, int(quota) // int(period))
    except (OSError, ValueError):
        pass
    return len(os.sched_getaffinity(0))


torch.set_num_threads(int(os.environ.get("THREADS", min(4, cpu_limit()))))
forecaster = TimestralForecaster.from_pretrained(MODEL)
LEVELS = list(forecaster.quantile_levels)
MODEL_NAME = MODEL.split("/")[-1]


def forecast_batch(ctx: np.ndarray, h: int) -> np.ndarray:
    """(B, T) contexts to (B, H, 21) quantiles. Up to 8 rows, the mirrored pass of symmetric averaging runs in the
    same call as the forward one: the same numbers, in about 60% of the time."""
    if len(ctx) > 8:
        return forecaster.forecast(ctx, prediction_length=h, clip_nonnegative=False).quantiles[:, 0].numpy()
    q = forecaster.forecast(np.concatenate([ctx, -ctx]), prediction_length=h, clip_nonnegative=False,
                            symmetric_averaging=False).quantiles[:, 0]
    return (0.5 * (q[: len(ctx)] + (-q[len(ctx):]).flip(-1))).numpy()


forecast_batch(np.full((1, 64), 0.5, dtype=np.float32), 256)  # the first call is slow; pay it before serving


class Job:
    __slots__ = ("y", "h", "loop", "future")

    def __init__(self, y: np.ndarray, h: int, loop: asyncio.AbstractEventLoop):
        self.y, self.h, self.loop, self.future = y, h, loop, loop.create_future()


def resolve(future: asyncio.Future, result) -> None:
    if not future.done():  # the request may have timed out meanwhile
        future.set_result(result)


class Batcher:
    """One worker thread. It takes every request waiting at that moment, left-pads them with missing values to a
    common length, and forecasts them together. Padding with missing values matches what the model does itself.
    At peak load, sorting by length and running 16 at a time costs 1.6 ms per stroke instead of 2.4."""

    def __init__(self):
        self.queue: list[Job] = []
        self.lock = threading.Lock()
        self.ready = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    async def submit(self, y: np.ndarray, h: int) -> tuple[np.ndarray | Exception, float]:
        job = Job(y, h, asyncio.get_running_loop())
        with self.lock:
            self.queue.append(job)
            self.ready.set()
        try:
            return await asyncio.wait_for(job.future, TIMEOUT)
        except asyncio.TimeoutError:
            raise HTTPException(503, "The forecaster is busy. Try again in a moment.")

    def _forecast(self, jobs: list[Job]) -> list[np.ndarray]:
        t, h = max(len(j.y) for j in jobs), max(j.h for j in jobs)
        ctx = np.full((len(jobs), t), np.nan, dtype=np.float32)
        for i, j in enumerate(jobs):
            ctx[i, t - len(j.y):] = j.y
        q = forecast_batch(ctx, h)
        return [q[i, : j.h] for i, j in enumerate(jobs)]

    def _run(self) -> None:
        while True:
            self.ready.wait()
            with self.lock:
                waiting = [j for j in self.queue if not j.future.done()]  # skip requests that already timed out
                jobs, self.queue = waiting[:MAX_BATCH], waiting[MAX_BATCH:]
                if not self.queue:
                    self.ready.clear()
            jobs.sort(key=lambda j: len(j.y))
            for i in range(0, len(jobs), CHUNK):
                chunk = jobs[i:i + CHUNK]
                started = time.perf_counter()
                try:
                    results = self._forecast(chunk)
                except Exception:  # one degenerate stroke must not fail the others: retry them one by one
                    results = []
                    for j in chunk:
                        try:
                            results += self._forecast([j])
                        except Exception as e:
                            log.warning("could not forecast a stroke of %d values, h=%d: %s", len(j.y), j.h, e)
                            results.append(e)
                ms = (time.perf_counter() - started) * 1000
                for j, r in zip(chunk, results):
                    j.loop.call_soon_threadsafe(resolve, j.future, (r, ms))


batcher = Batcher()
app = FastAPI(title="Sketch-a-Series")


class ForecastRequest(BaseModel):
    y: list[float] = Field(min_length=8, max_length=1024)
    h: int = Field(ge=1, le=1024)


@app.exception_handler(RequestValidationError)
async def rejected(request: Request, exc: RequestValidationError):
    # Say in the logs why a request was refused (not what it held), so failures seen by users can be traced.
    log.warning("refused %s %s: %s", request.method, request.url.path,
                "; ".join(f"{'.'.join(map(str, e['loc'][:2]))} {e['msg'][:80]}" for e in exc.errors()[:3]))
    return await request_validation_exception_handler(request, exc)


@app.post("/api/forecast")
async def forecast(req: ForecastRequest, request: Request):
    y = np.asarray(req.y, dtype=np.float32)
    if not np.isfinite(y).all():
        raise HTTPException(422, "The stroke contains invalid values.")
    q, ms = await batcher.submit(y, req.h)
    if isinstance(q, Exception):
        raise HTTPException(422, f"Could not forecast this stroke: {q}")
    # 21 quantile levels x horizon, rounded: plenty for drawing. Rounded as float64, so 0.4221 is sent as such
    # and not as 0.4221000075340271, and gzipped: about 16 KB instead of 117 KB for a full-width forecast.
    q = np.ascontiguousarray(np.round(np.nan_to_num(q.T.astype(np.float64), nan=0.0, posinf=1e3, neginf=-1e3), 4))
    body = orjson.dumps({"q": q, "levels": LEVELS, "ms": round(ms, 1), "model": MODEL_NAME},
                        option=orjson.OPT_SERIALIZE_NUMPY)
    if "gzip" in request.headers.get("accept-encoding", ""):
        return Response(gzip.compress(body, 1), media_type="application/json",
                        headers={"Content-Encoding": "gzip", "Vary": "Accept-Encoding"})
    return Response(body, media_type="application/json", headers={"Vary": "Accept-Encoding"})


class Static(StaticFiles):
    """The page's code is revalidated on every load (a 304 when unchanged), so a fix reaches everyone at once
    instead of after the browser's own guess at freshness. Fonts never change and are kept for a week."""

    def file_response(self, full_path, *args, **kwargs):
        response = super().file_response(full_path, *args, **kwargs)
        response.headers["Cache-Control"] = "public, max-age=604800" if "/fonts/" in str(full_path) else "no-cache"
        return response


@app.api_route("/", methods=["GET", "HEAD"])
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", Static(directory=STATIC), name="static")
