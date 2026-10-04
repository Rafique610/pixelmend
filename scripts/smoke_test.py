"""End-to-end smoke test for a running PixelMend API (local uvicorn or Docker).

Run: uv run python scripts/smoke_test.py [--base-url http://localhost]
Exits non-zero if any check fails. CPU-only, no GPU required.
"""

import argparse
import io
import sys

import httpx
from PIL import Image

RESTORE = ["universal", "hard-routed", "soft-moe"]
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, info: str = "") -> None:
    results.append((name, ok, info))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}")


def png(size=(128, 128)) -> bytes:
    img = Image.effect_noise(size, 40).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost")
    base = ap.parse_args().base_url.rstrip("/")
    c = httpx.Client(base_url=base, timeout=60)
    img = png()

    r = c.get("/health")
    h = r.json() if r.status_code == 200 else {}
    loaded = h.get("models_loaded", {})
    check("health ok, 7/7 models loaded", h.get("status") == "ok" and len(loaded) == 7 and all(loaded.values()), str(loaded))
    check("provider is CPU", h.get("execution_provider") == "CPUExecutionProvider")
    check("docs reachable", c.get("/docs").status_code == 200)

    for ep in RESTORE:
        r = c.post(f"/api/v1/restore/{ep}", files={"file": ("a.png", img, "image/png")},
                   data={"apply_corruption": "true", "corruption_type": "salt_and_pepper", "severity": "2"})
        j = r.json() if r.status_code == 200 else {}
        check(f"{ep} 200", r.status_code == 200, r.text[:120] if r.status_code != 200 else "")
        check(f"{ep} restored image", str(j.get("restored_image", "")).startswith("data:image/png;base64,"))
        if ep == "soft-moe":
            w = j.get("routing_weights") or j.get("weights") or {}
            vals = list(w.values()) if isinstance(w, dict) else list(w)
            check("soft-moe weights sum to 1", len(vals) == 4 and abs(sum(vals) - 1) < 1e-3, str(w))

    for style in (1, 2, 3):
        r = c.post("/api/v1/sketch/generate", files={"file": ("a.png", img, "image/png")}, data={"style": str(style)})
        check(f"sketch style {style}", r.status_code == 200 and r.json().get("selected_style") == style)

    for ep in [f"/api/v1/restore/{e}" for e in RESTORE] + ["/api/v1/sketch/generate"]:
        d = {"style": "1"}
        check(f"{ep} text->400", c.post(ep, files={"file": ("a.txt", b"nope", "text/plain")}, data=d).status_code == 400)
        check(f"{ep} empty->400", c.post(ep, files={"file": ("a.png", b"", "image/png")}, data=d).status_code == 400)
        big = b"\x89PNG" + b"0" * (10 * 1024 * 1024 + 1)
        check(f"{ep} >10MB->413", c.post(ep, files={"file": ("a.png", big, "image/png")}, data=d).status_code == 413)

    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
