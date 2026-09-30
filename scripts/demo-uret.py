#!/usr/bin/env python3
"""Regenerate the README terminal demo from real command output.

    uv pip install .         # the demo runs the installed `nvidia-nim-mcp` and scripts/sonda.py
    python scripts/demo-uret.py [output-dir]        # default: docs/demo

Nothing on screen is typed by hand. Each command below is run in a scratch
folder; its exit code and everything it printed are recorded, and the terminal
page replays that record with a typing animation. `komutlar.txt` is the same
record as plain text (command, output, exit code, date).

Files written:
    komutlar.txt      the record
    demo.html         the replay; add ?dikey for the 1080x1920 layout, no caption
    demo.mp4/.gif     landscape recording   } only if node + playwright + ffmpeg
    demo-dikey.mp4    vertical recording    } are available (scripts/demo-kayit.js)

`--kurulum` also times a cold `uvx --from git+... nvidia-nim-mcp --version` with an
empty uv cache and writes it to kurulum.txt. It needs the network and GitHub.

The look is the FRK-OS terminal scene of the daily videos (sosyal/uret/tema.mjs,
theme "klasik"): black #0e0d0b, cream #f1ece2, yellow #ffc21a, JetBrains Mono
(SIL OFL 1.1, assets/yazi/).
"""
from __future__ import annotations

import base64
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SERVER = "https://github.com/Furkiozknn/nvidia-nim-mcp"

# The pipeline is `set -o pipefail`, so `| tail` reports the probe's own exit
# code rather than tail's. --no-keys: nothing is sent to any provider, so the
# recording is the same for a person who has no key and cannot spend quota.
COMMANDS = [
    "nvidia-nim-mcp --version",
    "python scripts/sonda.py --no-keys -- nvidia-nim-mcp",
    "python scripts/sonda.py --no-keys --call ask_llm '{\"question\": \"hi\"}' -- nvidia-nim-mcp | tail -n 3",
    "python scripts/sonda.py --no-keys --call nope '{}' -- nvidia-nim-mcp | tail -n 2",
]



def bash() -> str:
    found = shutil.which("bash")
    if not found:
        sys.exit("bash is needed to run the demo commands (Git Bash on Windows).")
    return found


def run_all(workdir: str) -> list[dict]:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", NO_COLOR="1")
    for key in ("NVIDIA_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY", "GEMINI_API_KEY", "CEREBRAS_API_KEY"):
        env.pop(key, None)
    # The commands run in an empty folder that holds only the probe, so the
    # screen shows `scripts/sonda.py` exactly as the README writes it.
    os.makedirs(os.path.join(workdir, "scripts"))
    shutil.copy(ROOT / "scripts" / "sonda.py", os.path.join(workdir, "scripts", "sonda.py"))
    record = []
    for command in COMMANDS:
        started = time.time()
        done = subprocess.run(
            [bash(), "-c", "set -o pipefail; " + command],
            cwd=workdir, env=env, capture_output=True, encoding="utf-8",
            errors="replace",
        )
        output = (done.stdout + done.stderr).replace("\r", "").rstrip()
        record.append({
            "k": command, "c": output, "kod": done.returncode,
            "sure": round(time.time() - started, 2),
        })
    return record


def checkout_sha() -> str:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, encoding="utf-8").stdout.strip()

    dirty = " (+ uncommitted changes)" if git("status", "--porcelain", "--untracked-files=no") else ""
    return git("log", "-1", "--format=%h %cs") + dirty


def write_record(out: Path, record: list[dict], sha: str, version: str) -> None:
    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        "# nvidia-nim-mcp terminal demo: real commands, real output",
        f"# tarih: {now}",
        f"# surum: {version} (uv pip install . ile kuruldu, PATH'teki nvidia-nim-mcp)",
        f"# kaynak: {SERVER} @ {sha}",
        "# saglayici anahtari yok (--no-keys) ve ortamdan da silindi: hicbir saglayiciya istek gitmedi",
        "# her komut bash -c 'set -o pipefail; ...' ile bos bir klasorde kosuldu; hicbir satir elle yazilmadi",
        "",
    ]
    for item in record:
        lines.append("$ " + item["k"])
        if item["c"]:
            lines.append(item["c"])
        lines.append(f"[cikis kodu {item['kod']}] ({item['sure']:.2f} s)")
        lines.append("")
    (out / "komutlar.txt").write_text("\n".join(lines), encoding="utf-8", newline="\n")


PAGE = r"""<!doctype html><html lang="en"><meta charset="utf-8"><title>nvidia-nim-mcp demo</title>
<style>
@font-face{font-family:JB;font-weight:400;src:url(data:font/ttf;base64,__FONT400__) format("truetype")}
@font-face{font-family:JB;font-weight:700;src:url(data:font/ttf;base64,__FONT700__) format("truetype")}
:root{--zemin:#0e0d0b;--panel:#14120e;--krem:#f1ece2;--sari:#ffc21a;--sonuk:#b6ae9d;--mercan:#ff4d6d;--turuncu:#ff7a1a;--cam:#19d3e6}
html,body{margin:0;height:100%;background:var(--zemin)}
body{display:flex;align-items:center;justify-content:center;background-image:linear-gradient(rgba(241,236,226,.045) 1px,transparent 1px),linear-gradient(90deg,rgba(241,236,226,.045) 1px,transparent 1px);background-size:48px 48px}
.p{display:flex;flex-direction:column;width:1120px;height:640px;box-sizing:border-box;background:var(--panel);border:1px solid #3a352b;border-left:6px solid var(--sari);border-radius:8px;padding:22px 28px;font:400 20px/1.5 JB,Consolas,monospace;color:var(--krem);overflow:hidden;position:relative}
.b{font:700 13px JB,monospace;letter-spacing:.12em;color:var(--sari);margin:0 0 14px;text-transform:uppercase}
.y{color:var(--sari);font-weight:700}.d{color:var(--sonuk)}.k{color:var(--mercan)}.t{color:var(--turuncu)}.c{color:var(--cam)}
.w{flex:1;overflow:hidden;min-height:0}
pre{margin:0;white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}
.im{display:inline-block;width:11px;height:22px;background:var(--sari);vertical-align:-4px;margin-left:2px}
body.v .p{width:1040px;height:1760px;font-size:24px;padding:36px 36px}
body.v .b{display:none}
body.v .im{width:13px;height:28px;vertical-align:-6px}
@media (prefers-reduced-motion:reduce){.im{display:none}}
</style>
<div class="p"><div class="b">nvidia-nim-mcp &middot; real commands, real output</div><div class="w"><pre id="t" aria-live="off"></pre></div></div>
<script>
const DIKEY=location.search.includes("dikey");
if(DIKEY)document.body.classList.add("v");
const K=__DATA__;
const HIZ=30; // ms per typed character
let olay=[],t=700;
for(const x of K){
  olay.push({t,tip:"komut",k:x.k}); t+=x.k.length*HIZ+300;
  const s=x.c?x.c.split("\n"):[];
  const adim=s.length>8?70:220;
  for(const l of s){olay.push({t,tip:"satir",l});t+=adim}
  t+=s.length?900:500;
}
const el=document.getElementById("t");
const esc=s=>s.replace(/&/g,"&amp;").replace(/</g,"&lt;");
// Colour is presentation only: it never changes a character.
function boya(s){
  s=esc(s);
  s=s.replace(/^(initialize ok:)/,'<span class="c">$1</span>').replace(/^(tools\/list:)/,'<span class="c">$1</span>');
  s=s.replace(/^(sonda:)/,'<span class="k">$1</span>').replace(/(no provider configured\.)/,'<span class="t">$1</span>');
  return s;
}
const t0=performance.now();
function ciz(){
  const now=performance.now()-t0;let g="";
  for(const o of olay){
    if(o.t>now)break;
    if(o.tip==="komut"){const n=Math.min(o.k.length,Math.floor((now-o.t)/HIZ));g+='<span class="y">$ </span>'+esc(o.k.slice(0,n))+(n<o.k.length?'<span class="im"></span>':"")+"\n"}
    else g+='<span class="d">'+boya(o.l)+"</span>\n";
  }
  if(now>olay[olay.length-1].t+300)g+='<span class="y">$ </span><span class="im"></span>';
  el.innerHTML=g;
  el.style.marginTop=Math.min(0,el.parentNode.clientHeight-el.getBoundingClientRect().height)+"px";
  requestAnimationFrame(ciz);
}
window.__bitis=olay[olay.length-1].t+1500;
ciz();
</script></html>
"""


def write_pages(out: Path, record: list[dict]) -> None:
    data = json.dumps([{"k": r["k"], "c": r["c"]} for r in record], ensure_ascii=False)
    fonts = ROOT / "assets" / "yazi"

    def embed(name: str) -> str:
        return base64.b64encode((fonts / name).read_bytes()).decode()

    html = (PAGE.replace("__DATA__", data)
            .replace("__FONT400__", embed("JetBrainsMono-Regular.ttf"))
            .replace("__FONT700__", embed("JetBrainsMono-Bold.ttf")))
    # One page: landscape by default, 1080x1920 with `?dikey`. The fonts are
    # embedded (assets/yazi, SIL OFL 1.1), so it plays from file:// with
    # nothing else next to it.
    (out / "demo.html").write_text(html, encoding="utf-8", newline="\n")


def record_video(out: Path) -> None:
    recorder = Path(__file__).with_name("demo-kayit.js")
    if not (shutil.which("node") and shutil.which("ffmpeg") and recorder.exists()):
        print("node/ffmpeg not found: wrote komutlar.txt and the html pages only.")
        return
    subprocess.run(["node", str(recorder), str(out)], check=True)


def measure_install(out: Path) -> None:
    """Cold `uvx` install, with an empty uv cache, timed."""
    uvx = shutil.which("uvx")
    if not uvx:
        print("uvx not found: skipping the install timing.")
        return
    cache = tempfile.mkdtemp(prefix="nvidia-nim-mcp-uv-")
    env = dict(os.environ, UV_CACHE_DIR=cache, PYTHONIOENCODING="utf-8")
    command = [uvx, "--from", "git+https://github.com/Furkiozknn/nvidia-nim-mcp", "nvidia-nim-mcp", "--version"]
    rows = []
    for label in ("cold (empty uv cache)", "warm (same cache)"):
        started = time.time()
        done = subprocess.run(command, env=env, capture_output=True, encoding="utf-8", errors="replace")
        rows.append(f"{label}: {time.time() - started:.2f} s, exit {done.returncode}, "
                    f"output {done.stdout.strip()!r}")
    shutil.rmtree(cache, ignore_errors=True)
    stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    (out / "kurulum.txt").write_text(
        "# uvx --from git+https://github.com/Furkiozknn/nvidia-nim-mcp nvidia-nim-mcp --version\n"
        f"# tarih: {stamp}\n" + "\n".join(rows) + "\n",
        encoding="utf-8", newline="\n",
    )
    print("\n".join(rows))


def main(argv: list[str]) -> int:
    flags = [a for a in argv if a.startswith("--")]
    args = [a for a in argv if not a.startswith("--")]
    out = Path(args[0]).resolve() if args else ROOT / "docs" / "demo"
    out.mkdir(parents=True, exist_ok=True)
    if not shutil.which("nvidia-nim-mcp"):
        sys.exit("nvidia-nim-mcp is not on PATH: run `uv pip install .` from the repository root first.")
    version = subprocess.run(["nvidia-nim-mcp", "--version"], capture_output=True, encoding="utf-8").stdout.strip()
    with tempfile.TemporaryDirectory(prefix="nvidia-nim-mcp-demo-") as work:
        record = run_all(work)
        sha = checkout_sha()
    write_record(out, record, sha, version.replace("nvidia-nim-mcp ", ""))
    write_pages(out, record)
    if "--kurulum" in flags:
        measure_install(out)
    if "--sadece-kayit" not in flags:
        record_video(out)
    print(f"demo: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
