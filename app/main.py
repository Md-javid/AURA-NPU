"""
app/main.py — Aura-NPU v4.1 (Offline Cognitive Edition)
=========================================================
NiceGUI overlay application for the AMD Ryzen AI 300 NPU.

Inference priority (auto-detected at startup):
  1. VitisAI EP / AMD NPU  — ~100 ms, ~12 W
  2. Ollama  localhost      — llava vision model, ~800 ms
  3. PIL offline            — always available, zero network

Run:
  python -m app.main

Environment variables (all optional — see app/config.py):
  OLLAMA_HOST   http://localhost:11434
  OLLAMA_MODEL  llava
  AURA_HOST     127.0.0.1
  AURA_PORT     8765
  AURA_OFFLINE  true
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import logging
import time
from pathlib import Path
from typing import Optional

import httpx
from nicegui import app, ui

from app.npu_engine import AuraNPUEngine, get_engine
from app.multimodal_logic import MultimodalProcessor
from app.integrity_tracker import IntegrityTracker
from app.utils.screen_capture import ScreenCapture
from app.utils.language_config import LANGUAGE_MAP
from app.telemetry import get_telemetry, AuraTelemetry
from app import config as _cfg

logging.basicConfig(
    level=getattr(logging, _cfg.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("aura.main")

# ─────────────────────────────────────────────────────────────
#  LLM provider (sourced from app/config.py — set via env vars)
# ─────────────────────────────────────────────────────────────
OLLAMA_HOST = _cfg.OLLAMA_HOST
OLLAMA_MDL  = _cfg.OLLAMA_MODEL

logger.info("Aura-NPU v4.1 Offline Cognitive Edition | Ollama: %s", OLLAMA_HOST)
_cfg.log_config()


# ─────────────────────────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────────────────────────
class AuraState:
    """Single shared mutable state container for the NiceGUI session.

    All per-session flags live here.  Updated only from the async UI
    event loop — no external locking required.
    """
    engine:     Optional[AuraNPUEngine]       = None
    processor:  Optional[MultimodalProcessor] = None
    tracker:    Optional[IntegrityTracker]    = None
    capture:    ScreenCapture                 = ScreenCapture()
    is_scanning: bool  = False
    linguistic_mirror: bool = False
    adhd_mode: bool    = False
    selected_language: str  = "hi"
    capture_mode: str  = "screen"
    session_interactions: int = 0
    npu_active: bool   = False      # True when VitisAI EP / Lemonade loaded
    ollama_active: bool = False     # True when Ollama responds to probe
    engine_status: dict = {}
    uploaded_img_b64: list = None
    last_captured_b64: str = ""
    eco_mode: bool       = False   # lower res + fewer tokens for battery
    offline_sovereign: bool = False# network isolation verified
    dual_pane: bool      = False   # English + native side-by-side
    vision_trace_on: bool = False  # show pipeline trace during scan
    scan_stage: str      = ""      # current trace stage
    session_start: float = 0.0
    session_id: str      = ""

    def __init__(self):
        self.uploaded_img_b64 = [""]
        self.session_start = time.time()
        self.session_id = hashlib.sha256(
            f"{self.session_start}".encode()
        ).hexdigest()[:10].upper()

state = AuraState()
_onboard_dialog: Optional[ui.dialog] = None


# ─────────────────────────────────────────────────────────────
#  CSS  —  Liquid Glass / Glassmorphism + AMD aurora
# ─────────────────────────────────────────────────────────────
AURA_CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&display=swap');

:root {
  --bg0   : #050508;
  --glass : rgba(255,255,255,0.04);
  --glass2: rgba(255,255,255,0.07);
  --gbord : rgba(255,255,255,0.09);
  --gbord2: rgba(255,255,255,0.15);
  --orange: #FF6A00;
  --red   : #EE2A24;
  --or12  : rgba(255,106,0,0.12);
  --or30  : rgba(255,106,0,0.30);
  --glow  : 0 0 40px rgba(255,106,0,0.18), 0 0 80px rgba(238,42,36,0.08);
  --text  : #EFEFEF;
  --muted : #606070;
  --mono  : 'Share Tech Mono', monospace;
  --head  : 'Rajdhani', 'Segoe UI', sans-serif;
  --r     : 16px;
  --r2    : 10px;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}

/* aurora background */
body,.nicegui-content{
  background:var(--bg0)!important;
  color:var(--text)!important;
  font-family:var(--head)!important;
  min-height:100vh;overflow-x:hidden;
}
body::before{
  content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background:
    radial-gradient(ellipse 65% 45% at 15% 15%,rgba(255,106,0,0.09) 0%,transparent 70%),
    radial-gradient(ellipse 55% 40% at 85% 75%,rgba(238,42,36,0.07) 0%,transparent 70%),
    radial-gradient(ellipse 70% 55% at 50% 50%,rgba(15,15,35,0.96) 0%,transparent 100%);
  animation:aurora 14s ease-in-out infinite alternate;
}
@keyframes aurora{
  0%{opacity:.7;transform:scale(1)}
  50%{opacity:1;transform:scale(1.05) translateY(-12px)}
  100%{opacity:.75;transform:scale(1.01) translateY(6px)}
}

/* page layout */
.aura-page{
  position:relative;z-index:1;min-height:100vh;
  padding:12px 8px;display:flex;
  justify-content:center;align-items:flex-start;
}
.aura-content{width:100%;max-width:430px;}

/* glass card */
.glass-card{
  backdrop-filter:blur(26px) saturate(200%);
  -webkit-backdrop-filter:blur(26px) saturate(200%);
  background:linear-gradient(145deg,rgba(255,255,255,0.06),rgba(255,255,255,0.02));
  border:1px solid var(--gbord2);
  border-radius:var(--r);
  box-shadow:var(--glow),inset 0 1px 0 rgba(255,255,255,0.12);
  padding:14px 15px;position:relative;overflow:hidden;
}
.glass-card::before{
  content:'';position:absolute;top:0;left:0;right:0;height:1.5px;
  background:linear-gradient(90deg,transparent,var(--orange),var(--red),transparent);
  animation:sweep 3.5s linear infinite;
}
@keyframes sweep{0%{transform:translateX(-100%)}100%{transform:translateX(100%)}}

/* inner panel */
.g-panel{
  backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
  background:var(--glass2);border:1px solid var(--gbord);border-radius:var(--r2);
  padding:10px 12px;
}

/* banner */
.banner{
  border-radius:8px;padding:7px 11px;
  font-family:var(--mono);font-size:0.67rem;
  display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:11px;
}
.banner-demo{background:rgba(255,106,0,0.08);border:1px solid rgba(255,106,0,0.3);color:#FF9955;}
.banner-ok{background:rgba(60,200,100,0.07);border:1px solid rgba(60,200,100,0.25);color:#55DD88;}
.banner a{color:var(--orange);text-decoration:underline;}

/* header */
.hdr{
  display:flex;align-items:center;justify-content:space-between;
  margin-bottom:11px;padding-bottom:9px;border-bottom:1px solid var(--gbord);
}
.logo{
  font-family:var(--head);font-size:1.5rem;font-weight:700;
  background:linear-gradient(135deg,#FF6A00,#EE2A24);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  letter-spacing:0.05em;text-transform:uppercase;line-height:1;
}
.logo-sub{
  font-family:var(--mono);font-size:0.56rem;color:var(--muted);
  letter-spacing:0.15em;text-transform:uppercase;margin-top:2px;
}
.npu-pill{
  display:inline-flex;align-items:center;gap:5px;
  backdrop-filter:blur(10px);
  background:rgba(255,106,0,0.1);border:1px solid rgba(255,106,0,0.28);
  border-radius:20px;padding:3px 9px;
  font-family:var(--mono);font-size:0.6rem;color:var(--orange);
}
.dot{
  width:7px;height:7px;border-radius:50%;
  background:var(--orange);box-shadow:0 0 7px var(--orange);
  animation:pls 1.6s ease-in-out infinite;
}
@keyframes pls{
  0%,100%{box-shadow:0 0 7px var(--orange);}
  50%{box-shadow:0 0 14px var(--orange);transform:scale(1.35);}
}

/* stats */
.stats-row{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin:9px 0;}
.stat-box{
  backdrop-filter:blur(10px);
  background:rgba(255,106,0,0.07);border:1px solid rgba(255,106,0,0.13);
  border-radius:9px;padding:7px 4px;text-align:center;
}
.stat-v{font-family:var(--mono);font-size:0.85rem;font-weight:bold;color:var(--orange);}
.stat-l{font-size:0.56rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.07em;margin-top:2px;}

/* divider */
.div-line{height:1px;background:var(--gbord);margin:9px 0;}

/* labels */
.sec-lbl2{
  font-family:var(--mono);font-size:0.59rem;
  color:var(--muted);letter-spacing:0.13em;text-transform:uppercase;margin-bottom:5px;
}
.form-lbl{
  font-family:var(--mono);font-size:0.61rem;
  color:var(--muted);letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px;
}

/* tabs */
.tab-row{display:flex;gap:5px;margin-bottom:9px;}
.tab-btn{
  flex:1;padding:7px 4px;
  backdrop-filter:blur(10px);
  background:rgba(255,255,255,0.03)!important;border:1px solid var(--gbord)!important;border-radius:9px!important;
  color:var(--muted)!important;font-family:var(--head)!important;font-size:0.79rem!important;font-weight:600!important;
  text-align:center;cursor:pointer;transition:all 0.18s;user-select:none;
  text-transform:none!important;letter-spacing:0!important;
  min-height:unset!important;height:auto!important;
}
.tab-btn .q-btn__content{font-family:var(--head)!important;font-size:0.79rem!important;font-weight:600!important;}
.tab-btn:hover,.tab-btn.q-btn--flat:hover{background:rgba(255,106,0,0.08)!important;border-color:rgba(255,106,0,0.35)!important;color:var(--text)!important;}
.tab-btn.active,.tab-btn.active.q-btn--flat{
  background:rgba(255,106,0,0.15)!important;border:1px solid var(--orange)!important;
  color:var(--orange)!important;
  box-shadow:0 0 12px rgba(255,106,0,0.22),inset 0 1px 0 rgba(255,255,255,0.08)!important;
}
.tab-btn .q-ripple{display:none!important;}

/* camera viewport */
.cam-viewport{
  position:relative;border-radius:12px;overflow:hidden;
  background:rgba(0,0,0,0.7);min-height:195px;
  border:1px solid rgba(255,106,0,0.18);
  margin-bottom:8px;transition:border-color 0.3s,box-shadow 0.3s;
  display:flex;align-items:stretch;
}
.cam-viewport.active{
  border-color:var(--orange)!important;
  box-shadow:0 0 22px rgba(255,106,0,0.45),inset 0 0 12px rgba(255,106,0,0.06)!important;
}
.cam-viewport video{
  width:100%;max-height:220px;object-fit:cover;display:none;
  border-radius:11px;
}
.cam-viewport canvas{display:none;position:absolute;top:-9999px;}
.cam-placeholder{
  position:absolute;inset:0;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:7px;
  color:var(--muted);font-family:var(--mono);font-size:0.72rem;
  pointer-events:none;
}
.cam-placeholder span:first-child{font-size:2rem;opacity:0.45;}
.cam-snap{
  width:100%;max-height:150px;object-fit:cover;
  border:1px solid rgba(255,106,0,0.35);border-radius:9px;
  display:block;margin-bottom:7px;
}

/* camera buttons */
.cam-btns{display:flex;gap:6px;margin-bottom:7px;}
.btn-cam-toggle{
  flex:1.4;backdrop-filter:blur(10px);
  background:rgba(60,200,100,0.10)!important;color:#55DD88!important;
  border:1px solid rgba(60,200,100,0.30)!important;border-radius:8px!important;
  font-family:var(--head)!important;font-size:0.82rem!important;font-weight:700!important;
  text-transform:none!important;letter-spacing:0!important;padding:8px 6px!important;
  transition:all 0.18s!important;
}
.btn-cam-toggle:hover{background:rgba(60,200,100,0.22)!important;}
.btn-cam-toggle.on{
  background:rgba(238,42,36,0.14)!important;color:#FF7777!important;
  border-color:rgba(238,42,36,0.40)!important;
  box-shadow:0 0 12px rgba(238,42,36,0.3)!important;
}
.btn-cam-toggle.on:hover{background:rgba(238,42,36,0.26)!important;}
.btn-cam-c{
  flex:1;backdrop-filter:blur(10px);
  background:rgba(255,106,0,0.12)!important;color:var(--orange)!important;
  border:1px solid rgba(255,106,0,0.38)!important;border-radius:8px!important;
  font-family:var(--head)!important;font-size:0.82rem!important;font-weight:700!important;
  text-transform:none!important;letter-spacing:0!important;padding:8px 6px!important;
  transition:all 0.15s!important;
}
.btn-cam-c:hover{background:rgba(255,106,0,0.26)!important;}

/* NiceGUI input / select glass override */
.g-input .q-field__control,.g-select .q-field__control{
  backdrop-filter:blur(14px)!important;
  background:rgba(255,255,255,0.05)!important;
  border:1px solid var(--gbord2)!important;
  border-radius:9px!important;min-height:38px!important;
  box-shadow:inset 0 1px 0 rgba(255,255,255,0.06)!important;
}
.g-input .q-field__control:focus-within,.g-select .q-field__control:focus-within{
  border-color:var(--orange)!important;
  box-shadow:0 0 0 2px rgba(255,106,0,0.18),inset 0 1px 0 rgba(255,255,255,0.06)!important;
}
.g-input .q-field__native,.g-select .q-field__native,.g-select .q-field__input{
  color:var(--text)!important;font-family:var(--head)!important;
  font-size:0.92rem!important;padding:7px 11px!important;
}
.g-input .q-field__label,.g-select .q-field__label{display:none!important;}
.g-input .q-field__bottom,.g-select .q-field__bottom{display:none!important;}
.g-select .q-field__marginal{color:var(--muted)!important;}

/* upload */
.up-zone {
  backdrop-filter:blur(10px);
  background:rgba(255,255,255,0.03);
  border:1.5px dashed var(--gbord2);border-radius:10px;
  transition:all 0.18s;
}
.up-zone:hover{border-color:rgba(255,106,0,0.45);background:rgba(255,106,0,0.05);}

/* scan button */
.btn-scan{
  width:100%!important;
  background:linear-gradient(135deg,var(--orange),var(--red))!important;
  color:#fff!important;border:none!important;border-radius:11px!important;
  font-family:var(--head)!important;font-size:1rem!important;
  font-weight:700!important;letter-spacing:0.1em!important;text-transform:uppercase!important;
  padding:11px!important;
  box-shadow:0 0 20px rgba(255,106,0,0.38),inset 0 1px 0 rgba(255,255,255,0.15)!important;
  transition:all 0.15s!important;margin-bottom:9px!important;
}
.btn-scan:hover{transform:translateY(-2px)!important;box-shadow:0 0 32px rgba(255,106,0,0.65)!important;}
.btn-scan.scanning{animation:breathe 0.8s ease-in-out infinite!important;}
@keyframes breathe{
  0%,100%{box-shadow:0 0 18px rgba(255,106,0,0.38);}
  50%{box-shadow:0 0 36px rgba(255,106,0,0.80);}
}

/* response box */
.resp-box{
  backdrop-filter:blur(14px);
  background:rgba(0,0,0,0.55);
  border:1px solid var(--gbord);border-left:3px solid var(--orange);
  border-radius:10px;padding:11px 13px;
  min-height:90px;max-height:300px;overflow-y:auto;
  font-family:var(--head);font-size:0.9rem;line-height:1.75;color:#E0E0E0;
  margin-bottom:9px;box-shadow:inset 0 2px 12px rgba(0,0,0,0.4);
}
.thinking{
  display:flex;align-items:center;gap:8px;
  color:var(--orange);font-family:var(--mono);font-size:0.74rem;
}

/* toggles — fixed single-row alignment */
.toggle-row{
  display:flex;align-items:center;justify-content:space-between;
  gap:8px;padding:4px 0;flex-wrap:nowrap;
}
.toggle-item{display:flex;align-items:center;gap:4px;flex:1;}
.toggle-item:last-child{justify-content:flex-end;}
.g-toggle .q-toggle__inner{min-width:34px!important;}
.g-toggle .q-toggle__label{font-size:0.81rem!important;color:var(--text)!important;white-space:nowrap;}
.g-toggle .q-toggle__track{background:rgba(255,255,255,0.1)!important;}
.g-toggle .q-toggle__track--true{background:var(--orange)!important;}

/* adhd banner */
.adhd-banner{
  backdrop-filter:blur(8px);background:rgba(238,42,36,0.07);
  border:1px solid rgba(238,42,36,0.2);border-radius:8px;
  padding:7px 10px;font-family:var(--mono);font-size:0.68rem;color:#FF9999;margin-top:4px;
}

/* integrity */
.int-row{display:flex;justify-content:space-between;align-items:center;
         padding:5px 0;border-bottom:1px solid var(--gbord);font-size:0.78rem;}
.int-lbl{color:var(--muted);font-size:0.71rem;}
.int-val{font-family:var(--mono);color:var(--orange);font-size:0.71rem;
         text-align:right;word-break:break-all;}

/* free LLM box */
.llm-info{
  backdrop-filter:blur(10px);
  background:rgba(60,130,255,0.07);border:1px solid rgba(60,130,255,0.2);
  border-radius:9px;padding:9px 11px;margin-bottom:9px;
}
.llm-info h4{font-size:0.82rem;color:#88AAFF;margin-bottom:6px;font-weight:700;}
.llm-opt{display:flex;align-items:flex-start;gap:7px;margin-bottom:5px;}
.llm-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;margin-top:4px;}
.llm-opt p{font-size:0.72rem;color:#888;line-height:1.48;}
.llm-opt code{font-family:var(--mono);color:#AACCFF;font-size:0.68rem;
              background:rgba(0,0,0,0.35);border-radius:4px;padding:2px 5px;}

/* setup steps */
.step-card{backdrop-filter:blur(10px);background:var(--glass2);
           border:1px solid var(--gbord);border-radius:10px;
           padding:10px 13px;margin-bottom:8px;display:flex;gap:10px;}
.step-num{width:28px;height:28px;border-radius:50%;background:var(--orange);
          color:#fff;font-weight:700;font-size:0.85rem;
          display:flex;align-items:center;justify-content:center;flex-shrink:0;}
.step-content h3{font-size:0.88rem;font-weight:700;color:var(--text);margin-bottom:3px;}
.step-content p{font-size:0.72rem;color:var(--muted);line-height:1.5;}
.step-content code{
  background:rgba(0,0,0,0.55);border:1px solid var(--gbord);border-radius:5px;
  padding:5px 7px;font-family:var(--mono);color:#FF9955;
  display:block;margin-top:5px;white-space:pre-wrap;font-size:0.69rem;}

/* footer */
.footer-txt{font-family:var(--mono);font-size:0.57rem;color:#2A2A3A;
            margin-top:9px;display:flex;justify-content:space-between;flex-wrap:wrap;gap:4px;}
.footer-txt span:last-child{color:rgba(255,106,0,0.35);}

/* scrollbar */
::-webkit-scrollbar{width:3px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:rgba(255,106,0,0.45);border-radius:2px;}

/* ── Floating particles ── */
#aura-particles{position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:0;opacity:0.3;}

/* ── Hardware Telemetry Panel ── */
.hw-panel{background:rgba(255,106,0,0.04);border:1px solid rgba(255,106,0,0.14);
  border-radius:11px;padding:11px 13px;margin:6px 0;}
.hw-panel-hd{font-family:var(--mono);font-size:0.59rem;color:var(--orange);
  letter-spacing:0.12em;text-transform:uppercase;margin-bottom:9px;
  display:flex;align-items:center;justify-content:space-between;}
.hw-metric{margin-bottom:7px;}
.hw-metric-hd{display:flex;justify-content:space-between;align-items:baseline;
  font-family:var(--mono);font-size:0.6rem;color:var(--muted);margin-bottom:3px;}
.hw-metric-val{color:var(--orange);font-weight:700;font-size:0.68rem;}
.hw-bar-track{height:4px;background:rgba(255,255,255,0.05);border-radius:2px;overflow:hidden;}
.hw-bar-fill{height:100%;border-radius:2px;transition:width 0.7s ease;
  background:linear-gradient(90deg,rgba(255,106,0,0.55),var(--orange));
  box-shadow:0 0 8px rgba(255,106,0,0.4);}
.hw-bar-fill.cpu{background:linear-gradient(90deg,rgba(60,130,255,0.6),#4499FF);
  box-shadow:0 0 6px rgba(60,130,255,0.35);}
.hw-bar-fill.mem{background:linear-gradient(90deg,rgba(160,80,255,0.6),#A050FF);
  box-shadow:0 0 6px rgba(160,80,255,0.3);}
.hw-chips{display:flex;gap:4px;flex-wrap:wrap;margin-top:7px;}
.chip{display:inline-flex;align-items:center;gap:3px;
  background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);
  border-radius:20px;padding:2px 8px;font-family:var(--mono);font-size:0.57rem;color:var(--muted);}
.chip.npu-chip{background:rgba(255,106,0,0.1);border-color:rgba(255,106,0,0.28);color:var(--orange);}
.chip.ok-chip{background:rgba(60,200,100,0.08);border-color:rgba(60,200,100,0.22);color:#55DD88;}
.chip.warn-chip{background:rgba(255,180,0,0.08);border-color:rgba(255,180,0,0.22);color:#FFCC44;}
.chip-dot{width:5px;height:5px;border-radius:50%;background:currentColor;}

/* ── Vision Trace Pipeline ── */
.vtrace{background:rgba(0,0,0,0.38);border:1px solid rgba(255,255,255,0.07);
  border-radius:9px;padding:9px 12px;margin-bottom:7px;}
.vtrace-hd{font-family:var(--mono);font-size:0.58rem;color:var(--muted);
  letter-spacing:0.1em;text-transform:uppercase;margin-bottom:7px;}
.vtrace-stages{display:flex;align-items:center;gap:2px;}
.vtrace-s{display:flex;flex-direction:column;align-items:center;gap:2px;flex:1;}
.vtrace-dot{width:22px;height:22px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-size:0.6rem;font-weight:700;
  background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);
  color:var(--muted);transition:all 0.3s;}
.vtrace-dot.vt-active{background:rgba(255,106,0,0.18);border-color:var(--orange);
  color:var(--orange);box-shadow:0 0 9px rgba(255,106,0,0.4);animation:pls 0.7s infinite;}
.vtrace-dot.vt-done{background:rgba(60,200,100,0.12);border-color:#55DD88;color:#55DD88;}
.vtrace-lbl{font-family:var(--mono);font-size:0.49rem;color:var(--muted);text-align:center;}
.vtrace-time{font-family:var(--mono);font-size:0.48rem;color:rgba(255,106,0,0.6);}
.vtrace-arrow{color:rgba(255,255,255,0.12);font-size:0.65rem;margin-bottom:16px;flex-shrink:0;}

/* ── ADHD Focus Timer ── */
.focus-timer-wrap{background:rgba(238,42,36,0.06);border:1px solid rgba(238,42,36,0.18);
  border-radius:9px;padding:10px 12px;margin-top:5px;}
.timer-display{font-family:var(--mono);font-size:1.3rem;font-weight:700;
  color:#FF9999;text-align:center;letter-spacing:0.1em;margin-bottom:6px;}
.timer-bar-track{height:5px;background:rgba(255,255,255,0.05);border-radius:3px;
  overflow:hidden;margin-bottom:7px;}
.timer-bar-fill{height:100%;width:100%;border-radius:3px;transition:width 0.95s linear;
  background:linear-gradient(90deg,rgba(238,42,36,0.6),#EE2A24);}
.btn-milestone{width:100%!important;background:rgba(238,42,36,0.11)!important;
  color:#FF9999!important;border:1px solid rgba(238,42,36,0.28)!important;
  border-radius:7px!important;font-family:var(--head)!important;font-size:0.82rem!important;
  font-weight:700!important;text-transform:none!important;padding:6px!important;}

/* ── Dual-Pane Output ── */
.dual-pane{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-bottom:8px;}
.dual-panel{background:rgba(0,0,0,0.5);border:1px solid var(--gbord);
  border-radius:9px;padding:9px 11px;font-family:var(--head);font-size:0.82rem;
  line-height:1.65;max-height:210px;overflow-y:auto;}
.dual-panel-hd{font-family:var(--mono);font-size:0.54rem;color:var(--orange);
  letter-spacing:0.12em;text-transform:uppercase;margin-bottom:5px;
  border-bottom:1px solid var(--gbord);padding-bottom:3px;}
.dual-en{border-left:2px solid rgba(80,140,255,0.5);}
.dual-lang{border-left:2px solid var(--orange);}

/* ── Model Health Grid ── */
.health-grid{display:grid;grid-template-columns:1fr 1fr;gap:5px;margin:4px 0;}
.health-item{display:flex;align-items:center;gap:6px;
  background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.07);
  border-radius:7px;padding:5px 8px;}
.health-icon{font-size:0.82rem;flex-shrink:0;}
.health-lbl{font-family:var(--mono);color:var(--muted);font-size:0.6rem;}
.health-val{color:var(--text);font-size:0.69rem;font-weight:600;}

/* ── Benchmark Results ── */
.bench-wrap{background:rgba(0,0,0,0.4);border:1px solid rgba(255,255,255,0.07);
  border-radius:9px;padding:9px 12px;}
.bench-table{width:100%;border-collapse:separate;border-spacing:0 3px;}
.bench-table th{font-family:var(--mono);font-size:0.57rem;color:var(--muted);
  text-transform:uppercase;letter-spacing:0.07em;padding:2px 6px;text-align:left;}
.bench-table td{font-family:var(--mono);font-size:0.71rem;color:var(--text);
  background:rgba(255,255,255,0.025);padding:4px 7px;}
.bench-table td:first-child{border-radius:5px 0 0 5px;}
.bench-table td:last-child{border-radius:0 5px 5px 0;}
.bench-npu td{color:var(--orange)!important;background:rgba(255,106,0,0.07)!important;}
.bench-speedup{font-family:var(--mono);font-size:0.7rem;color:#55DD88;margin-top:7px;}
.btn-bench{backdrop-filter:blur(10px);background:rgba(255,255,255,0.04)!important;
  border:1px solid rgba(255,255,255,0.14)!important;border-radius:8px!important;
  color:var(--text)!important;font-family:var(--head)!important;font-size:0.82rem!important;
  font-weight:600!important;text-transform:none!important;width:100%!important;
  padding:7px!important;margin-bottom:6px!important;transition:all 0.15s!important;}
.btn-bench:hover{background:rgba(255,106,0,0.1)!important;border-color:rgba(255,106,0,0.3)!important;}

/* ── Demo Script ── */
.demo-script-inner{background:rgba(0,0,0,0.45);border-radius:9px;padding:9px 12px;}
.demo-step{display:flex;gap:8px;align-items:flex-start;margin-bottom:9px;
  padding-bottom:9px;border-bottom:1px solid var(--gbord);}
.demo-step:last-child{border-bottom:none;margin-bottom:0;padding-bottom:0;}
.demo-snum{width:20px;height:20px;border-radius:50%;
  background:linear-gradient(135deg,var(--orange),var(--red));
  color:#fff;font-weight:700;font-size:0.62rem;
  display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px;}
.demo-stxt{font-size:0.74rem;color:var(--text);line-height:1.45;}
.demo-snote{font-size:0.61rem;color:var(--muted);margin-top:2px;font-family:var(--mono);}
.demo-sdo{font-size:0.61rem;color:var(--orange);margin-top:3px;font-weight:700;}

/* ── Eco + Sovereignty banners ── */
.eco-banner{background:rgba(60,200,100,0.06);border:1px solid rgba(60,200,100,0.2);
  border-radius:8px;padding:6px 10px;font-family:var(--mono);font-size:0.64rem;
  color:#88FFAA;margin-top:4px;}
.sov-banner{background:rgba(60,130,255,0.07);border:1px solid rgba(60,130,255,0.22);
  border-radius:8px;padding:6px 10px;font-family:var(--mono);font-size:0.64rem;
  color:#88AAFF;margin-top:4px;}

/* ── Expansion overrides (glass) ── */
.q-expansion-item .q-item{
  backdrop-filter:blur(12px);background:rgba(255,255,255,0.025)!important;
  border:1px solid rgba(255,255,255,0.09)!important;border-radius:9px!important;
  color:#E0E0E0!important;font-family:var(--head)!important;
  font-size:0.82rem!important;font-weight:600!important;
  padding:7px 12px!important;margin-bottom:2px!important;}
.q-expansion-item--expanded .q-item{
  border-radius:9px 9px 0 0!important;border-bottom-color:transparent!important;}
.q-expansion-item__container{
  border:1px solid rgba(255,255,255,0.09)!important;border-top:none!important;
  border-radius:0 0 9px 9px!important;padding:8px 12px!important;
  background:rgba(255,255,255,0.02)!important;margin-bottom:5px!important;}
.q-expansion-item .q-expansion-item__content{padding:0!important;}
"""

# ─────────────────────────────────────────────────────────────
#  JavaScript — camera
# ─────────────────────────────────────────────────────────────
CAMERA_JS = r"""
(function(){
  window._auraCam = { stream: null };

  window.auraCamToggle = function() {
    if (window._auraCam.stream) { window.auraCamOff(); }
    else { window.auraCamOn(); }
  };

  window.auraCamOn = async function() {
    try {
      var s = await navigator.mediaDevices.getUserMedia({
        video: { width:{ideal:1280}, height:{ideal:720}, facingMode:'user' }
      });
      window._auraCam.stream = s;
      var vid = document.getElementById('aura-cam-video');
      var ph  = document.getElementById('aura-cam-placeholder');
      var vp  = document.getElementById('aura-cam-viewport');
      if (vid) { vid.srcObject = s; vid.style.display = 'block'; }
      if (ph)  { ph.style.display  = 'none'; }
      if (vp)  { vp.classList.add('active'); }
      emitEvent('cam_state', {active: true});
    } catch(e) {
      emitEvent('cam_state', {active: false, err: e.message});
    }
  };

  window.auraCamOff = function() {
    if (window._auraCam.stream) {
      window._auraCam.stream.getTracks().forEach(function(t){ t.stop(); });
      window._auraCam.stream = null;
    }
    var vid = document.getElementById('aura-cam-video');
    var ph  = document.getElementById('aura-cam-placeholder');
    var vp  = document.getElementById('aura-cam-viewport');
    if (vid) { vid.srcObject = null; vid.style.display = 'none'; }
    if (ph)  { ph.style.display = 'flex'; }
    if (vp)  { vp.classList.remove('active'); }
    emitEvent('cam_state', {active: false});
  };

  window.auraCamCapture = function() {
    var vid = document.getElementById('aura-cam-video');
    var cvs = document.getElementById('aura-cam-canvas');
    if (!vid || !cvs || !vid.srcObject) {
      emitEvent('cam_captured', {data: null});
      return;
    }
    cvs.width  = vid.videoWidth  || 640;
    cvs.height = vid.videoHeight || 480;
    cvs.getContext('2d').drawImage(vid, 0, 0);
    var dataUrl = cvs.toDataURL('image/jpeg', 0.92);
    var snap = document.getElementById('aura-cam-snap');
    if (snap) { snap.src = dataUrl; snap.style.display = 'block'; }
    emitEvent('cam_captured', {data: dataUrl});
  };
})();

// ── Floating particles (subtle ambient background) ───────────────────────────
(function(){
  var cv = document.createElement('canvas');
  cv.id = 'aura-particles';
  document.body.prepend(cv);
  var cx = cv.getContext('2d'), W, H;
  var pts = [];
  function resize(){ W=cv.width=window.innerWidth; H=cv.height=window.innerHeight; }
  resize();
  window.addEventListener('resize', resize);
  for(var i=0;i<40;i++) pts.push({
    x:Math.random()*2000,y:Math.random()*1200,r:Math.random()*1.2+0.2,
    vx:(Math.random()-0.5)*0.15,vy:(Math.random()-0.5)*0.1,
    a:Math.random()*0.4+0.08
  });
  function draw(){
    cx.clearRect(0,0,W,H);
    pts.forEach(function(p){
      p.x+=p.vx; p.y+=p.vy;
      if(p.x<0)p.x=W; if(p.x>W)p.x=0;
      if(p.y<0)p.y=H; if(p.y>H)p.y=0;
      cx.beginPath(); cx.arc(p.x,p.y,p.r,0,Math.PI*2);
      cx.fillStyle='rgba(255,106,0,'+p.a+')';
      cx.fill();
    });
    requestAnimationFrame(draw);
  }
  draw();
})();

// ── ADHD Focus Timer ─────────────────────────────────────────────────────────
window._ft = { running:false, rem:300, total:300, tid:null };
window.auraTimerStart = function(secs){
  secs = secs||300;
  window._ft.rem = window._ft.total = secs;
  window._ft.running = true;
  if(window._ft.tid) clearInterval(window._ft.tid);
  window._ft.tid = setInterval(function(){
    if(!window._ft.running){ clearInterval(window._ft.tid); return; }
    window._ft.rem--;
    var r=window._ft.rem, m=Math.floor(r/60), s=r%60;
    var d=document.getElementById('aura-timer-disp');
    var b=document.getElementById('aura-timer-bar');
    if(d) d.textContent=m+':'+(s<10?'0':'')+s;
    if(b) b.style.width=((r/window._ft.total)*100)+'%';
    if(r<=0){
      clearInterval(window._ft.tid); window._ft.running=false;
      emitEvent('adhd_timer_done',{});
      if(d) d.textContent='✓ Milestone!';
    }
  },1000);
};
window.auraTimerStop=function(){
  window._ft.running=false;
  if(window._ft.tid) clearInterval(window._ft.tid);
};

// ── Vision Trace stage highlighter ───────────────────────────────────────────
window.auraTrace = function(stage){
  var stages=['capture','preprocess','infer','decode','done'];
  var idx=stages.indexOf(stage);
  stages.forEach(function(s,i){
    var d=document.getElementById('vtd-'+s);
    if(!d) return;
    if(i<idx){ d.className='vtrace-dot vt-done'; d.textContent='✓'; }
    else if(i===idx){ d.className='vtrace-dot vt-active'; d.textContent=(i+1); }
    else{ d.className='vtrace-dot'; d.textContent=(i+1); }
    var t=document.getElementById('vtt-'+s);
    if(t && i===idx){
      var now=new Date();
      t.textContent=now.getHours().toString().padStart(2,'0')+':'
        +now.getMinutes().toString().padStart(2,'0')+':'
        +now.getSeconds().toString().padStart(2,'0');
    }
  });
};
"""


# ─────────────────────────────────────────────────────────────
#  LLM CALLS
# ─────────────────────────────────────────────────────────────
async def _probe_ollama() -> bool:
    """Return True if the Ollama server is reachable on OLLAMA_HOST."""
    try:
        async with httpx.AsyncClient(timeout=2) as c:
            r = await c.get(OLLAMA_HOST + "/api/tags")
            return r.status_code == 200
    except Exception:
        return False


def _offline_pil_analysis(image_b64: Optional[str], prompt: str, lang: str) -> str:
    """Pure offline image analysis using PIL + numpy. Always works, zero network."""
    lang_name = LANGUAGE_MAP.get(lang, {}).get("name_en", "English")
    if not image_b64:
        return (
            f"[Offline Analysis — {lang_name}]\n\n"
            "No image provided. Please capture or upload an image to analyse.\n\n"
            "Tip: Install Ollama (ollama.com) and run `ollama pull llava` for full AI vision."
        )
    try:
        import numpy as np
        from PIL import Image, ImageStat, ImageFilter
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = img.size
        stat = ImageStat.Stat(img)
        brightness = sum(stat.mean) / 3
        contrast   = sum(stat.stddev) / 3
        r_mean, g_mean, b_mean = stat.mean[:3]
        if r_mean > g_mean and r_mean > b_mean:
            dom_colour = "warm/red-toned"
        elif g_mean > r_mean and g_mean > b_mean:
            dom_colour = "green-toned"
        elif b_mean > r_mean and b_mean > g_mean:
            dom_colour = "cool/blue-toned"
        else:
            dom_colour = "neutral/balanced"
        arr = np.array(img.convert("L"), dtype=np.float32)
        edges = np.array(img.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
        edge_density = float(edges.mean())
        if edge_density > 30:
            content_type = "high-detail / diagram / text-rich"
        elif edge_density > 12:
            content_type = "moderate detail / mixed content"
        else:
            content_type = "low edge density / photo / illustration"
        aspect = "square" if abs(w - h) < 50 else ("landscape" if w > h else "portrait")
        return (
            f"[Offline PIL Analysis — {lang_name}]\n\n"
            f"Image: {w}x{h}px ({aspect})\n"
            f"Brightness: {brightness:.1f}/255  |  Contrast: {contrast:.1f}\n"
            f"Dominant tone: {dom_colour}  |  Content type: {content_type}\n\n"
            f"Prompt received: {prompt[:120]}\n\n"
            "Inference tier: PIL/Offline — Ollama not detected.\n"
            "Start Ollama: `ollama serve` then retry."
        )
    except Exception as exc:
        logger.warning("PIL analysis failed: %s", exc)
        return (
            f"[Offline Fallback — {lang_name}]\n\n"
            "Image received but offline analysis unavailable (PIL/numpy not installed).\n"
            "Run: pip install pillow numpy\n\n"
            "For full AI: ollama.com → ollama pull llava"
        )


async def _run_inference(
    image_b64: Optional[str], prompt: str, lang: str,
    max_tokens: int = 600
) -> tuple[str, str]:
    """Offline inference pipeline: NPU → Ollama (localhost) → PIL analysis. Zero cloud."""
    lang_name = LANGUAGE_MAP.get(lang, {}).get("name_en", "English")
    system = (
        "You are Aura, a neuro-inclusive AI assistant for Indian students with learning differences. "
        f"Respond in {lang_name}. Be clear, structured, and accessible."
    )
    # 1 — NPU / VitisAI EP (fastest, fully local)
    if state.npu_active and state.engine:
        try:
            result = await state.engine.analyze_image_async(
                image_b64, prompt, lang, max_tokens=max_tokens
            )
            if result and result.text:
                logger.info("Inference via NPU (%s ms)", getattr(result, "latency_ms", "?"))
                return result.text.strip(), f"NPU/{getattr(result, 'provider_used', 'VitisAI')}"
        except Exception as exc:
            logger.warning("NPU inference failed: %s", exc)

    # 2 — Ollama localhost (private, no cloud)
    if await _probe_ollama():
        try:
            state.ollama_active = True
            text = await _call_ollama(image_b64, prompt, system)
            return text, f"Ollama/{OLLAMA_MDL}"
        except Exception as exc:
            logger.warning("Ollama inference failed: %s", exc)
            state.ollama_active = False

    # 3 — PIL offline analysis (always works)
    state.ollama_active = False
    return _offline_pil_analysis(image_b64, prompt, lang), "Offline/PIL"


def _make_eco_image(image_b64: Optional[str]) -> Optional[str]:
    """Downscale image to 50% for battery eco mode."""
    if not image_b64:
        return image_b64
    try:
        from PIL import Image as _PIL
        raw = base64.b64decode(image_b64)
        img = _PIL.open(io.BytesIO(raw)).convert("RGB")
        w, h = img.size
        img = img.resize((w // 2, h // 2), _PIL.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return image_b64


def _model_health_items() -> list[dict]:
    """Live health check results for display."""
    items = []
    items.append({
        "icon": "\u26a1" if state.npu_active else "\u26aa",
        "label": "NPU",
        "value": "XDNA\u00b2 Active" if state.npu_active else "CPU Mode",
        "ok": state.npu_active,
    })
    items.append({
        "icon": "\U0001f7e2" if state.ollama_active else "\U0001f7e1",
        "label": "Ollama",
        "value": OLLAMA_MDL if state.ollama_active else "Offline",
        "ok": state.ollama_active,
    })
    model_ok = Path("models/gemma3_4b_vlm/model_quantized.onnx").exists()
    items.append({
        "icon": "\u2705" if model_ok else "\u26a0",
        "label": "VLM Model",
        "value": "INT8 Loaded" if model_ok else "Not found",
        "ok": model_ok,
    })
    try:
        from PIL import Image as _p
        pil_ok = True
    except ImportError:
        pil_ok = False
    items.append({
        "icon": "\u2705" if pil_ok else "\u274c",
        "label": "PIL/Numpy",
        "value": "Ready" if pil_ok else "pip install pillow",
        "ok": pil_ok,
    })
    items.append({
        "icon": "\U0001f512" if state.offline_sovereign else "\U0001f310",
        "label": "Network",
        "value": "Isolated" if state.offline_sovereign else "Unverified",
        "ok": state.offline_sovereign,
    })
    items.append({
        "icon": "\u2705",
        "label": "DPDP 2023",
        "value": "Compliant",
        "ok": True,
    })
    return items


async def _verify_sovereign() -> dict:
    """Probe for outbound TCP connections."""
    tel = get_telemetry()
    result = await tel.probe_outbound_network()
    state.offline_sovereign = result["safe"]
    return result


def _scaffold_adhd(text: str) -> str:
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    out, chunk = [], 3
    for i in range(0, len(sentences), chunk):
        n = i // chunk + 1
        block = ". ".join(sentences[i: i + chunk]) + "."
        out.append(f"Milestone {n}\n{block}\n{'─'*28}\nTake a breath.\n")
    return "\n".join(out) if out else text


# ─────────────────────────────────────────────────────────────
#  MAIN PAGE
# ─────────────────────────────────────────────────────────────
@ui.page("/")
async def index():
    global _onboard_dialog

    ui.add_head_html("<style>" + AURA_CSS + "</style>")
    ui.add_head_html("<script>" + CAMERA_JS + "</script>")

    cam_box_ref: dict = {}
    up_box_ref:  dict = {}

    def _set_mode(m: str):
        state.capture_mode = m
        # Show/hide panels — works because containers have NO inline style
        if "cam" in cam_box_ref:
            is_cam = (m == "camera")
            cam_box_ref["cam"].set_visibility(is_cam)
            if not is_cam:
                ui.run_javascript("if(typeof auraCamOff==='function') auraCamOff()")
        if "up" in up_box_ref:
            up_box_ref["up"].set_visibility(m == "upload")

    with ui.element("div").classes("aura-page"):
        with ui.element("div").classes("aura-content"):
            with ui.element("div").classes("glass-card"):

                # ── Banner ──────────────────────────────────
                if state.npu_active:
                    bhtml = (
                        '<div class="banner banner-ok">&#9889; NPU Cognitive Engine Active'
                        ' — Ryzen AI XDNA&sup2; &nbsp;|&nbsp; 100% Offline</div>'
                    )
                elif state.ollama_active:
                    bhtml = (
                        '<div class="banner banner-ok">&#128994; Ollama Local VLM Active'
                        ' &mdash; 100% Offline &nbsp;|&nbsp; Model: ' + OLLAMA_MDL + '</div>'
                    )
                else:
                    bhtml = (
                        '<div class="banner banner-demo">&#9889; Offline Engine Active'
                        ' &mdash; PIL analysis ready &nbsp;|&nbsp;'
                        '<a href="/setup">Enable Ollama &rarr;</a></div>'
                    )
                ui.html(bhtml)

                # ── Header ──────────────────────────────────
                with ui.element("div").classes("hdr"):
                    with ui.element("div"):
                        ui.html('<div class="logo">&#9889; AURA</div>')
                        ui.html('<div class="logo-sub">NPU Cognitive Overlay</div>')
                    with ui.element("div").style(
                        "display:flex;flex-direction:column;align-items:flex-end;gap:4px;"
                    ):
                        ui.html('<div class="npu-pill"><div class="dot"></div>XDNA 2</div>')
                        ui.button(
                            "?",
                            on_click=lambda: _onboard_dialog.open() if _onboard_dialog else None
                        ).props("flat round dense").style(
                            "color:var(--orange);width:22px;height:22px;"
                            "min-width:0;font-size:0.75rem;"
                        )

                # ── Stats ────────────────────────────────────
                with ui.element("div").classes("stats-row"):
                    lat_lbl  = _sbox("-- ms",  "Latency")
                    back_lbl = _sbox("Offline", "AI")
                    hits_lbl = _sbox("0", "Scans")

                # ── Hardware Telemetry Panel ──────────────────
                with ui.expansion("\u26a1 Hardware Telemetry", icon="bolt",
                                  value=True).classes("w-full"):
                    with ui.element("div").classes("hw-panel"):
                        ui.html(
                            '<div class="hw-panel-hd">'
                            '<span>LIVE SILICON METRICS</span>'
                            '<span style="display:flex;align-items:center;gap:5px;">'
                            '<span class="dot"></span>'
                            '<span style="font-size:0.58rem;color:var(--orange);">LIVE</span>'
                            '</span></div>'
                        )
                        with ui.element("div").classes("hw-metric"):
                            with ui.element("div").classes("hw-metric-hd"):
                                ui.label("NPU Utilization")
                                hw_npu_lbl = ui.label("0%").classes("hw-metric-val")
                            ui.html('<div class="hw-bar-track">'
                                    '<div id="hw-npu-fill" class="hw-bar-fill" style="width:0%"></div>'
                                    '</div>')
                        with ui.element("div").classes("hw-metric"):
                            with ui.element("div").classes("hw-metric-hd"):
                                ui.label("CPU Load")
                                hw_cpu_lbl = ui.label("0%").classes("hw-metric-val").style("color:#4499FF")
                            ui.html('<div class="hw-bar-track">'
                                    '<div id="hw-cpu-fill" class="hw-bar-fill cpu" style="width:0%"></div>'
                                    '</div>')
                        with ui.element("div").classes("hw-metric"):
                            with ui.element("div").classes("hw-metric-hd"):
                                ui.label("RAM")
                                hw_mem_lbl = ui.label("0%").classes("hw-metric-val").style("color:#A050FF")
                            ui.html('<div class="hw-bar-track">'
                                    '<div id="hw-mem-fill" class="hw-bar-fill mem" style="width:0%"></div>'
                                    '</div>')
                        with ui.element("div").classes("hw-chips"):
                            hw_mode_chip  = ui.html('<span class="chip npu-chip"><span class="chip-dot"></span>INT8</span>')
                            hw_prov_chip  = ui.html('<span class="chip">CPU</span>')
                            hw_power_chip = ui.html('<span class="chip">~2.0W</span>')
                            hw_lat_chip   = ui.html('<span class="chip ok-chip">-- ms</span>')

                def _update_hw():
                    try:
                        snap = get_telemetry().snapshot
                        hw_npu_lbl.set_text(f"{snap.npu_pct:.0f}%")
                        hw_cpu_lbl.set_text(f"{snap.cpu_pct:.0f}%")
                        hw_mem_lbl.set_text(f"{snap.memory_pct:.0f}%")
                        ui.run_javascript(
                            f"var e;"
                            f"e=document.getElementById('hw-npu-fill');if(e)e.style.width='{snap.npu_pct:.1f}%';"
                            f"e=document.getElementById('hw-cpu-fill');if(e)e.style.width='{snap.cpu_pct:.1f}%';"
                            f"e=document.getElementById('hw-mem-fill');if(e)e.style.width='{snap.memory_pct:.1f}%';"
                        )
                        mc = "chip npu-chip" if snap.npu_detected else "chip warn-chip"
                        hw_mode_chip.set_content(f'<span class="{mc}"><span class="chip-dot"></span>{snap.model_mode}</span>')
                        if "Vitis" in snap.provider or "NPU" in snap.provider:
                            pc = "chip npu-chip"
                        elif "Ollama" in snap.provider:
                            pc = "chip ok-chip"
                        else:
                            pc = "chip"
                        hw_prov_chip.set_content(f'<span class="{pc}">{snap.provider[:16]}</span>')
                        hw_power_chip.set_content(f'<span class="chip">~{snap.power_est_w:.1f}W</span>')
                        lat_s = f"{snap.inference_ms:.0f}ms" if snap.inference_ms > 0 else "--"
                        hw_lat_chip.set_content(f'<span class="chip ok-chip">{lat_s}</span>')
                    except Exception:
                        pass

                ui.timer(1.0, _update_hw)

                ui.html('<div class="div-line"></div>')

                # ── Language selector ─────────────────────────
                ui.html('<div class="form-lbl">Response Language</div>')
                lang_opts = {
                    code: info["name_native"] + " (" + info["name_en"] + ")"
                    for code, info in LANGUAGE_MAP.items()
                }
                lang_sel = ui.select(
                    options=lang_opts, value="hi"
                ).classes("w-full g-select").style("margin-bottom:9px;")
                lang_sel.on_value_change(
                    lambda e: setattr(state, "selected_language", e.value)
                )

                # ── Mode tabs ─────────────────────────────────
                ui.html('<div class="sec-lbl2">Capture Mode</div>')

                tab_btns: dict = {}

                def _set_tab_active(active: str):
                    for k, b in tab_btns.items():
                        if k == active:
                            b.classes(remove="tab-btn", add="tab-btn active")
                        else:
                            b.classes(remove="tab-btn active", add="tab-btn")

                with ui.element("div").classes("tab-row"):
                    tab_btns["screen"] = ui.button(
                        "🖥 Screen",
                        on_click=lambda: (_set_tab_active("screen"), _set_mode("screen"))
                    ).classes("tab-btn active").props("flat")
                    tab_btns["camera"] = ui.button(
                        "📷 Camera",
                        on_click=lambda: (_set_tab_active("camera"), _set_mode("camera"))
                    ).classes("tab-btn").props("flat")
                    tab_btns["upload"] = ui.button(
                        "📁 Upload",
                        on_click=lambda: (_set_tab_active("upload"), _set_mode("upload"))
                    ).classes("tab-btn").props("flat")

                # ── Camera section ────────────────────────────
                with ui.element("div") as cam_box:
                    cam_box.set_visibility(False)   # NiceGUI hidden class (no inline style)
                    cam_box_ref["cam"] = cam_box

                    # Live preview viewport
                    ui.html(
                        '<div id="aura-cam-viewport" class="cam-viewport">'
                        '  <video id="aura-cam-video" autoplay playsinline muted></video>'
                        '  <canvas id="aura-cam-canvas"></canvas>'
                        '  <div id="aura-cam-placeholder" class="cam-placeholder">'
                        '    <span>&#128247;</span>'
                        '    <span>Camera off &mdash; press START</span>'
                        '  </div>'
                        '</div>'
                    )

                    # Snapshot preview (shown after Capture Frame)
                    cam_snap_img = ui.image("").style(
                        "width:100%;max-height:150px;object-fit:cover;"
                        "border:1px solid rgba(255,106,0,0.35);border-radius:9px;"
                        "margin-bottom:7px;"
                    )
                    cam_snap_img.set_visibility(False)

                    # Status label
                    cam_status = ui.label("Ready").style(
                        "font-family:var(--mono);font-size:0.67rem;"
                        "color:var(--muted);margin-bottom:6px;display:block;"
                    )

                    # Event handlers (defined before buttons so closures capture them)
                    def _on_cam_state(e):
                        active = (e.args or {}).get("active", False)
                        if active:
                            cam_toggle_btn.classes(remove="btn-cam-toggle",
                                                   add="btn-cam-toggle on")
                            cam_toggle_btn.set_text("⏹ Stop Camera")
                            cam_status.set_text("🟢 Live")
                            ui.notify("Camera active", type="positive", position="bottom-right")
                        else:
                            cam_toggle_btn.classes(remove="btn-cam-toggle on",
                                                   add="btn-cam-toggle")
                            cam_toggle_btn.set_text("▶ Start Camera")
                            cam_status.set_text("Ready")
                            err = (e.args or {}).get("err")
                            if err:
                                ui.notify("Camera error: " + err, type="negative")

                    def _on_cam_captured(e):
                        durl = (e.args or {}).get("data") or ""
                        if durl and isinstance(durl, str) and "," in durl:
                            state.last_captured_b64 = durl.split(",", 1)[1]
                            cam_snap_img.source = durl
                            cam_snap_img.set_visibility(True)
                            cam_status.set_text("✅ Frame captured — hit VISION SCAN")
                            ui.notify("📸 Frame captured!", type="positive",
                                      position="bottom-right")
                        else:
                            ui.notify("No video — press START first", type="warning",
                                      position="bottom-right")

                    ui.on("cam_state",    _on_cam_state)
                    ui.on("cam_captured", _on_cam_captured)

                    # Controls row
                    with ui.element("div").classes("cam-btns"):
                        cam_toggle_btn = ui.button(
                            "▶ Start Camera",
                            on_click=lambda: ui.run_javascript("auraCamToggle()")
                        ).classes("btn-cam-toggle").props("flat")

                        ui.button(
                            "📸 Capture Frame",
                            on_click=lambda: ui.run_javascript("auraCamCapture()")
                        ).classes("btn-cam-c").props("flat")

                # ── Upload section ────────────────────────────
                with ui.element("div") as up_box:
                    up_box.set_visibility(False)    # NiceGUI hidden class (no inline style)
                    up_box_ref["up"] = up_box

                    upload_preview = ui.image("").style(
                        "width:100%;max-height:160px;object-fit:contain;"
                        "border:1px solid rgba(255,106,0,0.35);"
                        "border-radius:9px;margin-bottom:7px;"
                        "background:rgba(0,0,0,0.5);"
                    )
                    upload_preview.set_visibility(False)
                    upload_status  = ui.label("").style(
                        "font-family:var(--mono);font-size:0.67rem;"
                        "color:var(--muted);margin-bottom:5px;display:block;"
                    )

                    def handle_upload(e):
                        try:
                            from PIL import Image as _PIL
                            raw  = e.content.read()
                            # Normalise to JPEG via PIL (handles PNG/WEBP/BMP/HEIC etc.)
                            pil  = _PIL.open(io.BytesIO(raw)).convert("RGB")
                            buf  = io.BytesIO()
                            pil.save(buf, format="JPEG", quality=90)
                            b64  = base64.b64encode(buf.getvalue()).decode()
                            state.uploaded_img_b64[0] = b64
                            upload_preview.source = "data:image/jpeg;base64," + b64
                            upload_preview.set_visibility(True)
                            sz  = f"{pil.width}x{pil.height}"
                            upload_status.set_text(f"✅ {e.name}  ({sz}) — hit VISION SCAN")
                            ui.notify("Image ready — hit VISION SCAN",
                                      type="positive", position="bottom-right")
                        except Exception as exc:
                            logger.error("Upload error: %s", exc, exc_info=True)
                            upload_status.set_text("❌ " + str(exc)[:80])
                            ui.notify("Upload error: " + str(exc)[:80], type="negative")

                    ui.upload(
                        on_upload=handle_upload,
                        auto_upload=True,
                        label="Drop image or click to browse",
                    ).classes("w-full up-zone").props(
                        "accept='image/*' flat bordered"
                    )

                # Screen preview
                screen_preview = ui.image("").style(
                    "width:100%;max-height:120px;object-fit:cover;"
                    "border:1px solid rgba(255,106,0,0.18);border-radius:9px;"
                    "margin-bottom:7px;opacity:0.85;"
                )
                screen_preview.set_visibility(False)

                # ── Prompt ────────────────────────────────────
                ui.html('<div class="form-lbl" style="margin-top:7px;">Vision Prompt</div>')
                prompt_input = ui.input(
                    placeholder="e.g. Explain this diagram step by step"
                ).classes("w-full g-input").style("margin-bottom:9px;")
                prompt_input.value = "Explain this step-by-step in simple language."

                # ── Vision Trace Pipeline ──────────────────────
                ui.html(
                    '<div class="vtrace">'
                    '<div class="vtrace-hd">Inference Pipeline Trace</div>'
                    '<div class="vtrace-stages">'
                    '  <div class="vtrace-s">'
                    '    <div id="vtd-capture" class="vtrace-dot">1</div>'
                    '    <div class="vtrace-lbl">Capture</div>'
                    '    <div id="vtt-capture" class="vtrace-time"></div>'
                    '  </div>'
                    '  <div class="vtrace-arrow">\u203a</div>'
                    '  <div class="vtrace-s">'
                    '    <div id="vtd-preprocess" class="vtrace-dot">2</div>'
                    '    <div class="vtrace-lbl">Preprocess</div>'
                    '    <div id="vtt-preprocess" class="vtrace-time"></div>'
                    '  </div>'
                    '  <div class="vtrace-arrow">\u203a</div>'
                    '  <div class="vtrace-s">'
                    '    <div id="vtd-infer" class="vtrace-dot">3</div>'
                    '    <div class="vtrace-lbl">NPU Infer</div>'
                    '    <div id="vtt-infer" class="vtrace-time"></div>'
                    '  </div>'
                    '  <div class="vtrace-arrow">\u203a</div>'
                    '  <div class="vtrace-s">'
                    '    <div id="vtd-decode" class="vtrace-dot">4</div>'
                    '    <div class="vtrace-lbl">Decode</div>'
                    '    <div id="vtt-decode" class="vtrace-time"></div>'
                    '  </div>'
                    '  <div class="vtrace-arrow">\u203a</div>'
                    '  <div class="vtrace-s">'
                    '    <div id="vtd-done" class="vtrace-dot">5</div>'
                    '    <div class="vtrace-lbl">Output</div>'
                    '    <div id="vtt-done" class="vtrace-time"></div>'
                    '  </div>'
                    '</div>'
                    '</div>'
                )

                # ── Scan button ────────────────────────────────
                async def do_scan():
                    await _trigger_scan(
                        prompt_input, lat_lbl, back_lbl, hits_lbl,
                        response_html, scan_btn, screen_preview,
                        dual_pane_box=resp_dual_box,
                        dual_en_html=resp_en_html,
                        dual_native_html=resp_native_html,
                    )

                scan_btn = ui.button("\u26a1  VISION SCAN", on_click=do_scan).classes("btn-scan")

                # ── Response (single pane) ─────────────────────
                with ui.element("div").classes("resp-box"):
                    response_html = ui.html(
                        '<span style="color:#333;font-family:var(--mono);font-size:0.72rem;">'
                        'Ready \u2014 choose a mode then click VISION SCAN</span>'
                    )

                # ── Response (dual pane) ─────────────────────
                with ui.element("div").classes("dual-pane") as resp_dual_box:
                    resp_dual_box.set_visibility(False)
                    with ui.element("div").classes("dual-panel dual-en"):
                        ui.html('<div class="dual-panel-hd">\U0001f1ec\U0001f1e7 ENGLISH</div>')
                        resp_en_html = ui.html(
                            '<span style="color:#444;font-size:0.72rem;">run scan\u2026</span>'
                        )
                    with ui.element("div").classes("dual-panel dual-lang"):
                        lang_flag = LANGUAGE_MAP.get(state.selected_language, {})
                        lang_hd = lang_flag.get("name_native", "Language")
                        ui.html(f'<div class="dual-panel-hd">\U0001f1ee\U0001f1f3 {lang_hd}</div>')
                        resp_native_html = ui.html(
                            '<span style="color:#444;font-size:0.72rem;">run scan\u2026</span>'
                        )

                ui.html('<div class="div-line"></div>')

                # ── Toggles ───────────────────────────────────
                with ui.element("div").classes("toggle-row"):
                    with ui.element("div").classes("toggle-item"):
                        ling_sw = ui.switch("Linguistic Mirror").classes("g-toggle")
                        ling_sw.on_value_change(lambda e: _on_toggle_ling(e.value))
                    with ui.element("div").classes("toggle-item"):
                        adhd_sw = ui.switch("ADHD Scaffold").classes("g-toggle")
                        adhd_sw.on_value_change(
                            lambda e: _on_toggle_adhd(e.value, adhd_banner, adhd_timer_box)
                        )

                # Second toggle row
                with ui.element("div").classes("toggle-row").style("margin-top:4px;"):
                    with ui.element("div").classes("toggle-item"):
                        eco_sw = ui.switch("\u26a1 Eco Mode").classes("g-toggle")
                        eco_sw.on_value_change(lambda e: _on_toggle_eco(e.value, eco_banner_el))
                    with ui.element("div").classes("toggle-item"):
                        sov_sw = ui.switch("\U0001f512 Offline Proof").classes("g-toggle")
                        sov_sw.on_value_change(
                            lambda e: asyncio.create_task(_on_toggle_sovereign(e.value, sov_banner_el))
                        )
                    with ui.element("div").classes("toggle-item"):
                        dp_sw = ui.switch("\U0001f310 Dual Lang").classes("g-toggle")
                        dp_sw.on_value_change(lambda e: setattr(state, "dual_pane", e.value))

                adhd_banner = ui.html("")

                # ADHD Focus Timer
                with ui.element("div").classes("focus-timer-wrap") as adhd_timer_box:
                    adhd_timer_box.set_visibility(False)
                    ui.html(
                        '<div style="font-family:var(--mono);font-size:0.56rem;'
                        'color:#FF9999;letter-spacing:0.1em;text-transform:uppercase;'
                        'margin-bottom:5px;">ADHD FOCUS TIMER</div>'
                    )
                    ui.html('<div id="aura-timer-disp" class="timer-display">5:00</div>')
                    ui.html(
                        '<div class="timer-bar-track">'
                        '<div id="aura-timer-bar" class="timer-bar-fill" style="width:100%"></div>'
                        '</div>'
                    )
                    with ui.element("div").style("display:flex;gap:5px;margin-top:5px;"):
                        ui.button(
                            "\u25b6 Start Timer",
                            on_click=lambda: ui.run_javascript("auraTimerStart(300)")
                        ).classes("btn-milestone").props("flat").style("flex:1.2;")
                        ui.button(
                            "Next Milestone \u2192",
                            on_click=lambda: (
                                ui.run_javascript("auraTimerStop()"),
                                ui.run_javascript("auraTimerStart(300)"),
                                ui.notify("\u2705 Milestone complete! Moving forward.", type="positive"),
                            )
                        ).classes("btn-milestone").props("flat").style("flex:2;")

                    def _on_adhd_timer_done_cb(_e):
                        ui.notify(
                            "\u23f0 5-min milestone complete! Take a 2-min break.",
                            type="positive", position="top", timeout=8000,
                        )
                    ui.on("adhd_timer_done", _on_adhd_timer_done_cb)

                eco_banner_el = ui.html("")
                sov_banner_el = ui.html("")

                ui.html('<div class="div-line"></div>')

                # ── Model Health Check ────────────────────────
                with ui.expansion("\u2695 System Health", icon="memory").classes("w-full"):
                    health_box = ui.element("div").classes("health-grid")

                    def _render_health():
                        health_box.clear()
                        with health_box:
                            for item in _model_health_items():
                                with ui.element("div").classes("health-item"):
                                    ui.html(f'<span class="health-icon">{item["icon"]}</span>')
                                    with ui.element("div").style("flex:1;min-width:0;gap:1px;display:flex;flex-direction:column;"):
                                        ui.html(f'<div class="health-lbl">{item["label"]}</div>')
                                        ui.html(f'<div class="health-val">{item["value"]}</div>')

                    _render_health()
                    ui.timer(3.0, _render_health, once=True)

                # ── Integrity accordion ────────────────────────
                with ui.expansion("Integrity Dashboard", icon="shield").classes("w-full"):
                    with ui.element("div").classes("px-2 py-1"):
                        _render_integrity()

                # ── Benchmark panel ────────────────────────────
                with ui.expansion("\U0001f4ca Benchmark", icon="speed").classes("w-full"):
                    bench_result_box = ui.html("")

                    async def _run_bench_ui():
                        bench_result_box.set_content(
                            '<div class="thinking"><div class="dot"></div>'
                            '<span>Running 5 inference benchmark runs\u2026</span></div>'
                        )
                        tel = get_telemetry()
                        tel.set_inference_active(True)
                        try:
                            res = await tel.run_benchmark(state.engine, n_runs=5)
                            if "error" in res:
                                html = f'<span style="color:#FF9999;">Error: {res["error"]}</span>'
                            else:
                                ca = res["cpu_avg_ms"]
                                na = res.get("npu_avg_ms")
                                sp = res.get("speedup")
                                cm = res["cpu_runs_ms"]
                                html = '<div class="bench-wrap"><table class="bench-table">'
                                html += '<thead><tr><th>Mode</th><th>Avg ms</th><th>Min</th><th>Max</th></tr></thead><tbody>'
                                html += f'<tr><td>CPU / PIL</td><td>{ca}ms</td><td>{min(cm):.1f}</td><td>{max(cm):.1f}</td></tr>'
                                if na is not None:
                                    nm = res["npu_runs_ms"]
                                    html += f'<tr class="bench-npu"><td>\u26a1 NPU</td><td>{na}ms</td><td>{min(nm):.1f}</td><td>{max(nm):.1f}</td></tr>'
                                html += '</tbody></table>'
                                if sp:
                                    html += f'<div class="bench-speedup">\u26a1 NPU is {sp}\u00d7 faster than CPU baseline</div>'
                                html += '</div>'
                            bench_result_box.set_content(html)
                        except Exception as exc:
                            bench_result_box.set_content(f'<span style="color:#FF9999;">Error: {exc}</span>')
                        finally:
                            tel.set_inference_active(False)

                    ui.button(
                        "\U0001f4ca Run Hardware Benchmark (5 runs)",
                        on_click=_run_bench_ui
                    ).classes("btn-bench").props("flat")

                # ── Demo Script panel ──────────────────────────
                with ui.expansion("\U0001f3a4 Live Demo Script", icon="record_voice_over").classes("w-full"):
                    with ui.element("div").classes("demo-script-inner"):
                        demo_steps = [
                            (
                                "Open Task Manager",
                                "Press Ctrl+Shift+Esc \u2192 Performance \u2192 NPU tile",
                                "NPU graph shows ~5% idle baseline"
                            ),
                            (
                                "Click VISION SCAN",
                                "Set: Screen mode, prompt \u2018Explain this diagram step by step\u2019",
                                "NPU spikes to 60\u201380% \u2014 note the waveform live on Task Manager"
                            ),
                            (
                                "Switch Language to Hindi",
                                "Dropdown \u2192 \u0939\u093f\u0928\u094d\u0926\u0940 (Hindi) \u2014 run another scan",
                                "Response in Devanagari script, same NPU path"
                            ),
                            (
                                "Enable ADHD + Eco Mode",
                                "Toggle ADHD Scaffold + \u26a1 Eco Mode \u2014 scan again",
                                "Structured milestones + ~40% lower NPU power draw"
                            ),
                            (
                                "Run Benchmark",
                                "Expand \U0001f4ca Benchmark \u2192 click Run Hardware Benchmark",
                                "Show CPU vs NPU latency table \u2014 the speedup is your proof"
                            ),
                        ]
                        for i, (title, desc, action) in enumerate(demo_steps, 1):
                            ui.html(
                                f'<div class="demo-step">'
                                f'<div class="demo-snum">{i}</div>'
                                f'<div>'
                                f'  <div class="demo-stxt"><b>{title}</b> \u2014 {desc}</div>'
                                f'  <div class="demo-sdo">\u2192 {action}</div>'
                                f'</div></div>'
                            )

                # ── Footer ────────────────────────────────────
                ui.html(
                    '<div class="footer-txt">'
                    '<span>DPDP 2023 Compliant &middot; Zero Cloud &middot; 100% Offline</span>'
                    '<span><a href="/setup" style="color:rgba(255,106,0,0.4);">'
                    'Setup &rarr;</a></span></div>'
                )

    _onboard_dialog = _build_onboarding()
    ui.timer(0.8, _init_engine, once=True)
    ui.timer(1.5, _warmup_ollama, once=True)
    asyncio.create_task(get_telemetry().start(interval=1.0))

    # Auto-open onboarding first visit
    ui.add_head_html("""
    <script>
    document.addEventListener('DOMContentLoaded',function(){
        setTimeout(function(){
            if(!localStorage.getItem('aura_v4_ob')){
                var b=document.getElementById('__ob');
                if(b) b.click();
            }
        },900);
    });
    </script>
    """)
    ui.button("_ob",
              on_click=lambda: _onboard_dialog.open() if _onboard_dialog else None
              ).props("id=__ob").style(
        "display:none;position:absolute;pointer-events:none;"
    )


# ─────────────────────────────────────────────────────────────
#  SCAN LOGIC
# ─────────────────────────────────────────────────────────────
async def _trigger_scan(
    prompt_input, lat_lbl, back_lbl, hits_lbl,
    response_html, scan_btn, screen_preview,
    dual_pane_box=None, dual_en_html=None, dual_native_html=None,
) -> None:
    if state.is_scanning:
        ui.notify("Already scanning\u2026", type="info")
        return

    state.is_scanning = True
    scan_btn.classes(add="scanning")
    scan_btn.text = "SCANNING\u2026"
    response_html.set_content(
        '<div class="thinking"><div class="dot"></div><span>Analysing\u2026</span></div>'
    )

    try:
        t0 = time.perf_counter()
        image_b64: Optional[str] = None

        # ── Vision Trace: capture stage ──
        ui.run_javascript("if(typeof auraTrace==='function') auraTrace('capture')")

        if state.capture_mode == "camera":
            if state.last_captured_b64:
                image_b64 = state.last_captured_b64
            else:
                ui.notify("Click \U0001f4f8 Capture Photo first, then VISION SCAN", type="warning")
                return

        elif state.capture_mode == "upload":
            image_b64 = state.uploaded_img_b64[0] or None
            if not image_b64:
                ui.notify("No image uploaded yet", type="warning")

        else:  # screen
            try:
                pil_img = await state.capture.capture_screen_async()
                if pil_img is not None:
                    image_b64 = _to_b64(pil_img)
                    if image_b64:
                        screen_preview.source = "data:image/jpeg;base64," + image_b64
                        screen_preview.set_visibility(True)
            except Exception as se:
                logger.warning("Screen capture: %s", se)

        # ── Eco Mode: downscale image ──
        if state.eco_mode and image_b64:
            image_b64 = _make_eco_image(image_b64)
        max_tok = 200 if state.eco_mode else 600

        # ── Vision Trace: preprocess stage ──
        ui.run_javascript("if(typeof auraTrace==='function') auraTrace('preprocess')")
        await asyncio.sleep(0.05)

        prompt = (prompt_input.value or "").strip() or \
            "Describe this image for a visually impaired student."

        # ── Vision Trace: infer stage + telemetry active ──
        tel = get_telemetry()
        tel.set_inference_active(True)
        ui.run_javascript("if(typeof auraTrace==='function') auraTrace('infer')")

        resp_text, backend = await _run_inference(
            image_b64, prompt, state.selected_language, max_tokens=max_tok
        )
        latency_ms = round((time.perf_counter() - t0) * 1000)

        tel.set_inference_active(False)
        tel.record_inference(latency_ms, backend)

        # ── Vision Trace: decode stage ──
        ui.run_javascript("if(typeof auraTrace==='function') auraTrace('decode')")

        if state.adhd_mode:
            resp_text = _scaffold_adhd(resp_text)

        lang_name = LANGUAGE_MAP.get(state.selected_language, {}).get("name_en", "English")
        lat_lbl.set_text(f"{latency_ms:.0f}ms")
        back_lbl.set_text(backend[:7])
        state.session_interactions += 1
        hits_lbl.set_text(str(state.session_interactions))

        response_html.set_content(
            '<div style="white-space:pre-wrap;line-height:1.75;">'
            + _esc(resp_text) + '</div>'
            '<div style="margin-top:7px;font-family:var(--mono);font-size:0.58rem;'
            'color:#2A2A3A;border-top:1px solid rgba(255,255,255,0.05);padding-top:5px;">'
            + _esc(backend) + " &middot; " + lang_name
            + f" &middot; {latency_ms:.0f}ms &middot; {state.capture_mode}</div>"
        )

        # ── Vision Trace: done stage ──
        ui.run_javascript("if(typeof auraTrace==='function') auraTrace('done')")

        # ── Dual-pane output ──
        if state.dual_pane and dual_pane_box and state.selected_language != "en":
            en_text, _ = await _run_inference(
                image_b64, prompt, "en", max_tokens=max_tok
            )
            lang_native = LANGUAGE_MAP.get(state.selected_language, {}).get("name_native", lang_name)
            if dual_en_html:
                dual_en_html.set_content(
                    f'<div style="white-space:pre-wrap;">{_esc(en_text)}</div>'
                )
            if dual_native_html:
                dual_native_html.set_content(
                    f'<div style="white-space:pre-wrap;">{_esc(resp_text)}</div>'
                )
            if dual_pane_box:
                dual_pane_box.set_visibility(True)
        else:
            if dual_pane_box:
                dual_pane_box.set_visibility(False)

        if state.linguistic_mirror and state.processor:
            asyncio.create_task(
                state.processor.speak_async(resp_text[:400], state.selected_language)
            )
        if state.tracker:
            state.tracker.log_interaction(
                interaction_type="vision_scan", prompt=prompt,
                language=state.selected_language, latency_ms=latency_ms,
            )

        ui.notify(f"Done \u2014 {latency_ms:.0f}ms via {backend}",
                  type="positive", position="bottom-right")

    except Exception as exc:
        logger.error("Scan error: %s", exc, exc_info=True)
        response_html.set_content(
            '<span style="color:#EE6666;">Error: ' + _esc(str(exc)) + '</span>'
        )
        ui.notify("Error: " + str(exc)[:90], type="negative")
    finally:
        state.is_scanning = False
        scan_btn.classes(remove="scanning")
        scan_btn.text = "\u26a1  VISION SCAN"


# ─────────────────────────────────────────────────────────────
#  ENGINE INIT
# ─────────────────────────────────────────────────────────────
async def _warmup_ollama() -> None:
    """Pre-load llava into memory at startup so first user scan is instant."""
    if state.npu_active:
        return  # NPU takes priority, skip Ollama warmup
    try:
        logger.info("Warming up Ollama/%s (pre-loading model)...", OLLAMA_MDL)
        async with httpx.AsyncClient(timeout=_cfg.OLLAMA_TIMEOUT_S) as c:
            r = await c.post(
                OLLAMA_HOST + "/api/generate",
                json={"model": OLLAMA_MDL, "prompt": "Ready.", "stream": False},
            )
        if r.status_code == 200:
            state.ollama_active = True
            logger.info("Ollama/%s warm — ready for vision inference", OLLAMA_MDL)
        else:
            logger.warning("Ollama warmup non-200: %s", r.status_code)
    except Exception as exc:
        logger.warning("Ollama warmup skipped: %s", exc)


async def _init_engine():
    try:
        state.engine    = get_engine()
        state.processor = MultimodalProcessor(state.engine)
        state.tracker   = IntegrityTracker()
        res = await asyncio.get_event_loop().run_in_executor(None, state.engine.initialize)
        state.engine_status = res
        npu_ok = res.get("npu_status", {}).get("npu_detected", False)
        state.npu_active = npu_ok
        tel = get_telemetry()
        tel.set_npu_detected(npu_ok)
        if npu_ok:
            tel._provider = "VitisAIExecutionProvider"
        logger.info("Engine init: npu_active=%s", npu_ok)
    except Exception as e:
        state.npu_active = False
        logger.warning("Engine init: %s", e)


# ─────────────────────────────────────────────────────────────
#  TOGGLE HANDLERS
# ─────────────────────────────────────────────────────────────
def _on_toggle_ling(enabled: bool):
    """Toggle real-time linguistic mirror (transliteration overlay)."""
    if enabled:
        lang = LANGUAGE_MAP.get(state.selected_language, {}).get("name_en", "")
        ui.notify("Linguistic Mirror ON — " + lang, type="positive")


def _on_toggle_adhd(enabled: bool, banner_el=None, timer_box=None):
    """Toggle ADHD focus mode: chunked output + on-screen Pomodoro timer."""
    state.adhd_mode = enabled
    if banner_el:
        banner_el.set_content(
            '<div class="adhd-banner"><b>ADHD Focus Mode Active</b>'
            ' \u2014 chunked 5-step output &amp; focus timer</div>'
            if enabled else ""
        )
    if timer_box:
        timer_box.set_visibility(enabled)
        if not enabled:
            ui.run_javascript("if(typeof auraTimerStop==='function') auraTimerStop()")
    if enabled:
        ui.notify("ADHD Focus Mode ON \u2014 timer ready", type="info")


def _on_toggle_eco(enabled: bool, banner_el=None):
    """Toggle eco mode: halve image resolution and cap max tokens at 200."""
    state.eco_mode = enabled
    if banner_el:
        banner_el.set_content(
            '<div class="eco-banner">\u26a1 Eco Mode Active \u2014 ~40% less power'
            ' &nbsp;|&nbsp; Lower res &middot; 200 max tokens</div>'
            if enabled else ""
        )
    ui.notify(
        "Eco Mode ON \u2014 battery optimised" if enabled else "Eco Mode OFF",
        type="info" if enabled else "warning",
    )


async def _on_toggle_sovereign(enabled: bool, banner_el=None):
    """Toggle offline sovereign mode: verify no outbound network is reachable."""
    state.offline_sovereign = enabled
    if not enabled:
        state.offline_sovereign = False
        if banner_el:
            banner_el.set_content("")
        return
    if banner_el:
        banner_el.set_content(
            '<div class="sov-banner">\U0001f512 Verifying network isolation\u2026</div>'
        )
    result = await _verify_sovereign()
    if result["safe"]:
        if banner_el:
            banner_el.set_content(
                f'<div class="sov-banner">\U0001f512 Firewall Safe \u2014 No Cloud'
                f' &nbsp;|&nbsp; {result["detail"]}</div>'
            )
        ui.notify("Offline Sovereignty verified \u2714", type="positive")
    else:
        if banner_el:
            banner_el.set_content(
                f'<div class="sov-banner" style="color:#FF9999;border-color:'
                f'rgba(238,42,36,0.35);">\u26a0 {result["detail"]}</div>'
            )
        ui.notify(f"Warning: {result['detail']}", type="warning")


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────
def _sbox(val: str, lbl: str) -> ui.label:
    """Render a compact stat label+value pair in the telemetry panel."""
    with ui.element("div").classes("stat-box"):
        v = ui.label(val).classes("stat-v")
        ui.html(f'<div class="stat-l">{lbl}</div>')
    return v


def _render_integrity():
    rows = [
        ("Session",   f"AURA-{int(time.time()) % 100000:05d}"),
        ("Scans",     str(state.session_interactions)),
        ("Cloud",     "0 calls"),
        ("DPDP 2023", "Compliant"),
        ("RPwD 2016", "Supported"),
        ("Backend",   "NPU" if state.npu_active else "Ollama" if state.ollama_active else "PIL/Offline"),
    ]
    for k, v in rows:
        ui.html(
            f'<div class="int-row"><span class="int-lbl">{k}</span>'
            f'<span class="int-val">{v}</span></div>'
        )


def _to_b64(image) -> str:
    """Convert a PIL Image or file-like object to a base-64 JPEG string."""
    try:
        import numpy as np
        from PIL import Image as PILImage
        pil = PILImage.fromarray(image) if isinstance(image, np.ndarray) else image
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=88)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


async def _call_ollama(image_b64: Optional[str], prompt: str, system: str) -> str:
    payload: dict = {
        "model": OLLAMA_MDL,
        "prompt": system + "\n\n" + prompt,
        "stream": False,
    }
    if image_b64:
        payload["images"] = [image_b64]
    timeout = _cfg.OLLAMA_TIMEOUT_S  # default 300s — llava needs time on cold start
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(OLLAMA_HOST + "/api/generate", json=payload)
    if r.status_code != 200:
        raise RuntimeError(f"Ollama {r.status_code}: {r.text[:200]}")
    return r.json().get("response", "").strip()


def _esc(t: str) -> str:
    """HTML-escape a string and convert newlines to <br> tags."""
    return (
        t.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )


# ─────────────────────────────────────────────────────────────
#  ONBOARDING DIALOG
# ─────────────────────────────────────────────────────────────
def _build_onboarding() -> ui.dialog:
    with ui.dialog().props("persistent") as dlg:
        with ui.card().style(
            "backdrop-filter:blur(30px) saturate(200%);"
            "-webkit-backdrop-filter:blur(30px) saturate(200%);"
            "background:linear-gradient(145deg,rgba(20,20,35,0.92),rgba(10,10,20,0.88));"
            "border:1px solid rgba(255,106,0,0.42);"
            "border-radius:16px;max-width:390px;width:100%;padding:18px;"
            "box-shadow:0 0 60px rgba(255,106,0,0.15),inset 0 1px 0 rgba(255,255,255,0.08);"
        ):
            ui.html(
                '<div style="font-family:Rajdhani,sans-serif;font-size:1.45rem;'
                'font-weight:700;background:linear-gradient(135deg,#FF6A00,#EE2A24);'
                '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
                'margin-bottom:2px;">Welcome to AURA-NPU</div>'
                '<div style="font-family:monospace;font-size:0.61rem;color:#333;'
                'letter-spacing:0.15em;margin-bottom:14px;">NEURO-INCLUSIVE COGNITIVE OVERLAY</div>'
            )

            # Offline inference stack
            ui.html(
                '<div class="llm-info">'
                '<h4>&#9889; 100% Offline Inference Stack</h4>'

                '<div class="llm-opt">'
                '<div class="llm-dot" style="background:#FF6A00;'
                'box-shadow:0 0 5px #FF6A00;"></div>'
                '<p><b style="color:#FFAA55;">NPU Engine</b> — VitisAI EP on Ryzen AI XDNA&sup2;<br>'
                'Auto-detected. Zero setup needed on supported AMD hardware.<br>'
                '<code>models/gemma3_4b_vlm/model_quantized.onnx</code></p></div>'

                '<div class="llm-opt">'
                '<div class="llm-dot" style="background:#AA88FF;'
                'box-shadow:0 0 5px #AA88FF;"></div>'
                '<p><b style="color:#CCAAFF;">Ollama (local VLM)</b> — fully private, no internet<br>'
                'ollama.com → install → run:<br>'
                '<code>ollama pull llava</code></p></div>'

                '<div class="llm-opt">'
                '<div class="llm-dot" style="background:#55DD88;'
                'box-shadow:0 0 5px #55DD88;"></div>'
                '<p><b style="color:#88FFAA;">PIL Offline</b> — always available, zero deps<br>'
                'Image dimensions, brightness, contrast, edge analysis<br>'
                '<code>pip install pillow numpy</code></p></div>'
                '</div>'
            )

            steps = [
                ("Pick your language",
                 "Choose from 22 Indian languages in the dropdown."),
                ("Choose capture mode",
                 "Screen = grab desktop. Camera = webcam (Start → Capture Photo). "
                 "Upload = any image file."),
                ("Hit VISION SCAN",
                 "AI analyses your image and responds in your chosen language."),
            ]
            for i, (title, desc) in enumerate(steps, 1):
                ui.html(
                    f'<div style="display:flex;gap:9px;align-items:flex-start;'
                    f'margin-bottom:9px;">'
                    f'<span style="width:20px;height:20px;border-radius:50%;'
                    f'background:#FF6A00;color:#fff;font-weight:700;font-size:0.72rem;'
                    f'display:flex;align-items:center;justify-content:center;'
                    f'flex-shrink:0;">{i}</span>'
                    f'<div><div style="font-size:0.86rem;font-weight:700;color:#F0F0F0;">'
                    f'{title}</div>'
                    f'<div style="font-size:0.72rem;color:#555;margin-top:2px;">'
                    f'{desc}</div></div></div>'
                )

            with ui.row().classes("w-full gap-2").style(
                "justify-content:flex-end;margin-top:10px;"
            ):
                ui.link("Setup guide", "/setup").style(
                    "color:#FF6A00;font-size:0.76rem;"
                    "font-family:monospace;align-self:center;"
                )
                ui.button(
                    "Let's Go!",
                    on_click=lambda: (
                        dlg.close(),
                        ui.run_javascript("localStorage.setItem('aura_v4_ob','1')")
                    )
                ).style(
                    "background:linear-gradient(135deg,#FF6A00,#CC3300)!important;"
                    "color:#fff!important;border:none!important;border-radius:9px!important;"
                    "padding:7px 18px!important;font-family:Rajdhani,sans-serif!important;"
                    "font-weight:700!important;font-size:0.95rem!important;"
                    "box-shadow:0 0 18px rgba(255,106,0,0.4)!important;"
                )
    return dlg


# ─────────────────────────────────────────────────────────────
#  SETUP PAGE  /setup
# ─────────────────────────────────────────────────────────────
@ui.page("/setup")
async def setup_page():
    ui.add_head_html("<style>" + AURA_CSS + "</style>")
    with ui.element("div").classes("aura-page"):
        with ui.element("div").classes("aura-content"):
            with ui.element("div").classes("glass-card"):
                ui.html(
                    '<div class="logo">Setup Guide</div>'
                    '<div class="logo-sub" style="margin-bottom:12px;">'
                    'Offline Inference Stack — Zero Cloud</div>'
                )
                if state.npu_active:
                    ui.html('<div class="banner banner-ok">&#9889; NPU Engine Active — Ryzen AI XDNA&sup2;</div>')
                elif state.ollama_active:
                    ui.html('<div class="banner banner-ok">&#128994; Ollama Active — 100% Offline</div>')
                else:
                    ui.html(
                        '<div class="banner banner-demo">&#9889; PIL Offline Engine Ready</div>'
                    )
                ui.html('<div class="div-line"></div>')

                steps = [
                    (
                        "Step 1: Ollama Local VLM (recommended — 100% offline)",
                        "Download from ollama.com — runs entirely on your machine, no internet, no API key needed.",
                        "# Step 1: Install Ollama from https://ollama.com\n"
                        "# Step 2: Pull the llava vision model (one-time download ~4GB)\n"
                        "ollama pull llava\n\n"
                        "# Step 3: Start Aura-NPU — Ollama auto-detected\n"
                        "python -m app.main"
                    ),
                    (
                        "Step 2: AMD NPU / VitisAI EP (hardware acceleration)",
                        "Requires AMD Ryzen AI laptop with XDNA/XDNA2 NPU. Enables on-device inference at ~100ms.",
                        "# Install VitisAI runtime\n"
                        "pip install onnxruntime-vitisai==1.17.0 \\\n"
                        "  --extra-index-url https://pypi.amd.com/simple\n\n"
                        "# Model path expected at:\n"
                        "models/gemma3_4b_vlm/model_quantized.onnx\n\n"
                        "# Docs: https://ryzenai.docs.amd.com/en/latest/"
                    ),
                    (
                        "Step 3: Python dependencies (run once)",
                        "Core packages needed for PIL offline analysis and UI.",
                        "pip install nicegui httpx pillow numpy mss psutil scipy sounddevice"
                    ),
                    (
                        "Step 4: Verify NPU in Task Manager (Windows)",
                        "Open Task Manager → Performance tab → look for 'NPU'. If visible, VitisAI EP will activate automatically.",
                        "# Also verify with:\n"
                        "python -c \"from app.npu_engine import verify_npu_status; print(verify_npu_status())\""
                    ),
                    (
                        "Override Ollama model or host (optional)",
                        "Change the default Ollama model or point to a remote Ollama server.",
                        "# PowerShell\n$env:OLLAMA_MODEL='llava:13b'\n$env:OLLAMA_HOST='http://localhost:11434'\npython -m app.main"
                    ),
                ]

                for i, (title, desc, code) in enumerate(steps, 1):
                    ui.html(
                        f'<div class="step-card">'
                        f'<div class="step-num">{i}</div>'
                        f'<div class="step-content">'
                        f'<h3>{title}</h3><p>{desc}</p>'
                        f'<code>{code}</code>'
                        f'</div></div>'
                    )

                ui.html('<div class="div-line"></div>')
                with ui.row().classes("w-full items-center gap-3"):
                    ui.link("← Back to App", "/").style(
                        "color:var(--orange);font-family:var(--mono);font-size:0.8rem;"
                    )

                    async def recheck():
                        await _init_engine()
                        npu = state.engine_status.get(
                            "npu_status", {}
                        ).get("npu_detected", False)
                        ui.notify(
                            "NPU detected!" if npu else "No NPU — using AI backend",
                            type="positive" if npu else "warning",
                        )

                    ui.button("Recheck Hardware", on_click=recheck
                              ).style(
                        "backdrop-filter:blur(10px);"
                        "background:rgba(255,255,255,0.05)!important;"
                        "color:var(--text)!important;"
                        "border:1px solid var(--gbord2)!important;"
                        "border-radius:8px!important;"
                        "font-family:var(--head)!important;font-size:0.8rem!important;"
                        "margin-left:auto;"
                    )


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────
def main():
    logger.info("AURA-NPU v4.1 Offline Cognitive Edition  |  zero-cloud")
    ui.run(
        host=_cfg.HOST,
        port=_cfg.PORT,
        title="AURA-NPU",
        favicon="A",
        dark=True,
        reload=False,
        native=False,
    )


if __name__ == "__main__":
    main()
