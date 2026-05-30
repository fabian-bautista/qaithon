"""Experimentos completos de Qaithon, capturando TODOS los resultados como
documentación (docs/RESULTS.md). TODO en SIMULADOR (no hardware físico).

Secciones:
  A. TinyStories-1M (modelo real preentrenado) por tecnología.
  B. Inferencia de lo más grande posible por tecnología.
  C. Entrenamiento de lo más grande posible por tecnología (capas diferenciables).
  D. Salida de cada método (documentación de la API).
"""

import warnings
warnings.filterwarnings("ignore")
import io
import contextlib
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import qaithon

OUT = Path("/Users/fabianbautista/quantum-photonic-llm/docs/RESULTS.md")
B = []
def w(s=""): B.append(s);
def flush():
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(B))

def section(t): w(f"\n## {t}\n")
def block(title, body, kind=""):
    w(f"### {title}")
    if kind: w(f"*{kind}*")
    w("```"); w(str(body)[:1500]); w("```"); w("")
    flush()  # escribir incremental por si algo falla

w("# Qaithon — Resultados de experimentos (documentación)")
w(f"\n_Generado {datetime.now():%Y-%m-%d %H:%M} · TODO en SIMULADOR, NO hardware físico._\n")
w("> Cómputo (matmul) = algoritmo cuántico/fotónico real. Inferencia usa kernel de "
  "dilatación; entrenamiento usa capas diferenciables (MerLin/PennyLane).\n")

# ════════════════ A. TinyStories por tecnología ════════════════
section("A. TinyStories-1M (real preentrenado, dim 64) por tecnología")
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
prompt = "Once upon a time, there was a little"
ids = tok(prompt, return_tensors="pt").input_ids
def skip_big(name, mod):
    wt = getattr(mod, "weight", None)
    return wt is not None and wt.dim() >= 2 and max(wt.shape[0], wt.shape[1]) > 2048

for tech, backend in [("CUÁNTICO", "pennylane.sim"), ("FOTÓNICO", "quandela.sim")]:
    try:
        m = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M").eval()
        m, rep = qaithon.compile(m, backends=(backend,), skip=skip_big,
                                 min_in_features=1, min_out_features=1, return_report=True)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = m.generate(ids, max_new_tokens=25, do_sample=False)
        dt = time.perf_counter() - t0
        txt = tok.decode(out[0], skip_special_tokens=True)
        block(f"TinyStories en {tech} ({backend})",
              f"capas genuinas: {len(rep.decisions)}\ntiempo: {dt:.1f}s\nGENERADO: {txt}",
              f"GENUINO {tech.lower()} · inferencia")
    except Exception as e:
        block(f"TinyStories en {tech} ({backend})",
              f"NO SOPORTADO: {type(e).__name__}: {e}",
              f"{tech.lower()} · bloqueado por límite de hardware")

# ════════════════ B. Inferencia: lo más grande por tecnología ════════════════
section("B. Inferencia genuina — modelo más grande viable por tecnología")
def infer_toy(dim, backend, tech):
    try:
        m = qaithon.models.create_toy_transformer(dim=dim, n_layers=2)
        m, rep = qaithon.compile(m, backends=(backend,), return_report=True)
        t0=time.perf_counter()
        with torch.no_grad():
            o = m(torch.randint(0, 50, (1, 6)))
        dt=time.perf_counter()-t0
        return f"dim={dim}: {len(rep.decisions)} capas genuinas, forward {tuple((o.logits if hasattr(o,'logits') else o).shape)} en {dt:.2f}s"
    except Exception as e:
        return f"dim={dim}: BLOQUEADO ({type(e).__name__})"
# fotónico: barrer hasta donde aguante (≤128 por capa)
res_ph = [infer_toy(d, "quandela.sim", "fot") for d in [32, 40]]
block("Fotónico (quandela.sim) — inferencia toy", "\n".join(res_ph), "GENUINO fotónico")
# cuántico: puede más
res_q = [infer_toy(d, "pennylane.sim", "cua") for d in [64, 128]]
block("Cuántico (pennylane.sim) — inferencia toy", "\n".join(res_q), "GENUINO cuántico")

# ════════════════ C. Entrenamiento: lo más grande por tecnología ════════════════
section("C. Entrenamiento genuino — capas diferenciables, más grande viable")
from qaithon.layers.photonic_layer import PhotonicLayer
from qaithon.layers.quantum_layer import QuantumLayer
def train_layer(make, in_dim, tag):
    try:
        torch.manual_seed(0)
        net = make()
        X = torch.rand(96, in_dim); Y = (X[:,0] > X[:,1]).long()
        opt = torch.optim.Adam(net.parameters(), lr=0.05); lf = nn.CrossEntropyLoss(); l0=None
        t0=time.perf_counter()
        for _ in range(100):
            opt.zero_grad(); out=net(X); loss=lf(out,Y)
            if l0 is None: l0=loss.item()
            loss.backward(); opt.step()
        acc=(net(X).argmax(1)==Y).float().mean().item()
        return f"{tag}: loss {l0:.3f}->{loss.item():.3f}, acc {acc:.2f}, {time.perf_counter()-t0:.1f}s"
    except Exception as e:
        return f"{tag}: FALLA ({type(e).__name__}: {e})"
block("Fotónico — entrenar capa fotónica (PhotonicLayer)",
      train_layer(lambda: nn.Sequential(PhotonicLayer(8, 2, photons=2)), 8, "PhotonicLayer in=8"),
      "GENUINO fotónico · entrenable")
block("Cuántico — entrenar capa cuántica (QuantumLayer)",
      train_layer(lambda: nn.Sequential(QuantumLayer(16, 2, var_layers=2)), 16, "QuantumLayer in=16"),
      "GENUINO cuántico · entrenable")

# ════════════════ D. Salida de cada método (API) ════════════════
section("D. Salida real de cada método (utilidades + cómputo)")
def doc(title, fn, kind):
    cap = io.StringIO()
    try:
        with contextlib.redirect_stdout(cap):
            r = fn()
        body = (cap.getvalue().strip() + "\n" + (str(r) if r is not None else "")).strip() or "(ok)"
    except Exception as e:
        body = f"ERROR: {type(e).__name__}: {e}"
    block(title, body, kind)

doc("estimate_qubits_from_config(GPT-2)",
    lambda: qaithon.estimate_qubits_from_config(hidden_size=768,n_layers=12,n_heads=12,model_class="GPT-2").pretty(), "utilidad")
doc("validate_for_hardware(toy, IBM Heron)",
    lambda: qaithon.validate_for_hardware(qaithon.models.create_toy_transformer(dim=64,n_layers=1),target="IBM Heron").pretty(), "utilidad")
doc("measure_actual_circuit(16,16)",
    lambda: f"q={qaithon.measure_actual_circuit(16,16).n_qubits} gates={qaithon.measure_actual_circuit(16,16).n_gates}", "utilidad (circuito Qiskit real)")
doc("find_hardware(Belenos)", lambda: str(qaithon.find_hardware("Belenos")), "utilidad")
from qaithon import hardware_limits as HL
doc("hardware_limits.describe_limits(Belenos)", lambda: HL.describe_limits("Quandela Belenos"), "utilidad")
doc("hardware_limits.describe_limits(IBM Heron)", lambda: HL.describe_limits("IBM Heron"), "utilidad")
doc("explain('fidelity', 0.987)", lambda: qaithon.explain("fidelity", value=0.987), "utilidad")
doc("list_terms()", lambda: ", ".join(qaithon.list_terms()), "utilidad")
doc("pricing.estimate_cost_usd(quera, 1000 shots)", lambda: f"${qaithon.pricing.estimate_cost_usd('aws.braket.quera', n_shots=1000):.2f}", "utilidad")
doc("benchmarks.compare_backends(8,8)",
    lambda: qaithon.benchmarks.compare_backends(8,8,exclude=("aws.braket.quera","aws.braket.ionq","ibm.heron","quandela.belenos")).pretty(), "comparación")

flush()
print(f"LISTO. {len(B)} líneas en {OUT}")
