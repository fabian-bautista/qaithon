"""Command-line interface — ``qaithon`` entry point.

A tiny ``argparse``-based CLI for the most common workflows:

* ``qaithon list-backends`` — what backends are registered and available.
* ``qaithon inspect <model_id>`` — load a HuggingFace model, analyze it,
  print the replacement plan without modifying anything.
* ``qaithon compile <model_id> --backend mock --out path/`` — compile a
  model, save the result to ``path/`` (safetensors + config), print the
  CompileReport.

Designed so a developer can sanity-check Qaithon against any HF model in
two lines from their terminal, without writing Python::

    pip install qaithon[huggingface]
    qaithon inspect gpt2

The CLI never invents output paths or downloads outside the HF cache —
all destructive actions require explicit flags.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from qaithon._logging import enable_default_logging, get_logger

if TYPE_CHECKING:
    pass

__all__ = ["main"]

logger = get_logger(__name__)


def _cmd_list_backends(_args: argparse.Namespace) -> int:
    from qaithon.backends import get_backend, list_backends

    rows = []
    for name in list_backends():
        backend = get_backend(name)
        p = backend.profile
        rows.append(
            (
                name,
                p.kind,
                f"{p.energy_pj_per_mac:.3f}",
                f"{p.latency_us_per_op:.1f}",
                str(p.supports_autograd),
                str(backend.is_available()),
            )
        )

    header = ("name", "kind", "pJ/MAC", "µs/op", "autograd", "available")
    widths = [max(len(r[i]) for r in (header, *rows)) for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))
    return 0


def _load_hf_model(model_id: str) -> object:
    try:
        from transformers import AutoModel
    except ImportError as exc:
        print(
            "ERROR: transformers is required. Install with "
            "`pip install qaithon[huggingface]`.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    return AutoModel.from_pretrained(model_id)


def _cmd_inspect(args: argparse.Namespace) -> int:
    from qaithon.ir import analyze_model

    model = _load_hf_model(args.model_id)
    plan = analyze_model(model, strict_already_quantized=not args.allow_quantized)
    if args.json:
        print(
            json.dumps(
                {
                    "model_id": args.model_id,
                    "model_class": type(model).__name__,
                    "n_parameters": sum(p.numel() for p in model.parameters()),
                    "replaceable": [
                        {
                            "name": m.name,
                            "in_features": m.in_features,
                            "out_features": m.out_features,
                        }
                        for m in plan.matches
                    ],
                    "skipped": [{"name": n, "reason": r} for n, r in plan.skipped],
                },
                indent=2,
            )
        )
    else:
        print(plan.summary())
    return 0


def _cmd_compile(args: argparse.Namespace) -> int:
    import qaithon

    model = _load_hf_model(args.model_id)

    backends = tuple(args.backend) if args.backend else None
    qaithon.compile(
        model,
        optimize_for=args.optimize_for,
        backends=backends,
        strict=not args.allow_quantized,
    )

    report = model.qaithon_report  # type: ignore[attr-defined]
    print(report.pretty())

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            from safetensors.torch import save_file
        except ImportError:
            print(
                "ERROR: safetensors is required to save. Install with "
                "`pip install qaithon[huggingface]`.",
                file=sys.stderr,
            )
            return 2
        save_file(model.state_dict(), str(out_dir / "model.safetensors"))
        (out_dir / "qaithon_report.json").write_text(
            json.dumps(
                {
                    "model_id": args.model_id,
                    "report": {
                        "model_class": report.model_class,
                        "n_parameters": report.n_parameters,
                        "n_replaced": report.n_replaced,
                        "n_skipped": report.n_skipped,
                        "optimize_for": report.optimize_for,
                        "backends_used": list(report.backends_used),
                        "baseline_energy_pj": report.baseline_energy_pj,
                        "compiled_energy_pj": report.compiled_energy_pj,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote {out_dir / 'model.safetensors'} and qaithon_report.json")
    return 0


def _cmd_trace_inspect(args: argparse.Namespace) -> int:
    """Pretty-print a JSON trace previously produced by qaithon.tracing."""
    path = Path(args.path)
    if not path.exists():
        print(f"ERROR: trace file {path} does not exist.", file=sys.stderr)
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    events = data.get("events", [])

    print(f"Trace: {path}")
    print(f"  Events:        {summary.get('n_events', len(events))}")
    print(f"  Total latency: {summary.get('total_latency_us', 0):.0f} µs")
    print(f"  Total energy:  {summary.get('total_energy_pj', 0):.0f} pJ")
    print()

    if args.top:
        top_n = int(args.top)
        sorted_events = sorted(events, key=lambda e: e.get("latency_us", 0), reverse=True)[
            :top_n
        ]
        print(f"Top {top_n} events by latency:")
        for ev in sorted_events:
            print(
                f"  {ev.get('backend', '?'):20s}  "
                f"shape={tuple(ev.get('input_shape', []))}  "
                f"lat={ev.get('latency_us', 0):.1f} µs  "
                f"E={ev.get('estimated_energy_pj', 0):.0f} pJ"
            )
    return 0


def _cmd_plugins_list(_args: argparse.Namespace) -> int:
    """List third-party Qaithon plugins discovered via entry_points."""
    from qaithon import plugins

    discovered = plugins.list_plugins()
    if not discovered:
        # If list_plugins is empty because discover() wasn't called yet,
        # run discovery now.
        discovered = plugins.discover()

    if not discovered:
        print("No qaithon.backends plugins discovered.")
        print("(Third parties register their backends via the 'qaithon.backends' entry_points group.)")
        return 0

    print(f"{'name':24s}  {'distribution':24s}  {'status':10s}  details")
    print("-" * 80)
    for p in discovered:
        status = "ok" if p.loaded else "FAILED"
        details = "" if p.loaded else (p.error or "?")
        print(f"  {p.name:22s}  {p.distribution:22s}  {status:10s}  {details}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    """Run the cross-backend benchmark from the CLI."""
    import qaithon

    result = qaithon.benchmarks.compare_backends(
        in_features=args.in_features,
        out_features=args.out_features,
        exclude=tuple(args.exclude) if args.exclude else None,
        repeats=3,
    )
    print(result.pretty(explain=args.explain))
    return 0


def _cmd_glossary(args: argparse.Namespace) -> int:
    """Show one term or the whole glossary."""
    import qaithon

    if args.term:
        try:
            value = float(args.value) if args.value else None
        except ValueError:
            print(f"ERROR: --value must be a number, got {args.value!r}", file=sys.stderr)
            return 2
        try:
            print(qaithon.explain(args.term, value=value))
        except KeyError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    else:
        print("Available terms:")
        for term in qaithon.list_terms():
            entry = qaithon.glossary(term)
            print(f"  {term:30s} {entry.short[:80]}")
        print("\nUse `qaithon glossary <term>` for a detailed entry.")
    return 0


def _cmd_estimate(args: argparse.Namespace) -> int:
    """Estimate the qubit budget required to run a HuggingFace model on QPUs."""
    import qaithon

    model = _load_hf_model(args.model_id)
    report = qaithon.estimate_qubits(model)

    if args.json:
        import json

        print(
            json.dumps(
                {
                    "model": args.model_id,
                    "model_class": report.model_class,
                    "n_layers_analyzed": report.n_layers_analyzed,
                    "circuits_per_forward_pass": report.total_circuit_count,
                    "max_qubits_amplitude_encoding": report.max_qubits_amplitude,
                    "max_qubits_block_encoding": report.max_qubits_block,
                    "max_circuit_depth": report.max_circuit_depth,
                    "hardware_compatibility": [
                        {"name": s.name, "fits": fits, "reason": reason}
                        for s, fits, reason in report.hardware_compatibility()
                    ],
                },
                indent=2,
            )
        )
    else:
        print(report.pretty())
    return 0


def _cmd_doctor(_args: argparse.Namespace) -> int:
    """Diagnose the local environment for Qaithon developers."""
    import platform
    import sys

    print("== Qaithon doctor ==")
    print(f"  Python:        {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"  Platform:      {platform.system()} {platform.release()} {platform.machine()}")

    try:
        import torch

        print(f"  PyTorch:       {torch.__version__}")
        print(f"    CUDA:        {'available' if torch.cuda.is_available() else 'unavailable'}")
        print(f"    MPS:         {'available' if torch.backends.mps.is_available() else 'unavailable'}")
    except ImportError:
        print("  PyTorch:       NOT INSTALLED — install with `pip install qaithon`.")
        return 1

    import qaithon
    print(f"  Qaithon:       {qaithon.__version__}")
    print()

    print("== Backends ==")
    from qaithon.backends import get_backend, list_backends

    optional_hints = {
        "pennylane.sim": "pip install qaithon[pennylane]",
        "ibm.quantum": "pip install qaithon[pennylane]",
        "aws.braket": "pip install amazon-braket-sdk",
        "quandela.sim": "pip install qaithon[quandela]",
        "deepquantum": "pip install qaithon[deepquantum]",
    }

    for name in list_backends():
        backend = get_backend(name)
        marker = "✓" if backend.is_available() else "✗"
        line = f"  {marker} {name:18s} ({backend.profile.kind})"
        if not backend.is_available() and name in optional_hints:
            line += f"  →  {optional_hints[name]}"
        print(line)
    print()

    print("== Recommendations ==")
    have_real = any(
        get_backend(n).is_available()
        and get_backend(n).profile.kind != "mock"
        for n in list_backends()
    )
    if have_real:
        print("  All set. You have at least one non-mock backend available.")
    else:
        print(
            "  Only the mock backend is available. Install at least one of "
            "the [pennylane] / [quandela] / [deepquantum] extras to unlock "
            "the photonic / quantum benefits."
        )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qaithon",
        description=(
            "Compile any HuggingFace LLM to run on photonic / quantum backends."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable INFO-level logging from the qaithon package.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_list = subparsers.add_parser(
        "list-backends",
        help="Show all registered backends and their availability.",
    )
    p_list.set_defaults(func=_cmd_list_backends)

    p_inspect = subparsers.add_parser(
        "inspect",
        help="Load a HuggingFace model and print Qaithon's replacement plan.",
    )
    p_inspect.add_argument("model_id", help="HuggingFace model id (e.g. 'gpt2').")
    p_inspect.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    p_inspect.add_argument(
        "--allow-quantized",
        action="store_true",
        help="Continue even if the model is already quantized.",
    )
    p_inspect.set_defaults(func=_cmd_inspect)

    p_compile = subparsers.add_parser(
        "compile",
        help="Apply qaithon.compile() to a HF model and optionally save it.",
    )
    p_compile.add_argument("model_id", help="HuggingFace model id (e.g. 'gpt2').")
    p_compile.add_argument(
        "--backend",
        "-b",
        action="append",
        help="Restrict to this backend (repeatable).",
    )
    p_compile.add_argument(
        "--optimize-for",
        choices=("balanced", "speed", "energy"),
        default="balanced",
    )
    p_compile.add_argument(
        "--allow-quantized",
        action="store_true",
        help="Continue even if the model is already quantized.",
    )
    p_compile.add_argument(
        "--out",
        type=Path,
        help="Directory to save the compiled model and qaithon report into.",
    )
    p_compile.set_defaults(func=_cmd_compile)

    p_doctor = subparsers.add_parser(
        "doctor",
        help="Diagnose the local environment for Qaithon usage.",
    )
    p_doctor.set_defaults(func=_cmd_doctor)

    p_estimate = subparsers.add_parser(
        "estimate",
        help="Estimate qubit budget required to run a model on a real QPU.",
    )
    p_estimate.add_argument("model_id", help="HuggingFace model id (e.g. 'gpt2').")
    p_estimate.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    p_estimate.set_defaults(func=_cmd_estimate)

    p_glossary = subparsers.add_parser(
        "glossary",
        help="Look up quantum-computing terms in plain AI-developer language.",
    )
    p_glossary.add_argument(
        "term",
        nargs="?",
        help="Term to explain. Leave empty to list all available terms.",
    )
    p_glossary.add_argument(
        "--value",
        help="Optional numeric value of the metric — explanation will be contextualized.",
    )
    p_glossary.set_defaults(func=_cmd_glossary)

    p_bench = subparsers.add_parser(
        "benchmark",
        help="Run the same matmul on every available backend and report metrics.",
    )
    p_bench.add_argument(
        "--in-features", type=int, default=16, help="Input matmul dimension. Default 16.",
    )
    p_bench.add_argument(
        "--out-features", type=int, default=16, help="Output dimension. Default 16.",
    )
    p_bench.add_argument(
        "--exclude",
        nargs="*",
        default=["aws.braket.quera", "aws.braket.ionq", "ibm.heron", "quandela.belenos"],
        help="Backend names to skip (defaults exclude billable cloud QPUs).",
    )
    p_bench.add_argument("--explain", action="store_true", help="Inline explanations.")
    p_bench.set_defaults(func=_cmd_benchmark)

    p_trace = subparsers.add_parser(
        "trace",
        help="Operate on JSON traces produced by qaithon.tracing.",
    )
    trace_sub = p_trace.add_subparsers(dest="trace_command", required=True)
    p_trace_inspect = trace_sub.add_parser(
        "inspect",
        help="Pretty-print a JSON trace and show the top N events by latency.",
    )
    p_trace_inspect.add_argument("path", help="Path to a JSON trace file.")
    p_trace_inspect.add_argument(
        "--top",
        default="10",
        help="Show the N slowest events in detail. Default: 10.",
    )
    p_trace_inspect.set_defaults(func=_cmd_trace_inspect)

    p_plugins = subparsers.add_parser(
        "plugins",
        help="Operate on Qaithon plugins (third-party backends).",
    )
    plugins_sub = p_plugins.add_subparsers(dest="plugins_command", required=True)
    p_plugins_list = plugins_sub.add_parser(
        "list", help="List plugins discovered via entry_points."
    )
    p_plugins_list.set_defaults(func=_cmd_plugins_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.verbose:
        enable_default_logging()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
