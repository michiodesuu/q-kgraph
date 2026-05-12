"""
utils/logger.py — Rich Terminal Logging  [V1]
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Any
import torch
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class RichLogger:
    def __init__(self, name: str, log_file: Optional[Path] = None):
        self.name    = name
        self.console = Console() if RICH_AVAILABLE else None
        self._setup_file_logger(log_file)

    def _setup_file_logger(self, log_file: Optional[Path]):
        self._logger = logging.getLogger(self.name)
        self._logger.setLevel(logging.INFO)
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            self._logger.addHandler(fh)

    def info(self, msg: str): self._log("info", msg)
    def warning(self, msg: str): self._log("warning", f"[yellow]{msg}[/yellow]" if RICH_AVAILABLE else msg)
    def error(self, msg: str):   self._log("error",   f"[red]{msg}[/red]"     if RICH_AVAILABLE else msg)

    def _log(self, level: str, msg: str):
        if self.console and RICH_AVAILABLE:
            self.console.print(f"[dim]{self.name}[/dim] {msg}")
        else:
            print(f"[{self.name}] {msg}")
        getattr(self._logger, level)(msg.replace("[", "").replace("]", ""))

    def print_banner(self, title: str, color: str = "blue"):
        if self.console and RICH_AVAILABLE:
            self.console.print(Panel(f"[bold {color}]{title}[/bold {color}]", border_style=color))
        else:
            print(f"\n{'='*60}\n{title}\n{'='*60}")

    def print_metrics(self, metrics: dict[str, Any], title: str = "Metrics"):
        if self.console and RICH_AVAILABLE:
            t = Table(show_header=True, header_style="bold cyan", title=title)
            t.add_column("Metric"); t.add_column("Value", justify="right")
            for k, v in metrics.items():
                val = f"{v:.4f}" if isinstance(v, float) else str(v)
                t.add_row(k, val)
            self.console.print(t)
        else:
            print(f"\n{title}:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    def print_comparison_table(self, results: dict[str, dict[str, float]], title: str = "Results"):
        """Render a comparison table for paper Table 2."""
        if not results:
            return
        if self.console and RICH_AVAILABLE:
            t = Table(show_header=True, header_style="bold magenta", title=title)
            t.add_column("Model")
            first_metrics = next(iter(results.values()))
            for m in first_metrics:
                t.add_column(m, justify="right")
            for model_name, metrics in results.items():
                row = [model_name] + [f"{v:.4f}" for v in metrics.values()]
                t.add_row(*row)
            self.console.print(t)
        else:
            print(f"\n{title}:")
            for model_name, metrics in results.items():
                metric_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(f"  {model_name}: {metric_str}")
