"""Training and benchmark suites for AuditDDI."""

def __getattr__(name: str):
    if name == "run_cold_target_study":
        from src.training.benchmark_cold_target import run_cold_target_study
        return run_cold_target_study
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
