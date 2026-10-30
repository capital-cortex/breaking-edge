from dataclasses import dataclass

@dataclass
class KlinerArgs():
    interval_symbols : dict[str, list[str]]
    history_seconds  : float
    timer_seconds    : float
    heartbeat_url    : str | None = None
    log_path         : str | None = None
