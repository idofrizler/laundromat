"""
Timing utilities for profiling the inference pipeline.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import threading

@dataclass
class TimingResult:
    """Result of a timing measurement."""
    name: str
    duration_ms: float
    details: Optional[str] = None

@dataclass
class TimingContext:
    """Context for collecting timing measurements."""
    stages: List[TimingResult] = field(default_factory=list)
    _start_time: Optional[float] = field(default=None, repr=False)
    
    def start(self):
        """Start the overall timing."""
        self._start_time = time.perf_counter()
    
    def add(self, name: str, duration_ms: float, details: Optional[str] = None):
        """Add a timing measurement."""
        self.stages.append(TimingResult(name, duration_ms, details))
    
    @property
    def total_ms(self) -> float:
        """Get total elapsed time since start()."""
        if self._start_time is None:
            return sum(s.duration_ms for s in self.stages)
        return (time.perf_counter() - self._start_time) * 1000
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization."""
        result = {}
        for stage in self.stages:
            key = stage.name.lower().replace(' ', '_')
            result[key] = round(stage.duration_ms, 2)
        result['total_ms'] = round(self.total_ms, 2)
        return result
    
    def print_summary(self, title: str = "Timing Breakdown"):
        """Print a formatted timing summary."""
        print(f"\n[{title}]")
        for stage in self.stages:
            detail_str = f" ({stage.details})" if stage.details else ""
            print(f"  {stage.name:.<25} {stage.duration_ms:>8.2f} ms{detail_str}")
        print(f"  {'-' * 40}")
        print(f"  {'TOTAL':.<25} {self.total_ms:>8.2f} ms\n")

@contextmanager
def timed_section(timing_ctx: Optional[TimingContext], name: str, details: Optional[str] = None):
    """
    Context manager for timing a section of code.
    
    Args:
        timing_ctx: TimingContext to add measurement to (if None, timing is skipped)
        name: Name of the section being timed
        details: Optional details (e.g., "12 items")
        
    Usage:
        with timed_section(timing, "Feature extraction", f"{n} socks"):
            # code to time
    """
    if timing_ctx is None:
        yield
        return
    
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        timing_ctx.add(name, duration_ms, details)

class Timer:
    """Simple timer for inline measurements."""
    
    def __init__(self):
        self._start = None
        self._elapsed_ms = 0.0
    
    def start(self):
        """Start the timer."""
        self._start = time.perf_counter()
        return self
    
    def stop(self) -> float:
        """Stop the timer and return elapsed milliseconds."""
        if self._start is not None:
            self._elapsed_ms = (time.perf_counter() - self._start) * 1000
            self._start = None
        return self._elapsed_ms
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed milliseconds (call stop() first)."""
        return self._elapsed_ms

# Thread-local storage for current timing context
_timing_local = threading.local()

def get_current_timing() -> Optional[TimingContext]:
    """Get the current thread's timing context."""
    return getattr(_timing_local, 'timing', None)

def set_current_timing(timing: Optional[TimingContext]):
    """Set the current thread's timing context."""
    _timing_local.timing = timing

@contextmanager
def profiling_enabled():
    """
    Context manager to enable profiling for the current thread.
    
    Usage:
        with profiling_enabled() as timing:
            # ... run inference ...
            timing.print_summary()
    """
    timing = TimingContext()
    timing.start()
    old_timing = get_current_timing()
    set_current_timing(timing)
    try:
        yield timing
    finally:
        set_current_timing(old_timing)
