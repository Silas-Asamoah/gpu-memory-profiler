"""Real-time memory tracking and monitoring."""

import logging
import os
import socket
import time
import threading
from typing import Dict, List, Optional, Callable, Any, Union
from collections import deque
from dataclasses import dataclass

import torch
import psutil

from .utils import format_bytes, get_gpu_info
from .telemetry import telemetry_event_from_record, telemetry_event_to_dict

logger = logging.getLogger(__name__)


@dataclass
class TrackingEvent:
    """Represents a memory tracking event."""
    timestamp: float
    event_type: str  # 'allocation', 'deallocation', 'peak', 'warning', 'error'
    memory_allocated: int
    memory_reserved: int
    memory_change: int
    device_id: int
    context: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MemoryTracker:
    """Real-time memory tracker with alerts and monitoring."""

    def __init__(self,
                 device: Optional[Union[str, int, torch.device]] = None,
                 sampling_interval: float = 0.1,
                 max_events: int = 10000,
                 enable_alerts: bool = True):
        """
        Initialize the memory tracker.

        Args:
            device: GPU device to track
            sampling_interval: Sampling interval in seconds
            max_events: Maximum number of events to keep in memory
            enable_alerts: Whether to enable memory alerts
        """
        self.device = self._setup_device(device)
        self.sampling_interval = sampling_interval
        self.max_events = max_events
        self.enable_alerts = enable_alerts

        # Tracking state
        self.events: deque[TrackingEvent] = deque(maxlen=max_events)
        self.is_tracking = False
        self._tracking_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Memory thresholds for alerts
        self.thresholds: Dict[str, float] = {
            'memory_warning_percent': 80.0,  # Warn at 80% memory usage
            'memory_critical_percent': 95.0,  # Critical at 95% memory usage
            'memory_leak_threshold': float(100 * 1024 * 1024),  # 100MB growth
            'fragmentation_threshold': 0.3,  # 30% fragmentation
        }

        # Alert callbacks
        self.alert_callbacks: List[Callable[[TrackingEvent], None]] = []

        # Statistics
        self.stats: Dict[str, Any] = {
            'peak_memory': 0,
            'total_allocations': 0,
            'total_deallocations': 0,
            'total_allocation_bytes': 0,
            'total_deallocation_bytes': 0,
            'alert_count': 0,
            'tracking_start_time': None,
            'last_memory_check': 0
        }

        # Get GPU info for memory limits
        self.gpu_info = get_gpu_info(self.device)
        total_memory = self.gpu_info.get('total_memory', 0)
        self.total_memory = int(total_memory) if isinstance(total_memory, (int, float)) else 0

    def _setup_device(self, device: Union[str, int, torch.device, None]) -> torch.device:
        """Setup and validate the device for tracking."""
        resolved_device: torch.device

        if device is None:
            if torch.cuda.is_available():
                resolved_device = torch.device(f"cuda:{torch.cuda.current_device()}")
            else:
                raise RuntimeError(
                    "CUDA is not available, cannot track GPU memory")
        elif isinstance(device, int):
            resolved_device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            resolved_device = torch.device(device)
        else:
            resolved_device = device

        if resolved_device.type != 'cuda':
            raise ValueError(
                "Only CUDA devices are supported for GPU memory tracking")

        return resolved_device

    def start_tracking(self) -> None:
        """Start real-time memory tracking."""
        if self.is_tracking:
            return

        self.is_tracking = True
        self._stop_event.clear()
        self.stats['tracking_start_time'] = time.time()

        self._tracking_thread = threading.Thread(target=self._tracking_loop)
        self._tracking_thread.daemon = True
        self._tracking_thread.start()

        # Add initial event
        self._add_event('start', 0, "Memory tracking started")

    def stop_tracking(self) -> None:
        """Stop real-time memory tracking."""
        if not self.is_tracking:
            return

        self.is_tracking = False
        self._stop_event.set()

        if self._tracking_thread:
            self._tracking_thread.join(timeout=1.0)

        # Add final event
        self._add_event('stop', 0, "Memory tracking stopped")

    def _tracking_loop(self) -> None:
        """Main tracking loop running in background thread."""
        last_allocated = 0
        consecutive_warnings = 0

        while not self._stop_event.wait(self.sampling_interval):
            try:
                # Get current memory usage
                current_allocated = torch.cuda.memory_allocated(self.device)
                current_reserved = torch.cuda.memory_reserved(self.device)

                # Calculate change
                memory_change = current_allocated - last_allocated

                # Update statistics
                self.stats['last_memory_check'] = time.time()
                if current_allocated > self.stats['peak_memory']:
                    self.stats['peak_memory'] = current_allocated
                    self._add_event(
                        'peak', memory_change, f"New peak memory: {format_bytes(current_allocated)}")

                # Track allocations/deallocations
                if memory_change > 0:
                    self.stats['total_allocations'] += 1
                    self.stats['total_allocation_bytes'] += memory_change
                    self._add_event(
                        'allocation', memory_change, f"Memory allocated: {format_bytes(memory_change)}")
                elif memory_change < 0:
                    self.stats['total_deallocations'] += 1
                    self.stats['total_deallocation_bytes'] += abs(
                        memory_change)
                    self._add_event('deallocation', memory_change,
                                    f"Memory freed: {format_bytes(abs(memory_change))}")

                # Check for alerts
                if self.enable_alerts:
                    self._check_alerts(current_allocated,
                                       current_reserved, memory_change)

                last_allocated = current_allocated

            except Exception as e:
                self._add_event('error', 0, f"Tracking error: {str(e)}")
                time.sleep(1.0)  # Back off on errors

    def _add_event(self, event_type: str, memory_change: int, context: str,
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a tracking event."""
        try:
            current_allocated = torch.cuda.memory_allocated(self.device)
            current_reserved = torch.cuda.memory_reserved(self.device)
        except Exception as exc:
            logger.debug("Could not query CUDA memory in _add_event: %s", exc)
            current_allocated = 0
            current_reserved = 0

        event = TrackingEvent(
            timestamp=time.time(),
            event_type=event_type,
            memory_allocated=current_allocated,
            memory_reserved=current_reserved,
            memory_change=memory_change,
            device_id=self.device.index if self.device.index is not None else torch.cuda.current_device(),
            context=context,
            metadata=metadata
        )

        self.events.append(event)

        # Trigger callbacks for alerts
        if event_type in ['warning', 'critical', 'error']:
            self.stats['alert_count'] += 1
            for callback in self.alert_callbacks:
                try:
                    callback(event)
                except Exception as exc:
                    logger.debug("Alert callback error (suppressed): %s", exc)

    def _check_alerts(self, allocated: int, reserved: int, change: int) -> None:
        """Check for memory alerts and warnings."""
        if self.total_memory == 0:
            return

        # Memory usage percentage
        usage_percent = (allocated / self.total_memory) * 100

        # Critical memory usage
        if usage_percent >= self.thresholds['memory_critical_percent']:
            self._add_event('critical', change,
                            f"CRITICAL: Memory usage at {usage_percent:.1f}%",
                            {'usage_percent': usage_percent})

        # Warning memory usage
        elif usage_percent >= self.thresholds['memory_warning_percent']:
            self._add_event('warning', change,
                            f"WARNING: Memory usage at {usage_percent:.1f}%",
                            {'usage_percent': usage_percent})

        # Large allocation warning
        if change > self.thresholds['memory_leak_threshold']:
            self._add_event('warning', change,
                            f"Large allocation detected: {format_bytes(change)}",
                            {'large_allocation': True})

        # Fragmentation warning
        if reserved > 0:
            fragmentation = (reserved - allocated) / reserved
            if fragmentation > self.thresholds['fragmentation_threshold']:
                self._add_event('warning', change,
                                f"High fragmentation: {fragmentation:.1%}",
                                {'fragmentation': fragmentation})

    def add_alert_callback(self, callback: Callable[[TrackingEvent], None]) -> None:
        """Add a callback function to be called on alerts."""
        self.alert_callbacks.append(callback)

    def remove_alert_callback(self, callback: Callable[[TrackingEvent], None]) -> None:
        """Remove an alert callback."""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)

    def get_events(self, event_type: Optional[str] = None,
                   last_n: Optional[int] = None,
                   since: Optional[float] = None) -> List[TrackingEvent]:
        """
        Get tracking events with optional filtering.

        Args:
            event_type: Filter by event type
            last_n: Get last N events
            since: Get events since timestamp

        Returns:
            List of filtered events
        """
        events = list(self.events)

        # Filter by type
        if event_type:
            events = [e for e in events if e.event_type == event_type]

        # Filter by time
        if since:
            events = [e for e in events if e.timestamp >= since]

        # Limit results
        if last_n:
            events = events[-last_n:]

        return events

    def get_memory_timeline(self, interval: float = 1.0) -> Dict[str, List]:
        """
        Get memory usage timeline with specified interval.

        Args:
            interval: Time interval in seconds for aggregation

        Returns:
            Dictionary with timeline data
        """
        if not self.events:
            return {'timestamps': [], 'allocated': [], 'reserved': []}

        # Group events by time intervals
        start_time = self.events[0].timestamp
        end_time = self.events[-1].timestamp

        timestamps = []
        allocated_values = []
        reserved_values = []

        current_time = start_time
        while current_time <= end_time:
            # Find events in this interval
            interval_events = [
                e for e in self.events
                if current_time <= e.timestamp < current_time + interval
            ]

            if interval_events:
                # Use the last event in the interval
                last_event = interval_events[-1]
                timestamps.append(current_time)
                allocated_values.append(last_event.memory_allocated)
                reserved_values.append(last_event.memory_reserved)

            current_time += interval

        return {
            'timestamps': timestamps,
            'allocated': allocated_values,
            'reserved': reserved_values
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tracking statistics."""
        current_stats = self.stats.copy()

        if self.events:
            # Calculate additional statistics
            recent_events = [
                e for e in self.events if e.timestamp > time.time() - 3600]  # Last hour

            current_stats.update({
                'total_events': len(self.events),
                'events_last_hour': len(recent_events),
                'current_memory_allocated': torch.cuda.memory_allocated(self.device) if torch.cuda.is_available() else 0,
                'current_memory_reserved': torch.cuda.memory_reserved(self.device) if torch.cuda.is_available() else 0,
                'memory_utilization_percent': (torch.cuda.memory_allocated(self.device) / self.total_memory * 100) if self.total_memory > 0 else 0,
                'average_allocation_size': self.stats['total_allocation_bytes'] / max(self.stats['total_allocations'], 1),
                'average_deallocation_size': self.stats['total_deallocation_bytes'] / max(self.stats['total_deallocations'], 1),
            })

            # Time-based statistics
            if self.stats['tracking_start_time']:
                tracking_duration = time.time(
                ) - self.stats['tracking_start_time']
                current_stats.update({
                    'tracking_duration_seconds': tracking_duration,
                    'allocations_per_second': self.stats['total_allocations'] / max(tracking_duration, 1),
                    'bytes_allocated_per_second': self.stats['total_allocation_bytes'] / max(tracking_duration, 1)
                })

        return current_stats

    def export_events(self, filename: str, format: str = 'csv') -> None:
        """
        Export tracking events to file.

        Args:
            filename: Output filename
            format: Export format ('csv' or 'json')
        """
        import pandas as pd
        import json

        if not self.events:
            return

        host = socket.gethostname()
        pid = os.getpid()
        sampling_interval_ms = int(round(self.sampling_interval * 1000))

        # Convert events to canonical telemetry records.
        records = []
        for event in self.events:
            legacy = {
                'timestamp': event.timestamp,
                'event_type': event.event_type,
                'memory_allocated': event.memory_allocated,
                'memory_reserved': event.memory_reserved,
                'memory_change': event.memory_change,
                'device_id': event.device_id,
                'context': event.context,
                'metadata': event.metadata or {},
                'total_memory': self.total_memory or None,
                'pid': pid,
                'host': host,
                'collector': 'gpumemprof.cuda_tracker',
                'sampling_interval_ms': sampling_interval_ms,
            }
            telemetry_event = telemetry_event_from_record(
                legacy,
                default_collector='gpumemprof.cuda_tracker',
                default_sampling_interval_ms=sampling_interval_ms,
            )
            records.append(telemetry_event_to_dict(telemetry_event))

        if format == 'csv':
            df = pd.DataFrame(records)
            df.to_csv(filename, index=False)
        elif format == 'json':
            with open(filename, 'w') as f:
                json.dump(records, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def clear_events(self) -> None:
        """Clear all tracking events."""
        self.events.clear()

        # Reset statistics
        self.stats.update({
            'peak_memory': 0,
            'total_allocations': 0,
            'total_deallocations': 0,
            'total_allocation_bytes': 0,
            'total_deallocation_bytes': 0,
            'alert_count': 0
        })

    def set_threshold(self, threshold_name: str, value: Union[int, float]) -> None:
        """
        Set alert threshold.

        Args:
            threshold_name: Name of the threshold
            value: Threshold value
        """
        if threshold_name in self.thresholds:
            self.thresholds[threshold_name] = value
        else:
            raise ValueError(f"Unknown threshold: {threshold_name}")

    def get_alerts(self, last_n: Optional[int] = None) -> List[TrackingEvent]:
        """Get all alert events (warnings, critical, errors)."""
        alert_types = ['warning', 'critical', 'error']
        alerts = [e for e in self.events if e.event_type in alert_types]

        if last_n:
            alerts = alerts[-last_n:]

        return alerts

    def __enter__(self) -> "MemoryTracker":
        """Context manager entry."""
        self.start_tracking()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.stop_tracking()


class MemoryWatchdog:
    """Memory watchdog for automated memory management."""

    def __init__(self,
                 tracker: MemoryTracker,
                 auto_cleanup: bool = True,
                 cleanup_threshold: float = 0.9,
                 aggressive_cleanup_threshold: float = 0.95):
        """
        Initialize memory watchdog.

        Args:
            tracker: MemoryTracker instance to monitor
            auto_cleanup: Whether to automatically clean up memory
            cleanup_threshold: Memory usage threshold to trigger cleanup
            aggressive_cleanup_threshold: Threshold for aggressive cleanup
        """
        self.tracker = tracker
        self.auto_cleanup = auto_cleanup
        self.cleanup_threshold = cleanup_threshold
        self.aggressive_cleanup_threshold = aggressive_cleanup_threshold

        # Register alert callback
        self.tracker.add_alert_callback(self._handle_alert)

        self.cleanup_count = 0
        self.last_cleanup_time = 0.0
        self.min_cleanup_interval = 30.0  # Minimum 30 seconds between cleanups

    def _handle_alert(self, event: TrackingEvent) -> None:
        """Handle memory alerts."""
        if not self.auto_cleanup:
            return

        current_time = time.time()

        # Avoid too frequent cleanups
        if current_time - self.last_cleanup_time < self.min_cleanup_interval:
            return

        # Check if cleanup is needed
        if event.event_type == 'critical' or (
            event.event_type == 'warning' and
            event.metadata and
            event.metadata.get(
                'usage_percent', 0) >= self.cleanup_threshold * 100
        ):
            self._perform_cleanup(aggressive=event.event_type == 'critical')
            self.last_cleanup_time = current_time

    def _perform_cleanup(self, aggressive: bool = False) -> None:
        """Perform memory cleanup."""
        self.cleanup_count += 1

        try:
            # Basic cleanup
            torch.cuda.empty_cache()

            if aggressive:
                # More aggressive cleanup
                import gc
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            # Log cleanup event
            cleanup_type = "aggressive" if aggressive else "standard"
            self.tracker._add_event(
                'cleanup', 0, f"Performed {cleanup_type} memory cleanup")

        except Exception as e:
            self.tracker._add_event('error', 0, f"Cleanup failed: {str(e)}")

    def force_cleanup(self, aggressive: bool = False) -> None:
        """Force immediate memory cleanup."""
        self._perform_cleanup(aggressive)

    def get_cleanup_stats(self) -> Dict[str, Any]:
        """Get cleanup statistics."""
        return {
            'cleanup_count': self.cleanup_count,
            'last_cleanup_time': self.last_cleanup_time,
            'auto_cleanup_enabled': self.auto_cleanup,
            'cleanup_threshold': self.cleanup_threshold,
            'aggressive_cleanup_threshold': self.aggressive_cleanup_threshold
        }
