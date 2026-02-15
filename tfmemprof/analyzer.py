"""TensorFlow Memory Analysis"""

import logging
import statistics
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, cast

import numpy as np
from scipy import stats as scipy_stats

from gpumemprof.telemetry import TelemetryEventV2
from .utils import format_memory


@dataclass
class GapFinding:
    """A classified finding from hidden-memory gap analysis."""
    classification: str  # 'transient_spike' | 'persistent_drift' | 'fragmentation_like'
    severity: str  # 'info', 'warning', 'critical'
    confidence: float  # 0.0 to 1.0
    evidence: Dict[str, Any]
    description: str
    remediation: List[str]


class MemoryAnalyzer:
    """Advanced TensorFlow memory analysis and optimization."""

    def __init__(self, sensitivity: float = 0.05) -> None:
        self.sensitivity = sensitivity

        # Hidden-memory gap analysis thresholds
        self.gap_thresholds = {
            'gap_ratio_threshold': 0.05,  # 5% of device total = significant gap
            'gap_spike_zscore': 2.0,  # z-score for transient spike detection
            'gap_drift_r_squared': 0.6,  # R-squared for persistent drift
            'gap_fragmentation_ratio': 0.3,  # reserved-allocated / reserved
        }

    def detect_memory_leaks(self, tracking_results: Any) -> List[Dict[str, Any]]:
        """Detect potential memory leaks using statistical analysis."""
        if not hasattr(tracking_results, 'memory_usage') or len(tracking_results.memory_usage) < 10:
            return []

        memory_data = tracking_results.memory_usage
        timestamps = tracking_results.timestamps

        leaks: List[Dict[str, Any]] = []

        # Check for monotonic increase
        increasing_samples = sum(1 for i in range(1, len(memory_data))
                                 if memory_data[i] > memory_data[i-1])
        increase_ratio = increasing_samples / (len(memory_data) - 1)

        if increase_ratio > 0.7:  # 70% of samples show increase
            leaks.append({
                'type': 'monotonic_increase',
                'severity': 'high' if increase_ratio > 0.9 else 'medium',
                'description': f'Memory shows monotonic increase in {increase_ratio:.1%} of samples',
                'growth_rate': tracking_results.memory_growth_rate if hasattr(tracking_results, 'memory_growth_rate') else 0
            })

        # Check for sudden spikes
        if len(memory_data) > 5:
            mean_memory = statistics.mean(memory_data)
            std_memory = statistics.stdev(memory_data)

            spikes = [m for m in memory_data if m >
                      mean_memory + 3 * std_memory]
            if spikes:
                leaks.append({
                    'type': 'memory_spikes',
                    'severity': 'medium',
                    'description': f'Detected {len(spikes)} memory spikes above 3σ',
                    'spike_values': spikes
                })

        # Check for lack of cleanup
        final_memory = memory_data[-5:]  # Last 5 samples
        initial_memory = memory_data[:5]  # First 5 samples

        if final_memory and initial_memory:
            final_avg = statistics.mean(final_memory)
            initial_avg = statistics.mean(initial_memory)

            if final_avg > initial_avg * 1.5:  # 50% increase from start
                leaks.append({
                    'type': 'insufficient_cleanup',
                    'severity': 'medium',
                    'description': f'Final memory {final_avg:.1f}MB is {final_avg/initial_avg:.1f}x initial memory',
                    'initial_avg': initial_avg,
                    'final_avg': final_avg
                })

        return leaks

    def analyze_fragmentation(self, profile_result: Any) -> Dict[str, float]:
        """Analyze memory fragmentation patterns."""
        if not hasattr(profile_result, 'snapshots') or len(profile_result.snapshots) < 2:
            return {'fragmentation_score': 0.0, 'trend': 0.0}

        fragmentation_scores: List[float] = []

        for snapshot in profile_result.snapshots:
            if snapshot.gpu_memory_reserved_mb > 0:
                utilization = snapshot.gpu_memory_mb / snapshot.gpu_memory_reserved_mb
                fragmentation = 1.0 - utilization
                fragmentation_scores.append(fragmentation)

        if not fragmentation_scores:
            return {'fragmentation_score': 0.0, 'trend': 0.0}

        avg_fragmentation = statistics.mean(fragmentation_scores)

        # Calculate trend
        if len(fragmentation_scores) >= 10:
            early = statistics.mean(fragmentation_scores[:5])
            late = statistics.mean(fragmentation_scores[-5:])
            trend = late - early
        else:
            trend = 0.0

        return {
            'fragmentation_score': avg_fragmentation,
            'trend': trend,
            'max_fragmentation': max(fragmentation_scores),
            'min_fragmentation': min(fragmentation_scores)
        }

    def detect_patterns(self, tracking_results: Any) -> List[Dict[str, Any]]:
        """Detect memory usage patterns."""
        if not hasattr(tracking_results, 'memory_usage') or len(tracking_results.memory_usage) < 20:
            return []

        memory_data = tracking_results.memory_usage
        patterns: List[Dict[str, Any]] = []

        # Detect periodic patterns
        try:
            # Simple autocorrelation for periodic detection
            data_mean = statistics.mean(memory_data)
            normalized_data = [x - data_mean for x in memory_data]

            # Check for patterns in chunks
            chunk_size = min(10, len(normalized_data) // 4)
            if chunk_size > 2:
                chunk_correlations = []
                for i in range(0, len(normalized_data) - chunk_size, chunk_size):
                    chunk1 = normalized_data[i:i+chunk_size]
                    chunk2 = normalized_data[i+chunk_size:i+2*chunk_size]

                    if len(chunk2) == chunk_size:
                        correlation = np.corrcoef(chunk1, chunk2)[0, 1]
                        if not np.isnan(correlation):
                            chunk_correlations.append(correlation)

                if chunk_correlations:
                    avg_correlation = statistics.mean(chunk_correlations)
                    if avg_correlation > 0.7:
                        patterns.append({
                            'type': 'periodic_pattern',
                            'correlation': avg_correlation,
                            'description': f'Detected periodic memory pattern (correlation: {avg_correlation:.3f})'
                        })
        except Exception as exc:
            logging.debug("Periodic pattern detection failed: %s", exc)

        # Detect step patterns
        if len(memory_data) > 10:
            step_increases = 0
            for i in range(1, len(memory_data)):
                if memory_data[i] > memory_data[i-1] * 1.2:  # 20% sudden increase
                    step_increases += 1

            if step_increases > len(memory_data) * 0.1:  # More than 10% of samples
                patterns.append({
                    'type': 'step_pattern',
                    'step_count': step_increases,
                    'description': f'Detected {step_increases} sudden memory increases (step pattern)'
                })

        return patterns

    def analyze_efficiency(self, profile_result: Any) -> float:
        """Analyze memory usage efficiency (0-10 scale)."""
        if not hasattr(profile_result, 'peak_memory_mb'):
            return 0.0

        score = 10.0

        # Penalize high peak memory
        if profile_result.peak_memory_mb > 8000:  # > 8GB
            score -= 3.0
        elif profile_result.peak_memory_mb > 4000:  # > 4GB
            score -= 1.5

        # Penalize high memory growth rate
        if hasattr(profile_result, 'memory_growth_rate'):
            if profile_result.memory_growth_rate > 200:  # > 200 MB/s
                score -= 2.0
            elif profile_result.memory_growth_rate > 100:  # > 100 MB/s
                score -= 1.0

        # Penalize fragmentation
        if hasattr(profile_result, 'snapshots'):
            frag_info = self.analyze_fragmentation(profile_result)
            if frag_info['fragmentation_score'] > 0.5:
                score -= 2.0
            elif frag_info['fragmentation_score'] > 0.3:
                score -= 1.0

        # Penalize memory leaks
        if hasattr(profile_result, 'memory_usage'):
            # Create a simple tracking result for leak detection
            class SimpleTrackingResult:
                def __init__(self, memory_usage: List[float]) -> None:
                    self.memory_usage = memory_usage
                    self.timestamps = list(range(len(memory_usage)))
                    self.memory_growth_rate = 0

            simple_result = SimpleTrackingResult(
                [s.gpu_memory_mb for s in profile_result.snapshots])
            leaks = self.detect_memory_leaks(simple_result)

            high_severity_leaks = [
                l for l in leaks if l.get('severity') == 'high']
            if high_severity_leaks:
                score -= 3.0
            elif leaks:
                score -= 1.5

        return max(0.0, min(10.0, score))

    def correlate_with_performance(self, profile_result: Any) -> Dict[str, Any]:
        """Correlate memory usage with performance metrics."""
        correlation_data: Dict[str, Any] = {
            'memory_duration_correlation': 0.0,
            'function_efficiency': {},
            'recommendations': []
        }
        function_efficiency = cast(
            Dict[str, Dict[str, Any]], correlation_data['function_efficiency']
        )
        recommendations = cast(List[str], correlation_data['recommendations'])

        if not hasattr(profile_result, 'function_profiles'):
            return correlation_data

        # Analyze function efficiency
        for func_name, profile in profile_result.function_profiles.items():
            if profile.get('calls', 0) > 0:
                avg_memory_per_call = profile.get(
                    'total_memory_used', 0) / profile['calls']
                avg_duration_per_call = profile.get(
                    'total_duration', 0) / profile['calls']

                # Calculate efficiency score
                efficiency = 1.0
                if avg_memory_per_call > 1000:  # > 1GB per call
                    efficiency *= 0.5
                if avg_duration_per_call > 1.0:  # > 1 second per call
                    efficiency *= 0.7

                function_efficiency[func_name] = {
                    'avg_memory_per_call': avg_memory_per_call,
                    'avg_duration_per_call': avg_duration_per_call,
                    'efficiency_score': efficiency,
                    'total_calls': profile['calls']
                }

                # Generate recommendations
                if avg_memory_per_call > 2000:  # > 2GB per call
                    recommendations.append(
                        f"Function '{func_name}' uses high memory per call - consider optimization"
                    )

                if profile['calls'] > 100 and avg_duration_per_call > 0.1:
                    recommendations.append(
                        f"Function '{func_name}' called frequently - consider caching or @tf.function"
                    )

        return correlation_data

    # ------------------------------------------------------------------
    # Hidden-memory gap analysis (operates on TelemetryEventV2 series)
    # ------------------------------------------------------------------

    def analyze_memory_gaps(
        self, events: List[TelemetryEventV2]
    ) -> List[GapFinding]:
        """Classify allocator-vs-device hidden memory gaps over time.

        Args:
            events: Chronologically ordered telemetry samples.

        Returns:
            Prioritized list of gap findings (severity desc, confidence desc).
        """
        gaps: List[float] = []
        normalized: List[float] = []
        timestamps_ns: List[int] = []
        usable_events: List[TelemetryEventV2] = []

        for ev in events:
            if ev.device_total_bytes is None or ev.device_total_bytes <= 0:
                continue
            gap = ev.device_used_bytes - ev.allocator_reserved_bytes
            gaps.append(float(gap))
            normalized.append(gap / ev.device_total_bytes)
            timestamps_ns.append(ev.timestamp_ns)
            usable_events.append(ev)

        if len(gaps) < 3:
            return []

        threshold = self.gap_thresholds['gap_ratio_threshold']
        if max(abs(v) for v in normalized) < threshold:
            return []

        findings: List[GapFinding] = []
        findings.extend(self._detect_gap_transient_spikes(gaps, normalized, timestamps_ns))
        findings.extend(self._detect_gap_persistent_drift(gaps, normalized, timestamps_ns))
        findings.extend(self._detect_gap_fragmentation_pattern(usable_events, gaps, normalized))

        severity_order = {'critical': 0, 'warning': 1, 'info': 2}
        findings.sort(key=lambda f: (severity_order.get(f.severity, 9), -f.confidence))
        return findings

    def _detect_gap_transient_spikes(
        self,
        gaps: List[float],
        normalized: List[float],
        timestamps_ns: List[int],
    ) -> List[GapFinding]:
        """Detect transient spikes in the gap series using z-score."""
        arr = np.asarray(gaps, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if std == 0:
            return []

        z_threshold = self.gap_thresholds['gap_spike_zscore']
        spike_indices = [i for i in range(len(arr)) if (arr[i] - mean) / std > z_threshold]

        if not spike_indices:
            return []

        max_idx = int(np.argmax(arr))
        max_gap = float(arr[max_idx])
        max_z = (max_gap - mean) / std

        severity = 'critical' if max_z > 2 * z_threshold else 'warning'
        confidence = min(1.0, max_z / (3 * z_threshold))

        return [GapFinding(
            classification='transient_spike',
            severity=severity,
            confidence=round(confidence, 3),
            evidence={
                'spike_count': len(spike_indices),
                'max_gap_bytes': max_gap,
                'max_zscore': round(max_z, 3),
                'mean_gap_bytes': round(mean, 1),
                'std_gap_bytes': round(std, 1),
            },
            description=(
                f"Detected {len(spike_indices)} transient spike(s) in the "
                f"device-vs-allocator gap (max z-score {max_z:.1f})."
            ),
            remediation=[
                "Investigate non-allocator memory consumers active during spikes "
                "(cuDNN workspace, XLA temporaries, other frameworks).",
                "Use tf.config.experimental.get_memory_info() around spike windows.",
                "Consider setting tf.config.experimental.set_memory_growth(gpu, True).",
            ],
        )]

    def _detect_gap_persistent_drift(
        self,
        gaps: List[float],
        normalized: List[float],
        timestamps_ns: List[int],
    ) -> List[GapFinding]:
        """Detect persistent upward drift in the gap via linear regression."""
        if len(gaps) < 5:
            return []

        x = np.asarray(timestamps_ns, dtype=float)
        x = x - x[0]
        y = np.asarray(gaps, dtype=float)

        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, y)
        r_squared = r_value ** 2

        if slope <= 0 or r_squared < self.gap_thresholds['gap_drift_r_squared']:
            return []

        severity = 'critical' if r_squared > 0.85 else 'warning'
        confidence = round(min(1.0, r_squared), 3)

        slope_bytes_per_sec = slope * 1e9

        return [GapFinding(
            classification='persistent_drift',
            severity=severity,
            confidence=confidence,
            evidence={
                'slope_bytes_per_sec': round(slope_bytes_per_sec, 1),
                'r_squared': round(r_squared, 4),
                'p_value': round(p_value, 6),
                'gap_start_bytes': round(float(y[0]), 1),
                'gap_end_bytes': round(float(y[-1]), 1),
            },
            description=(
                f"Device-vs-allocator gap is drifting upward at "
                f"{format_memory(int(abs(slope_bytes_per_sec)))}/s "
                f"(R\u00b2={r_squared:.2f})."
            ),
            remediation=[
                "Look for non-TensorFlow GPU allocations accumulating over time "
                "(e.g. custom CUDA ops, third-party libraries).",
                "Monitor nvidia-smi used memory alongside TF allocator counters.",
                "If gap stabilises after warmup, it may be one-time runtime overhead.",
            ],
        )]

    def _detect_gap_fragmentation_pattern(
        self,
        events: List[TelemetryEventV2],
        gaps: List[float],
        normalized: List[float],
    ) -> List[GapFinding]:
        """Detect fragmentation-like behaviour: high reserved-allocated ratio."""
        frag_ratios: List[float] = []
        for ev in events:
            reserved = ev.allocator_reserved_bytes
            allocated = ev.allocator_allocated_bytes
            if reserved > 0:
                frag_ratios.append((reserved - allocated) / reserved)

        if not frag_ratios:
            return []

        avg_frag = float(np.mean(frag_ratios))
        max_frag = float(np.max(frag_ratios))

        if avg_frag < self.gap_thresholds['gap_fragmentation_ratio']:
            return []

        severity = 'critical' if avg_frag > 0.5 else 'warning'
        confidence = round(min(1.0, avg_frag / 0.6), 3)

        return [GapFinding(
            classification='fragmentation_like',
            severity=severity,
            confidence=confidence,
            evidence={
                'avg_fragmentation_ratio': round(avg_frag, 4),
                'max_fragmentation_ratio': round(max_frag, 4),
                'avg_gap_bytes': round(float(np.mean(gaps)), 1),
                'sample_count': len(frag_ratios),
            },
            description=(
                f"Allocator fragmentation averaging {avg_frag:.0%} suggests "
                f"reserved-but-unused memory is inflating the device footprint."
            ),
            remediation=[
                "Enable memory growth with tf.config.experimental.set_memory_growth(gpu, True).",
                "Reduce allocation churn by reusing tensors or using tf.function.",
                "Consider setting a hard memory limit with "
                "tf.config.set_logical_device_configuration().",
            ],
        )]

    def score_optimization(
        self,
        profile_result: Any,
        events: Optional[List[TelemetryEventV2]] = None,
    ) -> Dict[str, Any]:
        """Score optimization opportunities.

        Args:
            profile_result: TensorFlow profiling result object.
            events: Optional telemetry event series for gap analysis.
                    When provided, the result includes a ``gap_analysis`` section.
        """
        optimization_score: Dict[str, Any] = {
            'overall_score': 0.0,
            'categories': {},
            'top_recommendations': [],
            'priority_actions': []
        }
        categories = cast(Dict[str, float], optimization_score['categories'])
        priority_actions = cast(List[str], optimization_score['priority_actions'])

        # Memory efficiency
        efficiency_score = self.analyze_efficiency(profile_result)
        categories['memory_efficiency'] = efficiency_score

        # Fragmentation analysis
        if hasattr(profile_result, 'snapshots'):
            frag_info = self.analyze_fragmentation(profile_result)
            frag_score = max(0, 10 - frag_info['fragmentation_score'] * 10)
            categories['fragmentation'] = frag_score
        else:
            frag_score = 5.0

        # Performance correlation
        perf_corr = self.correlate_with_performance(profile_result)
        if perf_corr['function_efficiency']:
            avg_efficiency = statistics.mean([
                func['efficiency_score'] for func in perf_corr['function_efficiency'].values()
            ])
            perf_score = avg_efficiency * 10
        else:
            perf_score = 5.0

        categories['performance'] = perf_score

        # Overall score
        optimization_score['overall_score'] = statistics.mean([
            efficiency_score, frag_score, perf_score
        ])

        # Generate recommendations
        if efficiency_score < 6:
            priority_actions.append(
                "Address memory efficiency issues")

        if frag_score < 6:
            priority_actions.append(
                "Reduce memory fragmentation")

        if perf_score < 6:
            priority_actions.append(
                "Optimize function performance")

        # Top recommendations based on analysis
        from .utils import suggest_optimizations
        top_suggestions = suggest_optimizations(profile_result)
        optimization_score['top_recommendations'] = top_suggestions[:5]

        # Hidden-memory gap analysis (only when telemetry events are supplied).
        if events is not None:
            gap_findings = self.analyze_memory_gaps(events)
            optimization_score['gap_analysis'] = [f.__dict__ for f in gap_findings]

        return optimization_score
