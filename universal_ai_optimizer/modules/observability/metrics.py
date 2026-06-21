"""
Metrics Module
Collects and exports metrics in various formats (e.g., Prometheus)
"""

import logging
import time
import re
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, Optional, List, Callable
from universal_ai_optimizer.core.base import BaseOptimizerModule

logger = logging.getLogger(__name__)

try:
    from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary
    PROMETHEUS_CLIENT_AVAILABLE = True
except ImportError:
    PROMETHEUS_CLIENT_AVAILABLE = False

def _escape_prometheus_label_value(value: str) -> str:
    """Escape label value for Prometheus format"""
    if not isinstance(value, str):
        value = str(value)
    # Escape backslashes, double quotes, and newlines
    value = value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
    return value

class Metrics(BaseOptimizerModule):
    """
    Metrics collector and exporter for the Universal AI Optimizer
    """
    
    def process(self, prompt: str, context: Dict[str, Any], 
                model_adapter: Optional[Any] = None, 
                pipeline_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Dummy process method to satisfy BaseOptimizerModule interface"""
        return {}
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.enabled = self.config.get('enabled', True)
        self.prefix = self.config.get('prefix', 'universal_ai_optimizer')
        self.enable_prometheus = self.config.get('enable_prometheus', True)
        self.prometheus_port = self.config.get('prometheus_port', 8000)
        self.enable_gc_metrics = self.config.get('enable_gc_metrics', True)
        self.max_histogram_size = self.config.get('max_histogram_size', 10000)
        
        # In-memory storage for metrics
        self.counters = {}  # name -> value
        self.gauges = {}    # name -> value
        self.histograms = {}  # name -> deque of values (bounded)
        self.summaries = {}   # name -> {count: int, sum: float}
        
        # Label support (simplified)
        self.labelled_counters = {}  # (name, label_tuple) -> value
        self.labelled_gauges = {}    # (name, label_tuple) -> value
        
        # Thread safety for metrics storage
        self._lock = threading.RLock()
        
        # Start Prometheus server if enabled
        if self.enabled and self.enable_prometheus:
            self._start_prometheus_server()
    
    def _start_prometheus_server(self):
        """Start a Prometheus HTTP server to expose metrics"""
        if PROMETHEUS_CLIENT_AVAILABLE:
            try:
                start_http_server(self.prometheus_port)
                self.logger.info(f"Prometheus metrics server started on port {self.prometheus_port}")
                return
            except Exception as e:
                self.logger.warning(f"Failed to start prometheus_client server: {e}")

        class PrometheusHandler(BaseHTTPRequestHandler):
            metrics_ref = self

            def do_GET(self):
                if self.path == '/metrics':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/plain; version=0.0.4')
                    self.end_headers()
                    data = self.metrics_ref.get_prometheus_metrics().encode()
                    self.wfile.write(data)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass

        try:
            prometheus_bind = self.config.get('prometheus_bind', '127.0.0.1')
            self._prometheus_server = HTTPServer((prometheus_bind, self.prometheus_port), PrometheusHandler)
            self._prometheus_thread = threading.Thread(target=self._prometheus_server.serve_forever, daemon=True)
            self._prometheus_thread.start()
            self.logger.info(f"Prometheus metrics endpoint started on port {self.prometheus_port}")
        except Exception as e:
            self.logger.warning(f"Failed to start Prometheus HTTP server: {e}")
    
    def _get_full_name(self, name: str) -> str:
        """Get the full metric name with prefix"""
        if self.prefix:
            return f"{self.prefix}_{name}"
        return name
    
    def increment_counter(self, name: str, value: float = 1.0, 
                         labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        if not self.enabled:
            return
        
        full_name = self._get_full_name(name)
        with self._lock:
            if labels:
                label_tuple = tuple(sorted(labels.items()))
                key = (full_name, label_tuple)
                self.labelled_counters[key] = self.labelled_counters.get(key, 0) + value
            else:
                self.counters[full_name] = self.counters.get(full_name, 0) + value
        
        self.logger.debug(f"Incremented counter {name} by {value} with labels {labels}")
    
    def set_gauge(self, name: str, value: float, 
                 labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric"""
        if not self.enabled:
            return
        
        full_name = self._get_full_name(name)
        with self._lock:
            if labels:
                label_tuple = tuple(sorted(labels.items()))
                key = (full_name, label_tuple)
                self.labelled_gauges[key] = value
            else:
                self.gauges[full_name] = value
        
        self.logger.debug(f"Set gauge {name} to {value} with labels {labels}")
    
    def observe_histogram(self, name: str, value: float, 
                         labels: Optional[Dict[str, str]] = None):
        """Observe a value in a histogram"""
        if not self.enabled:
            return
        
        full_name = self._get_full_name(name)
        with self._lock:
            if labels:
                label_tuple = tuple(sorted(labels.items()))
                key = (full_name, label_tuple)
                if key not in self.histograms:
                    from collections import deque
                    self.histograms[key] = deque(maxlen=self.max_histogram_size)
                self.histograms[key].append(value)
            else:
                if full_name not in self.histograms:
                    from collections import deque
                    self.histograms[full_name] = deque(maxlen=self.max_histogram_size)
                self.histograms[full_name].append(value)
        
        self.logger.debug(f"Observed histogram {name} with value {value} and labels {labels}")
    
    def observe_summary(self, name: str, value: float, 
                       labels: Optional[Dict[str, str]] = None):
        """Observe a value in a summary"""
        if not self.enabled:
            return
        
        full_name = self._get_full_name(name)
        with self._lock:
            if labels:
                label_tuple = tuple(sorted(labels.items()))
                key = (full_name, label_tuple)
                if key not in self.summaries:
                    self.summaries[key] = {'count': 0, 'sum': 0.0}
                self.summaries[key]['count'] += 1
                self.summaries[key]['sum'] += value
            else:
                if full_name not in self.summaries:
                    self.summaries[full_name] = {'count': 0, 'sum': 0.0}
                self.summaries[full_name]['count'] += 1
                self.summaries[full_name]['sum'] += value
        
        self.logger.debug(f"Observed summary {name} with value {value} and labels {labels}")
    
    def get_prometheus_metrics(self) -> str:
        """Generate Prometheus-formatted metrics string"""
        if not self.enabled or not self.enable_prometheus:
            return "# Metrics collection disabled\n"
        
        lines = []
        # Add metadata
        lines.append(f"# HELP {self._get_full_name('info')} Service information")
        lines.append(f"# TYPE {self._get_full_name('info')} gauge")
        lines.append(f'{self._get_full_name("info")}{{service="{self._get_service_name()}",version="{self._get_service_version()}"}} 1')
        lines.append("")
        
        # Counters
        for name, value in self.counters.items():
            lines.append(f"# HELP {name} Counter metric")
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")
        
        # Labelled counters
        for (name, label_tuple), value in self.labelled_counters.items():
            labels_str = ",".join([f'{k}="{_escape_prometheus_label_value(v)}"' for k, v in label_tuple])
            lines.append(f"# HELP {name} Labelled counter metric")
            lines.append(f"# TYPE {name} counter")
            if labels_str:
                lines.append(f"{name}{{{labels_str}}} {value}")
            else:
                lines.append(f"{name} {value}")
        
        # Gauges
        for name, value in self.gauges.items():
            lines.append(f"# HELP {name} Gauge metric")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        
        # Labelled gauges
        for (name, label_tuple), value in self.labelled_gauges.items():
            labels_str = ",".join([f'{k}="{_escape_prometheus_label_value(v)}"' for k, v in label_tuple])
            lines.append(f"# HELP {name} Labelled gauge metric")
            lines.append(f"# TYPE {name} gauge")
            if labels_str:
                lines.append(f"{name}{{{labels_str}}} {value}")
            else:
                lines.append(f"{name} {value}")
        
        # Histograms
        for name, values in self.histograms.items():
            if not values:
                continue
            count = len(values)
            sum_vals = sum(values)
            # Bucket boundaries (example)
            buckets = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, float('inf')]
            bucket_counts = [0] * len(buckets)
            for v in values:
                for i, b in enumerate(buckets):
                    if v <= b:
                        bucket_counts[i] += 1
            lines.append(f"# HELP {name} Histogram metric")
            lines.append(f"# TYPE {name} histogram")
            for i, b in enumerate(buckets):
                if b == float('inf'):
                    le_str = "+Inf"
                else:
                    le_str = str(b)
                lines.append(f"{name}_bucket{{le=\"{le_str}\"}} {bucket_counts[i]}")
            lines.append(f"{name}_sum {sum_vals}")
            lines.append(f"{name}_count {count}")
        
        # Summaries
        for name, data in self.summaries.items():
            count = data['count']
            sum_vals = data['sum']
            lines.append(f"# HELP {name} Summary metric")
            lines.append(f"# TYPE {name} summary")
            lines.append(f"{name}_sum {sum_vals}")
            lines.append(f"{name}_count {count}")
        
        return "\n".join(lines) + "\n"
    
    def _get_service_name(self) -> str:
        """Get service name from configuration or default"""
        # In a real implementation, we might get this from telemetry or config
        return "universal_ai_optimizer"
    
    def _get_service_version(self) -> str:
        """Get service version from configuration or default"""
        return "0.1.0"
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics metrics"""
        base_metrics = super().get_metrics()
        base_metrics.update({
            'enabled': self.enabled,
            'prefix': self.prefix,
            'enable_prometheus': self.enable_prometheus,
            'prometheus_port': self.prometheus_port,
            'counters_count': len(self.counters),
            'gauges_count': len(self.gauges),
            'histograms_count': len(self.histograms),
            'summaries_count': len(self.summaries),
            'labelled_counters_count': len(self.labelled_counters),
            'labelled_gauges_count': len(self.labelled_gauges)
        })
        return base_metrics