"""
System Monitoring Module
Provides CPU, RAM, GPU, VRAM, Network, and Disk monitoring.
"""

import logging
import threading
import time
import os
from typing import Dict, Any, Optional
from universal_ai_optimizer.core.base import BaseOptimizerModule

logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import subprocess
    SUBPROCESS_AVAILABLE = True
except ImportError:
    SUBPROCESS_AVAILABLE = False


class SystemMonitor(BaseOptimizerModule):
    """
    Monitors system resources: CPU, RAM, GPU, VRAM, Network, Disk.
    All metrics are collected lazily on demand.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.enabled = self.config.get('enabled', True)
        self._gpu_available = False
        self._last_network = None
        self._last_net_time = None

        if self.enabled and PSUTIL_AVAILABLE:
            self._check_gpu()
            self.logger.info("System monitor initialized (psutil available)")
        elif self.enabled:
            self.logger.warning("psutil not installed. System monitoring limited.")

    def _check_gpu(self):
        """Check if nvidia-smi is available for GPU monitoring."""
        try:
            if SUBPROCESS_AVAILABLE:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                    capture_output=True, text=True, timeout=5
                )
                self._gpu_available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            self._gpu_available = False

    def process(self, prompt: str, context: Dict[str, Any],
                model_adapter=None, pipeline_state=None) -> Dict[str, Any]:
        if not self.enabled:
            return {}

        return self.get_system_stats()

    def get_system_stats(self) -> Dict[str, Any]:
        stats = {
            'timestamp': time.time(),
            'cpu': self.get_cpu_stats(),
            'ram': self.get_ram_stats(),
            'disk': self.get_disk_stats(),
            'network': self.get_network_stats(),
        }
        if self._gpu_available:
            stats['gpu'] = self.get_gpu_stats()
        return stats

    def get_cpu_stats(self) -> Dict[str, Any]:
        if not PSUTIL_AVAILABLE:
            return {'available': False}
        try:
            return {
                'available': True,
                'percent': psutil.cpu_percent(interval=0.1),
                'count_logical': psutil.cpu_count(),
            }
        except Exception as e:
            return {'available': False, 'error': str(e)}

    def get_ram_stats(self) -> Dict[str, Any]:
        if not PSUTIL_AVAILABLE:
            return {'available': False}
        try:
            mem = psutil.virtual_memory()
            return {
                'available': True,
                'total_gb': round(mem.total / (1024 ** 3), 2),
                'used_gb': round(mem.used / (1024 ** 3), 2),
                'available_gb': round(mem.available / (1024 ** 3), 2),
                'percent': mem.percent,
            }
        except Exception as e:
            return {'available': False, 'error': str(e)}

    def get_disk_stats(self) -> Dict[str, Any]:
        if not PSUTIL_AVAILABLE:
            return {'available': False}
        try:
            disk = psutil.disk_usage('/')
            return {
                'available': True,
                'total_gb': round(disk.total / (1024 ** 3), 2),
                'used_gb': round(disk.used / (1024 ** 3), 2),
                'free_gb': round(disk.free / (1024 ** 3), 2),
                'percent': disk.percent,
            }
        except Exception as e:
            return {'available': False, 'error': str(e)}

    def get_network_stats(self) -> Dict[str, Any]:
        if not PSUTIL_AVAILABLE:
            return {'available': False}
        try:
            net = psutil.net_io_counters()
            now = time.time()
            rates = {'bytes_sent_rate': 0, 'bytes_recv_rate': 0}

            if self._last_network and self._last_net_time:
                dt = now - self._last_net_time
                if dt > 0:
                    rates['bytes_sent_rate'] = (net.bytes_sent - self._last_network.bytes_sent) / dt
                    rates['bytes_recv_rate'] = (net.bytes_recv - self._last_network.bytes_recv) / dt

            self._last_network = net
            self._last_net_time = now

            return {
                'available': True,
                'bytes_sent': net.bytes_sent,
                'bytes_recv': net.bytes_recv,
                'packets_sent': net.packets_sent,
                'packets_recv': net.packets_recv,
                **rates,
            }
        except Exception as e:
            return {'available': False, 'error': str(e)}

    def get_gpu_stats(self) -> Dict[str, Any]:
        if not self._gpu_available or not SUBPROCESS_AVAILABLE:
            return {'available': False}
        try:
            result = subprocess.run(
                [
                    'nvidia-smi',
                    '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu',
                    '--format=csv,noheader,nounits'
                ],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return {'available': False, 'error': 'nvidia-smi failed'}

            parts = [p.strip() for p in result.stdout.strip().split(',')]
            if len(parts) >= 6:
                return {
                    'available': True,
                    'name': parts[0],
                    'vram_total_mb': int(parts[1]),
                    'vram_used_mb': int(parts[2]),
                    'vram_free_mb': int(parts[3]),
                    'gpu_utilization': int(parts[4]),
                    'temperature_c': int(parts[5]),
                }
            return {'available': False, 'error': 'unexpected nvidia-smi output'}
        except (subprocess.TimeoutExpired, ValueError, OSError) as e:
            return {'available': False, 'error': str(e)}

    def get_metrics(self) -> Dict[str, Any]:
        base_metrics = super().get_metrics()
        stats = self.get_system_stats() if self.enabled else {}
        base_metrics.update({
            'enabled': self.enabled,
            'psutil_available': PSUTIL_AVAILABLE,
            'gpu_available': self._gpu_available,
            'current_stats': stats,
        })
        return base_metrics
