"""
Monitoring Module
Provides health checks, system monitoring, and alerting capabilities
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, List, Callable
from universal_ai_optimizer.core.base import BaseOptimizerModule

logger = logging.getLogger(__name__)

class Monitoring(BaseOptimizerModule):
    """
    Monitoring component for health checks, system metrics, and alerting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.enabled = self.config.get('enabled', True)
        self.health_check_interval = self.config.get('health_check_interval', 30.0)  # seconds
        self.enable_system_monitoring = self.config.get('enable_system_monitoring', True)
        self.enable_custom_health_checks = self.config.get('enable_custom_health_checks', True)
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'cpu_usage_percent': 80.0,
            'memory_usage_percent': 85.0,
            'disk_usage_percent': 90.0,
            'latency_ms': 1000.0,
            'error_rate_percent': 5.0
        })
        
        # Health check registry
        self.health_checks = {}  # name -> callable
        self.health_status = {}  # name -> (status, timestamp, details)
        
        # System monitoring thread
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
        # Initialize default health checks
        self._init_default_health_checks()
        
        # Start monitoring if enabled
        if self.enabled and self.enable_system_monitoring:
            self._start_monitoring()
    
    def _init_default_health_checks(self):
        """Initialize default health checks"""
        self.register_health_check('system_resources', self._check_system_resources)
        self.register_health_check('internal_state', self._check_internal_state)
        self.logger.debug("Default health checks registered")
    
    def _start_monitoring(self):
        """Start the background monitoring thread"""
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info(f"Monitoring started with interval {self.health_check_interval}s")
    
    def _monitoring_loop(self):
        """Background loop for running health checks"""
        while not self.stop_monitoring.wait(self.health_check_interval):
            try:
                self.run_health_checks()
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def register_health_check(self, name: str, check_func: Callable[[], Dict[str, Any]]):
        """Register a health check function"""
        self.health_checks[name] = check_func
        self.logger.debug(f"Registered health check: {name}")
    
    def unregister_health_check(self, name: str):
        """Unregister a health check"""
        if name in self.health_checks:
            del self.health_checks[name]
            if name in self.health_status:
                del self.health_status[name]
            self.logger.debug(f"Unregistered health check: {name}")
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks and return overall status"""
        overall_status = 'healthy'
        overall_details = {}
        
        for name, check_func in self.health_checks.items():
            try:
                start_time = time.time()
                result = check_func()
                check_duration = time.time() - start_time
                
                # Expect result to be a dict with at least 'status' key
                # status can be bool, or string like 'healthy', 'unhealthy', 'degraded'
                status = result.get('status', True)
                if isinstance(status, bool):
                    status = 'healthy' if status else 'unhealthy'
                
                details = result.get('details', {})
                details['check_duration_ms'] = check_duration * 1000
                
                self.health_status[name] = {
                    'status': status,
                    'timestamp': time.time(),
                    'details': details
                }
                
                # Update overall status
                if status in ['unhealthy', 'critical']:
                    overall_status = 'unhealthy'
                elif status == 'degraded' and overall_status == 'healthy':
                    overall_status = 'degraded'
                
                overall_details[name] = self.health_status[name]
                
            except Exception as e:
                self.logger.error(f"Health check {name} failed: {e}")
                self.health_status[name] = {
                    'status': 'unhealthy',
                    'timestamp': time.time(),
                    'details': {'error': str(e)}
                }
                overall_status = 'unhealthy'
                overall_details[name] = self.health_status[name]
        
        return {
            'status': overall_status,
            'timestamp': time.time(),
            'checks': overall_details
        }
    
    def get_health_status(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get health status for a specific check or all checks"""
        if name:
            return self.health_status.get(name, {'status': 'unknown', 'details': {}})
        return {
            'status': self._get_overall_status(),
            'timestamp': time.time(),
            'checks': self.health_status
        }
    
    def _get_overall_status(self) -> str:
        """Calculate overall health status"""
        if not self.health_status:
            return 'unknown'
        
        statuses = [check['status'] for check in self.health_status.values()]
        if any(s in ['unhealthy', 'critical'] for s in statuses):
            return 'unhealthy'
        elif any(s == 'degraded' for s in statuses):
            return 'degraded'
        elif all(s == 'healthy' for s in statuses):
            return 'healthy'
        else:
            return 'unknown'
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            # In a real implementation, we would use psutil or similar
            # For now, we'll use placeholder values or try to import psutil
            try:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                cpu_healthy = cpu_percent < self.alert_thresholds['cpu_usage_percent']
                memory_healthy = memory.percent < self.alert_thresholds['memory_usage_percent']
                disk_healthy = disk.percent < self.alert_thresholds['disk_usage_percent']
                
                status = 'healthy' if (cpu_healthy and memory_healthy and disk_healthy) else 'unhealthy'
                
                return {
                    'status': status,
                    'details': {
                        'cpu_usage_percent': cpu_percent,
                        'memory_usage_percent': memory.percent,
                        'disk_usage_percent': disk.percent,
                        'memory_available_gb': memory.available / (1024**3),
                        'disk_free_gb': disk.free / (1024**3)
                    }
                }
            except ImportError:
                # Fallback if psutil not available
                return {
                    'status': 'healthy',  # Assume healthy if we can't check
                    'details': {
                        'note': 'psutil not installed, skipping detailed resource check'
                    }
                }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'details': {'error': str(e)}
            }
    
    def _check_internal_state(self) -> Dict[str, Any]:
        """Check internal state of the optimizer"""
        # This would check things like:
        #   - Whether critical components are initialized
        #   - Queue depths, etc.
        # For now, return a placeholder
        return {
            'status': 'healthy',
            'details': {
                'component': 'universal_ai_optimizer',
                'message': 'Internal state check placeholder'
            }
        }
    
    def shutdown(self):
        """Shutdown the monitoring thread"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.stop_monitoring.set()
            self.monitoring_thread.join(timeout=5.0)
            self.logger.info("Monitoring shutdown")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get monitoring metrics"""
        base_metrics = super().get_metrics()
        base_metrics.update({
            'enabled': self.enabled,
            'health_check_interval': self.health_check_interval,
            'registered_health_checks': list(self.health_checks.keys()),
            'health_check_count': len(self.health_checks),
            'overall_status': self._get_overall_status()
        })
        return base_metrics