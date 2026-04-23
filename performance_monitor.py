# Performance monitoring for VividMedi
# Tracks: response times, cache hits, API latency, transcription speed

import time
import json
from datetime import datetime
from functools import wraps
from collections import defaultdict

class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = time.time()
    
    def record_endpoint(self, endpoint, method, status, duration_ms, cache_status=None):
        """Record endpoint performance"""
        self.metrics[f"{method} {endpoint}"].append({
            "status": status,
            "duration_ms": duration_ms,
            "cache_status": cache_status,
            "timestamp": datetime.utcnow().isoformat()
        })
        # Keep only last 1000 records per endpoint
        if len(self.metrics[f"{method} {endpoint}"]) > 1000:
            self.metrics[f"{method} {endpoint}"] = self.metrics[f"{method} {endpoint}"][-1000:]
    
    def get_stats(self, endpoint=None):
        """Calculate statistics"""
        if endpoint:
            records = self.metrics.get(endpoint, [])
        else:
            records = [r for records in self.metrics.values() for r in records]
        
        if not records:
            return None
        
        durations = [r["duration_ms"] for r in records]
        statuses = defaultdict(int)
        cache_hits = 0
        
        for r in records:
            statuses[r["status"]] += 1
            if r.get("cache_status") == "HIT":
                cache_hits += 1
        
        return {
            "requests": len(records),
            "avg_duration_ms": sum(durations) / len(durations),
            "min_duration_ms": min(durations),
            "max_duration_ms": max(durations),
            "p95_duration_ms": sorted(durations)[int(len(durations) * 0.95)] if durations else 0,
            "p99_duration_ms": sorted(durations)[int(len(durations) * 0.99)] if durations else 0,
            "status_codes": dict(statuses),
            "cache_hit_rate": (cache_hits / len(records) * 100) if records else 0,
            "uptime_seconds": time.time() - self.start_time
        }
    
    def get_all_stats(self):
        """Get stats for all endpoints"""
        stats = {}
        for endpoint in self.metrics.keys():
            stats[endpoint] = self.get_stats(endpoint)
        return stats

monitor = PerformanceMonitor()

def track_performance(f):
    """Decorator to track endpoint performance"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = f(*args, **kwargs)
            duration_ms = (time.time() - start) * 1000
            
            # Extract endpoint and status
            from flask import request
            endpoint = request.path
            method = request.method
            status = 200
            cache_status = None
            
            # Try to get response status
            if hasattr(result, 'status_code'):
                status = result.status_code
            
            monitor.record_endpoint(endpoint, method, status, duration_ms, cache_status)
            return result
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            from flask import request
            monitor.record_endpoint(request.path, request.method, 500, duration_ms)
            raise
    
    return wrapper
