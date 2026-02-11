import time
import numpy as np


class LatencyMonitor:
    def __init__(self, predictor):
        self.predictor = predictor
    def measure(self, text, value, n_runs=100):
        latencies = []
        for _ in range(n_runs):
            start = time.time()
            self.predictor.predict(text, value)
            end = time.time()
            latencies.append((end - start) * 1000)

        latencies = np.array(latencies)
        return {
            "avg_latency_ms": float(latencies.mean()),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "max_latency_ms": float(latencies.max()),
        }
