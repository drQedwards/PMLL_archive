# grpc/adaptive_batcher.py
import asyncio
import time
from collections import deque
from typing import List, Any, Dict, Optional
import grpc
from concurrent import futures
import torch
from prometheus_client import Counter, Histogram, Gauge
import logging

# Metrics
BATCH_SIZE = Histogram('pmll_batch_size', 'Size of processed batches')
BATCH_LATENCY = Histogram('pmll_batch_latency', 'Latency of batch processing')
QUEUE_SIZE = Gauge('pmll_queue_size', 'Current queue size')
THROUGHPUT = Counter('pmll_throughput', 'Total items processed')

class AdaptiveBatcher:
    """Adaptive batching system with gRPC integration"""
    
    def __init__(self, 
                 min_batch_size: int = 1,
                 max_batch_size: int = 64,
                 max_latency_ms: int = 50,
                 compression_ratio: float = 0.8960):
        
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.max_latency_ms = max_latency_ms
        self.compression_ratio = compression_ratio
        
        self.queue = asyncio.Queue()
        self.pending_batch = deque()
        self.last_batch_time = time.time()
        
        # Adaptive parameters
        self.current_batch_size = min_batch_size
        self.latency_history = deque(maxlen=100)
        
        self.logger = logging.getLogger(__name__)
        
    async def add_request(self, request: Any) -> asyncio.Future:
        """Add request to batch queue"""
        future = asyncio.Future()
        await self.queue.put((request, future))
        QUEUE_SIZE.set(self.queue.qsize())
        return future
    
    async def run_batcher(self, process_func):
        """Main batching loop"""
        while True:
            batch = []
            futures = []
            
            # Collect batch
            deadline = time.time() + (self.max_latency_ms / 1000.0)
            
            while len(batch) < self.current_batch_size and time.time() < deadline:
                try:
                    remaining_time = deadline - time.time()
                    if remaining_time <= 0:
                        break
                        
                    request, future = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=remaining_time
                    )
                    batch.append(request)
                    futures.append(future)
                    
                except asyncio.TimeoutError:
                    break
            
            if batch:
                # Process batch
                start_time = time.time()
                try:
                    results = await process_func(batch)
                    
                    # Distribute results
                    for future, result in zip(futures, results):
                        future.set_result(result)
                    
                    # Update metrics
                    latency = (time.time() - start_time) * 1000
                    BATCH_SIZE.observe(len(batch))
                    BATCH_LATENCY.observe(latency)
                    THROUGHPUT.inc(len(batch))
                    
                    # Adaptive sizing
                    self._adapt_batch_size(len(batch), latency)
                    
                except Exception as e:
                    # Handle errors
                    for future in futures:
                        future.set_exception(e)
                    self.logger.error(f"Batch processing error: {e}")
            
            await asyncio.sleep(0.001)  # Small delay to prevent CPU spinning
    
    def _adapt_batch_size(self, batch_size: int, latency_ms: float):
        """Adapt batch size based on performance"""
        self.latency_history.append(latency_ms)
        
        if len(self.latency_history) >= 10:
            avg_latency = sum(self.latency_history) / len(self.latency_history)
            
            if avg_latency < self.max_latency_ms * 0.7:
                # We have headroom, increase batch size
                self.current_batch_size = min(
                    self.current_batch_size + 1,
                    self.max_batch_size
                )
            elif avg_latency > self.max_latency_ms * 0.9:
                # Getting close to limit, decrease batch size
                self.current_batch_size = max(
                    self.current_batch_size - 1,
                    self.min_batch_size
                )
