# grpc/server.py
import grpc
from concurrent import futures
import asyncio
import logging
from typing import Any
import torch

from core.pmll.lattice import PMLLLattice
from core.xgraph.memory_graph import XGraphMemory
from grpc.adaptive_batcher import AdaptiveBatcher
from observability.metrics import setup_metrics

# Import generated protobuf
from grpc.protos import pmll_pb2, pmll_pb2_grpc

class PMLLService(pmll_pb2_grpc.PMLLServiceServicer):
    """Main gRPC service for PMLL"""
    
    def __init__(self, config: dict):
        self.config = config
        self.lattice = PMLLLattice(config)
        self.x_graph = XGraphMemory(
            dimensions=(config['x_graph_dim'], config['x_graph_dim']),
            compression_ratio=config['compression_ratio']
        )
        self.batcher = AdaptiveBatcher(
            max_batch_size=config['max_batch_size'],
            max_latency_ms=config['max_latency_ms'],
            compression_ratio=config['compression_ratio']
        )
        
        # Start batcher
        asyncio.create_task(self.batcher.run_batcher(self.process_batch))
        
        self.logger = logging.getLogger(__name__)
    
    async def Process(self, request, context):
        """Process single request through PMLL"""
        # Convert request to tensor
        input_tensor = torch.tensor(request.data).reshape(request.shape)
        
        # Add to batch queue
        future = await self.batcher.add_request({
            'tensor': input_tensor,
            'metadata': request.metadata
        })
        
        # Wait for batch processing
        result = await future
        
        # Convert result back to response
        response = pmll_pb2.ProcessResponse()
        response.result.CopyFrom(result.numpy().tobytes())
        response.compression_ratio = self.config['compression_ratio']
        
        return response
    
    async def process_batch(self, batch: list) -> list:
        """Process a batch of requests"""
        tensors = [item['tensor'] for item in batch]
        stacked = torch.stack(tensors)
        
        # Process through lattice
        result = await self.lattice.process_x_graph(stacked)
        
        # Route through x-graph memory
        path = self._compute_optimal_path(result)
        result = self.x_graph.route_data(result, path)
        
        # Split results back
        results = torch.unbind(result, dim=0)
        return results
    
    def _compute_optimal_path(self, tensor: torch.Tensor) -> list:
        """Compute optimal path through x-graph"""
        # Simplified path computation
        dim = self.config['x_graph_dim']
        path = []
        for i in range(dim):
            for j in range(dim):
                if (i + j) % 2 == 0:  # X pattern
                    path.append(f"x_{i}_{j}")
        return path

async def serve():
    """Start gRPC server"""
    config = {
        'memory_size_gb': 64,
        'attention_petals': 8,
        'hidden_dim': 768,
        'memory_nodes': 4,
        'x_graph_dim': 16,
        'compression_ratio': 0.8960,
        'max_batch_size': 64,
        'max_latency_ms': 50
    }
    
    # Setup metrics
    setup_metrics()
    
    # Create server
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    pmll_pb2_grpc.add_PMLLServiceServicer_to_server(
        PMLLService(config), server
    )
    
    # Listen on port
    server.add_insecure_port('[::]:50051')
    
    logging.info("Starting PMLL gRPC server on port 50051")
    await server.start()
    await server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())
