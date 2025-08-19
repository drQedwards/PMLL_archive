# tests/integration/test_pmll_system.py
import pytest
import torch
import asyncio
import grpc
from grpc.protos import pmll_pb2, pmll_pb2_grpc

@pytest.fixture
async def pmll_client():
    """Create gRPC client for testing"""
    channel = grpc.aio.insecure_channel('localhost:50051')
    stub = pmll_pb2_grpc.PMLLServiceStub(channel)
    yield stub
    await channel.close()

@pytest.mark.asyncio
async def test_single_request(pmll_client):
    """Test single request processing"""
    # Create test data
    test_data = torch.randn(1, 768)
    
    # Create request
    request = pmll_pb2.ProcessRequest()
    request.data = test_data.numpy().tobytes()
    request.shape.extend(test_data.shape)
    
    # Send request
    response = await pmll_client.Process(request)
    
    # Verify response
    assert response.compression_ratio == 0.8960
    assert len(response.result) > 0

@pytest.mark.asyncio
async def test_batch_processing(pmll_client):
    """Test batch processing efficiency"""
    requests = []
    
    # Create multiple requests
    for _ in range(32):
        test_data = torch.randn(1, 768)
        request = pmll_pb2.ProcessRequest()
        request.data = test_data.numpy().tobytes()
        request.shape.extend(test_data.shape)
        requests.append(request)
    
    # Send requests concurrently
    tasks = [pmll_client.Process(req) for req in requests]
    responses = await asyncio.gather(*tasks)
    
    # Verify all responses
    assert len(responses) == 32
    for response in responses:
        assert response.compression_ratio == 0.8960

@pytest.mark.asyncio  
async def test_x_graph_routing():
    """Test x-graph memory routing"""
    from core.xgraph.memory_graph import XGraphMemory
    
    x_graph = XGraphMemory(dimensions=(8, 8), compression_ratio=0.8960)
    test_tensor = torch.randn(1, 768)
    
    # Test routing
    path = ['x_0_0', 'x_1_1', 'x_2_2', 'x_3_3']
    result = x_graph.route_data(test_tensor, path)
    
    assert result.shape == test_tensor.shape
    assert not torch.isnan(result).any()

@pytest.mark.asyncio
async def test_attention_flower():
    """Test attention flower mechanism"""
    from core.xgraph.attention import AttentionFlower
    
    flower = AttentionFlower(num_petals=8, hidden_dim=768)
    test_input = torch.randn(2, 10, 768)  # batch=2, seq_len=10
    
    output = await flower.forward(test_input)
    
    assert output.shape == test_input.shape
    assert not torch.isnan(output).any()
