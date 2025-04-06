"""
JaxType与Beartype集成使用示例

本文件展示了如何将JaxType与Beartype结合使用，以提供更强大的张量类型检查。
"""

import torch
import jaxtyping
from jaxtyping import Array, Float, Int, Bool
from typing import Tuple, List, Dict, Optional, Union

try:
    from beartype import beartype
    from beartype.vale import Is
    HAS_BEARTYPE = True
except ImportError:
    def beartype(func):
        return func
    HAS_BEARTYPE = False

@beartype
def tensor_shape_check(x: Float[Array, "batch_size seq_len hidden_dim"]) -> Float[Array, "batch_size hidden_dim"]:
    """检查张量形状并返回正确形状的张量"""
    batch_size, seq_len, hidden_dim = x.shape
    return x.mean(dim=1)  # 返回形状为 [batch_size, hidden_dim] 的张量

@beartype
def tensor_dtype_check(
    x: Float[Array, "batch_size seq_len"], 
    mask: Bool[Array, "batch_size seq_len"]
) -> Float[Array, "batch_size"]:
    """检查张量数据类型并返回正确类型的张量"""
    batch_size = x.shape[0]
    return (x * mask).sum(dim=1)

@beartype
def complex_tensor_op(
    query: Float[Array, "batch_size seq_len head_dim"],
    key: Float[Array, "batch_size seq_len head_dim"],
    value: Float[Array, "batch_size seq_len head_dim"],
    mask: Optional[Bool[Array, "batch_size seq_len"]] = None
) -> Float[Array, "batch_size seq_len head_dim"]:
    """复杂的张量操作示例"""
    scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)
    
    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
    
    attn = torch.softmax(scores, dim=-1)
    
    output = torch.matmul(attn, value)
    
    return output

if HAS_BEARTYPE:
    from beartype.vale import Is
    
    @beartype
    def precise_tensor_check(
        x: Float[Array, "batch_size seq_len hidden_dim"] & Is[lambda t: t.shape[1] <= 512],
        dropout_prob: float & Is[lambda p: 0.0 <= p <= 1.0]
    ) -> Float[Array, "batch_size seq_len hidden_dim"]:
        """使用beartype.vale进行更精确的类型检查"""
        if dropout_prob > 0 and torch.is_grad_enabled():
            return torch.nn.functional.dropout(x, p=dropout_prob, training=True)
        return x

@beartype
def device_aware_op(
    x: Float[Array, "batch_size seq_len hidden_dim"],
    y: Float[Array, "batch_size seq_len hidden_dim"]
) -> Float[Array, "batch_size seq_len hidden_dim"]:
    """处理不同设备上的张量"""
    if x.device != y.device:
        y = y.to(device=x.device)
    
    return x + y

def main():
    batch_size, seq_len, hidden_dim = 2, 10, 20
    x = torch.randn(batch_size, seq_len, hidden_dim)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    result1 = tensor_shape_check(x)
    print(f"Shape check result shape: {result1.shape}")
    
    result2 = tensor_dtype_check(x[:,:,0], mask)
    print(f"Dtype check result shape: {result2.shape}")
    
    query = key = value = torch.randn(batch_size, seq_len, hidden_dim // 2)
    result3 = complex_tensor_op(query, key, value, mask)
    print(f"Complex op result shape: {result3.shape}")
    
    if HAS_BEARTYPE:
        try:
            result4 = precise_tensor_check(x, 0.1)
            print(f"Precise check result shape: {result4.shape}")
        except Exception as e:
            print(f"Precise check error: {e}")
    
    y = torch.randn(batch_size, seq_len, hidden_dim)
    result5 = device_aware_op(x, y)
    print(f"Device aware op result shape: {result5.shape}")

if __name__ == "__main__":
    main()
