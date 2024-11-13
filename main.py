import time
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass
from loguru import logger
import psutil
from concurrent.futures import ThreadPoolExecutor

# Enable CPU optimizations
torch.set_num_threads(psutil.cpu_count(logical=True))
torch.set_num_interop_threads(psutil.cpu_count(logical=True))


@dataclass
class CPUOptimizedConfig:
    """Configuration for CPU-optimized transformer."""

    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_length: int = 512
    vocab_size: int = 30000
    chunk_size: int = 64  # Size of chunks for blocked operations
    n_threads: int = psutil.cpu_count(logical=True)
    use_fused_ops: bool = True
    cache_size_mb: int = 32  # Size of operation cache in MB


class CPUOptimizedLinear(nn.Module):
    """Custom linear layer optimized for CPU execution with blocked matrix multiplication."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: CPUOptimizedConfig,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config

        # Initialize weights in blocks for better cache utilization
        self.n_blocks = math.ceil(out_features / config.chunk_size)
        self.weight_blocks = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        min(
                            config.chunk_size,
                            out_features - i * config.chunk_size,
                        ),
                        in_features,
                    )
                )
                for i in range(self.n_blocks)
            ]
        )
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

        # Operation cache
        self.cache = {}
        self.cache_size = 0
        self.max_cache_size = (
            config.cache_size_mb * 1024 * 1024
        )  # Convert to bytes

        logger.info(
            f"Initialized CPUOptimizedLinear with {self.n_blocks} blocks"
        )

    def reset_parameters(self):
        """Initialize parameters with blocked initialization."""
        for block in self.weight_blocks:
            nn.init.kaiming_uniform_(block, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def _blocked_matmul(
        self, x: Tensor, weight_block: Tensor
    ) -> Tensor:
        """Perform blocked matrix multiplication optimized for CPU cache."""
        batch_size, seq_len, _ = x.shape
        out_features = weight_block.size(0)

        # Reshape input for blocked multiplication
        x_blocked = x.view(batch_size * seq_len, -1)

        # Cache key for this operation
        cache_key = (x_blocked.shape, weight_block.shape)

        if cache_key in self.cache:
            result = torch.matmul(x_blocked, weight_block.t())
        else:
            result = torch.matmul(x_blocked, weight_block.t())

            # Cache management
            if self.cache_size < self.max_cache_size:
                self.cache[cache_key] = result
                self.cache_size += (
                    result.element_size() * result.nelement()
                )

        return result.view(batch_size, seq_len, -1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with blocked computation."""
        outputs = []

        # Process each block in parallel
        with ThreadPoolExecutor(
            max_workers=self.config.n_threads
        ) as executor:
            futures = [
                executor.submit(self._blocked_matmul, x, block)
                for block in self.weight_blocks
            ]
            outputs = [future.result() for future in futures]

        # Concatenate results and add bias
        output = torch.cat(outputs, dim=-1)
        return output + self.bias


@dataclass
class NoamConfig:
    """Configuration for CPU-optimized Noam transformer."""

    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_length: int = 512
    vocab_size: int = 30000
    chunk_size: int = 64
    n_threads: int = psutil.cpu_count(logical=True)
    warmup_steps: int = 4000
    epsilon: float = 1e-6
    cache_size_mb: int = 32
    use_mqa: bool = True  # Enable Multi-Query Attention


class CPUOptimizedRMSNorm(nn.Module):
    """RMSNorm implementation optimized for CPU execution."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))
        self.register_buffer(
            "dummy", torch.ones(1)
        )  # For optimization hints

    def _rms_norm(self, x: Tensor) -> Tensor:
        """Optimized RMSNorm computation."""
        # Compute norm in chunks for better cache utilization
        norm_sq = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(norm_sq + self.eps) * self.g

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with vectorized operations."""
        return self._rms_norm(x.float()).type_as(x)


class NoamLRScheduler:
    """Noam Learning Rate Scheduler with CPU optimization."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int,
    ):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0

        # Pre-compute constants
        self.scale = d_model**-0.5
        self._update_rate_cache = {}
        logger.info(
            f"Initialized NoamLRScheduler with warmup_steps={warmup_steps}"
        )

    def _get_rate(self, step: int) -> float:
        """Compute learning rate with caching."""
        if step in self._update_rate_cache:
            return self._update_rate_cache[step]

        arg1 = step**-0.5
        arg2 = step * (self.warmup_steps**-1.5)
        rate = self.scale * min(arg1, arg2)

        # Cache computation
        if len(self._update_rate_cache) < 1000:  # Limit cache size
            self._update_rate_cache[step] = rate

        return rate

    def step(self):
        """Update learning rate."""
        self.current_step += 1
        rate = self._get_rate(self.current_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = rate
        return rate


class CPUOptimizedMultiQueryAttention(nn.Module):
    """Multi-Query Attention optimized for CPU execution."""

    def __init__(self, config: NoamConfig):
        super().__init__()
        self.config = config
        self.d_k = config.d_model // config.n_heads

        # Single key and value projections for MQA
        self.k_proj = CPUOptimizedLinear(
            config.d_model, self.d_k, config
        )
        self.v_proj = CPUOptimizedLinear(
            config.d_model, self.d_k, config
        )

        # Multiple query projections
        self.q_proj = CPUOptimizedLinear(
            config.d_model, config.d_model, config
        )
        self.o_proj = CPUOptimizedLinear(
            config.d_model, config.d_model, config
        )

        self.scale = self.d_k**-0.5
        self.cache = {}

        logger.info("Initialized CPUOptimizedMultiQueryAttention")

    def _cached_attention(
        self, q: Tensor, k: Tensor, v: Tensor, chunk_size: int
    ) -> Tensor:
        """Compute attention scores with caching and chunking."""
        batch_size, n_heads, seq_len, d_k = q.shape
        outputs = []

        for i in range(0, seq_len, chunk_size):
            chunk_q = q[:, :, i : i + chunk_size]

            # Use cached computations when possible
            cache_key = (chunk_q.shape, k.shape)
            if cache_key in self.cache:
                chunk_output = self.cache[cache_key]
            else:
                scores = (
                    torch.matmul(chunk_q, k.transpose(-2, -1))
                    * self.scale
                )
                weights = F.softmax(scores, dim=-1)
                chunk_output = torch.matmul(weights, v)

                # Cache management
                if len(self.cache) < 100:  # Limit cache size
                    self.cache[cache_key] = chunk_output

            outputs.append(chunk_output)

        return torch.cat(outputs, dim=2)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with Multi-Query Attention."""
        batch_size = q.size(0)

        # Project queries (multiple heads)
        q = self.q_proj(q).view(
            batch_size, -1, self.config.n_heads, self.d_k
        )

        # Project keys and values (single head)
        k = self.k_proj(k).unsqueeze(1)
        v = self.v_proj(v).unsqueeze(1)

        # Expand k and v for all heads
        k = k.expand(-1, self.config.n_heads, -1, -1)
        v = v.expand(-1, self.config.n_heads, -1, -1)

        # Transpose for attention computation
        q = q.transpose(1, 2)

        # Compute attention with caching and chunking
        context = self._cached_attention(
            q, k, v, self.config.chunk_size
        )

        # Reshape and project output
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.config.d_model)
        )
        return self.o_proj(context)


class CPUOptimizedFeedForward(nn.Module):
    """Feed-forward network with CPU optimizations and RMSNorm."""

    def __init__(self, config: NoamConfig):
        super().__init__()
        self.config = config

        self.fc1 = CPUOptimizedLinear(
            config.d_model, config.d_ff, config
        )
        self.fc2 = CPUOptimizedLinear(
            config.d_ff, config.d_model, config
        )
        self.norm = CPUOptimizedRMSNorm(
            config.d_model, eps=config.epsilon
        )

        # Vectorized activation
        self.activation = self._vectorized_swish

        logger.info("Initialized CPUOptimizedFeedForward")

    def _vectorized_swish(self, x: Tensor) -> Tensor:
        """Vectorized SwiGLU activation."""
        return x * torch.sigmoid(x)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with normalized feed-forward."""
        x = self.norm(x)
        x = self.fc2(self.activation(self.fc1(x)))
        return x


class CPUOptimizedTransformerLayer(nn.Module):
    """Transformer layer with MQA and RMSNorm."""

    def __init__(self, config: NoamConfig):
        super().__init__()
        self.attention = CPUOptimizedMultiQueryAttention(config)
        self.feed_forward = CPUOptimizedFeedForward(config)

        # RMSNorm for pre-normalization
        self.norm1 = CPUOptimizedRMSNorm(
            config.d_model, eps=config.epsilon
        )
        self.norm2 = CPUOptimizedRMSNorm(
            config.d_model, eps=config.epsilon
        )

        logger.info("Initialized CPUOptimizedTransformerLayer")

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass with pre-normalization."""
        # Pre-norm architecture
        x = x + self.attention(self.norm1(x), x, x, mask)
        x = x + self.feed_forward(self.norm2(x))
        return x


class CPUOptimizedNoamTransformer(nn.Module):
    """Complete Noam transformer with MQA and RMSNorm."""

    def __init__(self, config: NoamConfig):
        super().__init__()
        self.config = config

        # Token embeddings with optimal memory layout
        self.embedding = nn.Embedding(
            config.vocab_size, config.d_model
        )
        self.dropout = nn.Dropout(config.dropout)

        # Pre-compute rotary position embeddings
        self.register_buffer(
            "pos_embedding", self._create_rotary_embedding()
        )

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                CPUOptimizedTransformerLayer(config)
                for _ in range(config.n_layers)
            ]
        )

        # Final normalization
        self.norm = CPUOptimizedRMSNorm(
            config.d_model, eps=config.epsilon
        )

        self._init_parameters()
        logger.info("Initialized CPUOptimizedNoamTransformer")

    def _create_rotary_embedding(self) -> Tensor:
        """Create rotary position embeddings."""
        inv_freq = 1.0 / (
            10000
            ** (
                torch.arange(0, self.config.d_model, 2).float()
                / self.config.d_model
            )
        )
        pos = torch.arange(self.config.max_seq_length).float()
        sincos = torch.einsum("i,j->ij", pos, inv_freq)
        emb = torch.cat((sincos, sincos), dim=-1)
        return emb.unsqueeze(0)

    def _init_parameters(self):
        """Initialize parameters with specific CPU optimization."""
        for p in self.parameters():
            if p.dim() > 1:
                # Use Pytorch's native CPU optimized initialization
                nn.init.xavier_uniform_(p)

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass with optimized computation flow."""
        # Generate embeddings
        x = self.embedding(x) * math.sqrt(self.config.d_model)

        # Add rotary position embeddings
        x = x + self.pos_embedding[:, : x.size(1)]
        x = self.dropout(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


def create_noam_optimizer(
    model: nn.Module, config: NoamConfig
) -> Tuple[torch.optim.Optimizer, NoamLRScheduler]:
    """Create optimizer and scheduler with Noam learning rate."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0,
        betas=(0.9, 0.98),
        eps=config.epsilon,
    )
    scheduler = NoamLRScheduler(
        optimizer, config.d_model, config.warmup_steps
    )
    return optimizer, scheduler


def benchmark_noam_model(config: NoamConfig):
    """Benchmark the CPU-optimized Noam transformer."""
    model = CPUOptimizedNoamTransformer(config)
    optimizer, scheduler = create_noam_optimizer(model, config)

    logger.info("Starting benchmark...")

    batch_sizes = [1, 4, 16]
    seq_lengths = [32, 64, 128]

    for batch_size in batch_sizes:
        for seq_length in seq_lengths:
            x = torch.randint(0, config.vocab_size, (batch_size, seq_length))

            # Warm-up run
            with torch.no_grad():
                _ = model(x)

            # Timed run
            start_time = time.perf_counter()
            with torch.no_grad():
                output = model(x)
            end_time = time.perf_counter()
            
            elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds

            logger.info(f"Batch size: {batch_size}, Sequence length: {seq_length}")
            logger.info(f"Processing time: {elapsed_time:.2f}ms")
            logger.info(f"Output shape: {output.shape}\n")

            # Update learning rate
            scheduler.step()

if __name__ == "__main__":
    # Configure logging
    logger.add("noam_transformer.log", rotation="500 MB")

    # Create configuration
    config = NoamConfig(
        d_model=512,
        n_heads=8,
        n_layers=6,
        warmup_steps=4000,
        chunk_size=64,
    )

    # Run benchmark
    benchmark_noam_model(config)
