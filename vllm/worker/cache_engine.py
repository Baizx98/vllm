"""CacheEngine class for managing the KV cache."""
from typing import List, Union, Callable, Tuple
from queue import Queue

import torch

from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    FakeList,
    Device,
    get_dtype_size,
    is_pin_memory_available,
)

logger = init_logger(__name__)


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device_config = device_config

        self.head_size = model_config.get_head_size()
        # Models like Jamba, have mixed typed layers, E.g Mamba
        self.num_attention_layers = model_config.get_num_attention_layers(
            parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        if self.num_gpu_blocks:
            self.num_gpu_blocks //= parallel_config.pipeline_parallel_size
        self.num_cpu_blocks = cache_config.num_cpu_blocks
        if self.num_cpu_blocks:
            self.num_cpu_blocks //= parallel_config.pipeline_parallel_size

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        self.attn_backend = get_attn_backend(self.head_size,
                                             model_config.dtype,
                                             cache_config.cache_dtype,
                                             self.block_size,
                                             model_config.is_attention_free)
        
        assert self.num_gpu_blocks is not None
        assert self.num_cpu_blocks is not None

        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache(
            self.num_gpu_blocks, self.device_config.device_type)
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")

        # Transfer
        self.offload_data_stream = torch.cuda.Stream()
        self.offload_monitor_queue: Queue[Tuple[torch.cuda.Event, Callable]] = (
            Queue()
        )
        self.prefetch_data_stream = torch.cuda.Stream()
        self.prefetch_monitor_queue: Queue[
            Tuple[torch.cuda.Event, Callable]
        ] = Queue()

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: Union[List[torch.Tensor], FakeList[torch.Tensor]]
        if self.cache_config.enable_layer_wise_block:
            kv_cache = FakeList(
                torch.zeros(kv_cache_shape,
                            dtype=self.dtype,
                            pin_memory=pin_memory,
                            device=device), self.num_attention_layers)
        else:
            kv_cache = []
            for _ in range(self.num_attention_layers):
                # null block in CpuGpuBlockAllocator requires at least that
                # block to be zeroed-out.
                # We zero-out everything for simplicity.
                kv_cache.append(
                    torch.zeros(kv_cache_shape,
                                dtype=self.dtype,
                                pin_memory=pin_memory,
                                device=device))
        return kv_cache  # type: ignore

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        if self.cache_config.enable_layer_wise_block:
            self.attn_backend.swap_blocks(self.cpu_cache, self.gpu_cache, # type: ignore
                                          src_to_dst)  # type: ignore
            return
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                          src_to_dst)

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        if self.cache_config.enable_layer_wise_block:
            self.attn_backend.swap_blocks(self.gpu_cache, self.cpu_cache, # type: ignore
                                          src_to_dst)  # type: ignore
            return
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                          src_to_dst)

    def copy(self, src_to_dsts: torch.Tensor) -> None:
        # TODO need to change this to support layer-wise block?
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_attention_layers = model_config.get_num_attention_layers(
            parallel_config)

        key_cache_block = cache_config.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        if cache_config.enable_layer_wise_block:
            total = key_cache_block + value_cache_block
        else:
            total = num_attention_layers * (key_cache_block +
                                            value_cache_block)
        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        dtype_size = get_dtype_size(dtype)
        return dtype_size * total
    
    def transfer_blocks_async(
        self,
        blocks_to_transfer: torch.Tensor,
        src_device: Device,
        dst_device: Device,
        transfer_stream: torch.cuda.Stream,
        callback_fn: Callable,
        add_event: Callable,
    ):
        """Transfer blocks asynchronously between devices."""
        # TODO: 现在是临时过渡版本，后续根据src和dst设备添加更优雅的处理逻辑
        src_block_ids = blocks_to_transfer[:, 0]
        dst_block_ids = blocks_to_transfer[:, 1]

        src_cache = (
            self.gpu_cache if src_device == Device.GPU else self.cpu_cache
        )
        dst_cache = (
            self.cpu_cache if dst_device == Device.CPU else self.gpu_cache
        )

        # index maybe wrong
        with torch.cuda.stream(stream=transfer_stream):  # type: ignore
            tmp_tensor = src_cache[0][:, src_block_ids, :].contiguous()
            dst_cache[0][:, dst_block_ids, :].copy_(
                tmp_tensor, non_blocking=True
            )
            event: torch.cuda.Event = torch.cuda.Event(blocking=False)  # type: ignore
            event.record(transfer_stream)
            if callback_fn is not None:
                add_event(event, callback_fn)
