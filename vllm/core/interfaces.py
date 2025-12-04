import enum
from abc import ABC, abstractmethod
from typing import List
from typing import Sequence as GenericSequence
from typing import Tuple

from vllm.sequence import Sequence, SequenceGroup
from vllm.utils import Device
from vllm.core.block.interfaces import Block


class AllocStatus(enum.Enum):
    """Result for BlockSpaceManager.can_allocate

    1. Ok: seq_group can be allocated now.
    2. Later: seq_group cannot be allocated.
      The capacity of allocator is larger than seq_group required.
    3. Never: seq_group can never be allocated.
      The seq_group is too large to allocated in GPU.
    """
    OK = enum.auto()
    LATER = enum.auto()
    NEVER = enum.auto()


class BlockSpaceManager(ABC):

    @staticmethod
    def get_block_space_manager_class(version: str):
        version = version.lower()

        if version == "selfattn":
            from vllm.core.block_manager import SelfAttnBlockSpaceManager
            return SelfAttnBlockSpaceManager

        if version == "placeholder":
            from vllm.core.placeholder_block_space_manager import (
                PlaceholderBlockSpaceManager)
            return PlaceholderBlockSpaceManager

        if version == "layer":
            from vllm.core.layer_block_space_maneger import (
                LayerBlockSpaceManager)
            return LayerBlockSpaceManager

        raise ValueError(f"Unknown version {version=}")
    
    @abstractmethod
    def can_allocate_blocks(self, device: Device,num_blocks: int) -> bool:
        pass

    @abstractmethod
    def can_allocate(self,
                     seq_group: SequenceGroup,
                     num_lookahead_slots: int = 0) -> AllocStatus:
        pass

    @abstractmethod
    def allocate(self, seq_group: SequenceGroup) -> None:
        pass

    @abstractmethod
    def can_append_slots(self, seq_group: SequenceGroup,
                         num_lookahead_slots: int) -> bool:
        pass

    @abstractmethod
    def append_slots(
        self,
        seq: Sequence,
        num_lookahead_slots: int,
    ) -> List[Tuple[int, int]]:
        pass

    @abstractmethod
    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        pass

    @abstractmethod
    def can_swap_in(self, seq_group: SequenceGroup,
                    num_lookahead_slots: int) -> AllocStatus:
        pass

    @abstractmethod
    def swap_in(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        pass

    @abstractmethod
    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        pass

    @abstractmethod
    def swap_out(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        pass

    @abstractmethod
    def free(self, seq: Sequence) -> None:
        pass

    @abstractmethod
    def get_block_table(self, seq: Sequence) -> List[int]:
        pass

    @abstractmethod
    def get_num_free_gpu_blocks(self) -> int:
        pass

    @abstractmethod
    def get_num_free_cpu_blocks(self) -> int:
        pass

    @abstractmethod
    def access_all_blocks_in_seq(
        self,
        seq: Sequence,
        access_time: float,
    ) -> None:
        pass

    @abstractmethod
    def get_common_computed_block_ids(
            self, seqs: List[Sequence]) -> GenericSequence[int]:
        pass

    @abstractmethod
    def mark_blocks_as_computed(self, seq_group: SequenceGroup,
                                token_chunk_size: int):
        pass

    @abstractmethod
    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        """Prefix cache hit rate. -1 means not supported or disabled."""
        pass

    @abstractmethod
    def can_allocate_block_ids(self, device: Device, num_blocks: int) ->bool:
        pass

    @abstractmethod
    def allocate_block_id(self, device: Device) -> int:
        pass

    @abstractmethod
    def free_block_id(self, device: Device, block_id: int) -> None:
        pass

    @abstractmethod
    def get_device_and_pid(self, block_id: int) -> Tuple[Device, int]:
        pass

    @abstractmethod
    def get_gid(self, device: Device, pid: int) -> int:
        pass

    @abstractmethod
    def is_device_block(self, block_id:int,device:Device) -> bool:
        pass

    @abstractmethod
    def get_layer_blocks_by_importance(self, layer:int)-> List[Block]:
        pass

    @abstractmethod
    def predict_next_layer_needed_blocks(self, layer: int) -> List[Block]:
        pass

    @abstractmethod
    def get_transfer_plan(
        self, blocks: List[Block], src_device: Device, dst_device: Device
    ) -> List[Tuple[int, int]]:
        pass

    @abstractmethod
    def update_blocks_after_transfer(
        self,
        plan: List[Tuple[int, int]],
        original_blocks: List[Block],
        src_device: Device,
        dst_device: Device,
    ) -> None:
        pass

    @abstractmethod
    def kv_cache_ready(self, batch: List[Sequence], layer: int) -> bool:
        pass

    @abstractmethod
    def wait_for_kv_cache_ready(self, batch: List[Sequence], layer: int) -> None:
        pass

    @property
    @abstractmethod
    def watermark(self) -> float:
        pass

    @abstractmethod
    def free_block_num(self, device: Device) -> int:
        pass

    @property
    @abstractmethod
    def num_gpu_blocks(self) -> int:
        pass

    @property
    @abstractmethod
    def num_attn_layers(self) -> int:
        pass