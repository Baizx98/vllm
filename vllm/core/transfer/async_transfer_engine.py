import threading
import torch
from typing import List
from vllm.worker.cache_engine import CacheEngine
from vllm.core.block.interfaces import Block
from vllm.core.interfaces import BlockSpaceManager
from vllm.core.transfer.event_monitor import EventMonitor
from vllm.utils import Device
from vllm.utils import is_pin_memory_available

class AsyncTransferEngine:
    def __init__(
        self,
        block_manager: BlockSpaceManager,
        cache_engine: CacheEngine,
        transfer_unit: int,
        src_device: Device,
        dst_device: Device,
        name: str,
    ) -> None:
        self.block_manager = block_manager
        self.cache_engine = cache_engine
        self.transfer_unit = transfer_unit
        self.name = name

        self.src_device = src_device
        self.dst_device = dst_device

        self._condition = threading.Condition()
        self._shutdown = False

        self.event_monitor = EventMonitor()
        # TODO monitor queue 的使用方法应该重新考虑，到底是用注册还是函数传递呢

        self.transfer_stream = torch.cuda.Stream()

        self.transfer_thread = threading.Thread(
            target=self._run, name=f"{name}_thread"
        )

    def start(self):
        """Start the transfer and monitor threads."""
        self.transfer_thread.start()

    def shutdown(self):
        with self._condition:
            self._shutdown = True
            self._condition.notify()
        if hasattr(self, "transfer_thread") and self.transfer_thread:
            self.transfer_thread.join()
        self.event_monitor.unregister()
        print(f"🔚 {getattr(self, 'name', 'AsyncTransferEngine')} shutdown.")

    def update_transfer_unit(self, num_blocks: int, current_step: int) -> int:
        if is_pin_memory_available():
            # self.transfer_unit = min(
            #     self.block_manager.num_gpu_blocks, self.transfer_unit * 2
            # )
            pass
        else:
            self.transfer_unit = max(1, self.transfer_unit // 2)
        self.transfer_unit = min(self.transfer_unit, num_blocks - current_step)
        return self.transfer_unit

    def notify(self, layer: int):
        raise NotImplementedError(
            "Subclasses must implement the notify method."
        )

    # 自定义的判断条件
    def _should_wait(self) -> bool:
        return not self._shutdown

    def _get_task(self):
        raise NotImplementedError

    def _transfer(self, task):
        raise NotImplementedError

    def _run(self):
        while True:
            with self._condition:
                while self._should_wait():
                    self._condition.wait()
                if self._shutdown:
                    break
                task = self._get_task()

            if task is not None:
                self._transfer(task)

    def _transfer_unit(self, plan: List[tuple[int, int]], blocks: List[Block]):
        """
        Transfer a unit of blocks according to the offload plan.
        This method should be implemented in subclasses.
        """
        blocks_to_transfer = torch.tensor(
            plan, device="cpu", dtype=torch.int64
        ).view(-1, 2)

        def on_transfer_complete():
            # TODO 该函数尚未完成，还不能使用transfer unit函数
            self.block_manager.update_blocks_after_transfer(
                plan, blocks, self.src_device, self.dst_device
            )

        self.cache_engine.transfer_blocks_async(
            blocks_to_transfer,
            self.src_device,
            self.dst_device,
            self.transfer_stream,  # type: ignore
            callback_fn=on_transfer_complete,
            add_event=self.event_monitor.add_event,
        )
