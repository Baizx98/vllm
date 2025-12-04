import threading
from typing import List, Optional, Tuple

import torch

from vllm.core.transfer.async_transfer_engine import AsyncTransferEngine
from vllm.core.interfaces import BlockSpaceManager
from vllm.core.block.interfaces import Block
from vllm.worker.cache_engine import CacheEngine
from vllm.utils import Device


class AsyncOffloader(AsyncTransferEngine):
    def __init__(
        self,
        block_manager: BlockSpaceManager,
        cache_engine: CacheEngine,
        transfer_unit: int,
    ):
        super().__init__(
            block_manager,
            cache_engine,
            transfer_unit,
            src_device=Device.GPU,
            dst_device=Device.CPU,
            name="AsyncOffloader",
        )

        # 特有的变量
        self._request_layer: Optional[int] = None
        self._abort_event = threading.Event()

        self.start()

    def notify(self, layer: int):
        # notify the offload thread to offload the layer
        print(f"offloader notify layer:{layer}")
        self._abort_event.set()
        with self._condition:
            self._request_layer = layer
            self._condition.notify()

    def shutdown(self):
        self._abort_event.set()  # ✅ 中止当前 transfer 操作

        with self._condition:
            self._shutdown = True
            self._condition.notify()

        if self.transfer_thread:
            self.transfer_thread.join()
        self.event_monitor.unregister()  # ✅ 注销事件监控器
        print(f"🔚 {self.name} shutdown complete.")

    def _should_wait(self) -> bool:
        # Custom condition to wait for offload requests
        return self._request_layer is None and not self._shutdown

    def _get_task(self):
        layer = self._request_layer
        self._request_layer = None
        self._abort_event.clear()
        return layer

    def _transfer(self, task):
        layer = task
        sorted_blocks = self.block_manager.get_layer_blocks_by_importance(layer)
        num_blocks = len(sorted_blocks)
        current_step = 0

        print(f"⬇️ Start offloading layer {layer} with {num_blocks} blocks...")

        while current_step < num_blocks:
            if self._abort_event.is_set() or self._shutdown:
                print(f"🟡 L-{layer} OL interrupted at step {current_step}.")
                return

            blocks = sorted_blocks[
                current_step : current_step + self.transfer_unit
            ]
            if not blocks:
                break

            # plan = self.block_manager.get_offload_plan(blocks)
            plan = self.block_manager.get_transfer_plan(
                blocks, self.src_device, self.dst_device
            )
            print(
                f"🧠 Offload plan for layer {layer} at {current_step}: {plan}"
            )
            self._offload_unit(plan, blocks)
            current_step += self.transfer_unit

        print(f"✅ Offload complete for layer {layer}.")

    def _offload_unit(self, plan: List[Tuple[int, int]], blocks: List[Block]):
        blocks_to_offload = torch.tensor(
            plan, device="cpu", dtype=torch.int64
        ).view(-1, 2)

        def on_transfer_complete():
            self.block_manager.update_blocks_after_transfer(
                plan, blocks, self.src_device, self.dst_device
            )

        self.cache_engine.transfer_blocks_async(
            blocks_to_offload,
            self.src_device,
            self.dst_device,
            self.transfer_stream,  # type: ignore
            callback_fn=on_transfer_complete,
            add_event=self.event_monitor.add_event,
        )
