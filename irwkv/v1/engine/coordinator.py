# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing
import multiprocessing.connection
import time
import weakref

import msgspec.msgpack
import zmq

from irwkv.config import LoggingConfig, ParallelConfig
from irwkv.logger import configure_logging, init_logger
from irwkv.utils.network_utils import make_zmq_socket
from irwkv.utils.system_utils import get_mp_context, set_process_title
from irwkv.v1.engine import EngineCoreOutputs
from irwkv.v1.serial_utils import MsgpackDecoder
from irwkv.v1.utils import get_engine_client_zmq_addr, shutdown

logger = init_logger(__name__)


class DPCoordinator:
    """Publish request queue statistics for independent data-parallel engines."""

    def _wait_for_zmq_addrs(self, zmq_addr_pipe) -> tuple[str, str, str]:
        try:
            timeout = 120
            ready = multiprocessing.connection.wait(
                [zmq_addr_pipe, self.proc.sentinel], timeout=timeout
            )
            if not ready:
                raise RuntimeError(
                    "DP Coordinator process failed to report ZMQ addresses "
                    f"within timeout={timeout} seconds during startup."
                )
            try:
                return zmq_addr_pipe.recv()
            except EOFError:
                raise RuntimeError(
                    "DP Coordinator process failed during startup."
                ) from None
        finally:
            zmq_addr_pipe.close()

    def __init__(
        self,
        parallel_config: ParallelConfig,
        logging_config: LoggingConfig | None = None,
    ):
        dp_size = parallel_config.data_parallel_size
        assert dp_size > 1, "Coordinator only used for data parallel"

        host = parallel_config.data_parallel_master_ip

        # Assume coordinator is colocated with front-end procs when not in
        # either external or hybrid DP LB mode.
        local_only = not parallel_config.local_engines_only
        local_only_eng = dp_size == parallel_config.data_parallel_size_local

        front_publish_address = get_engine_client_zmq_addr(local_only, host=host)
        back_publish_address = get_engine_client_zmq_addr(local_only_eng, host=host)
        back_output_address = get_engine_client_zmq_addr(local_only_eng, host=host)

        context = get_mp_context()
        parent_zmq_addr_pipe, child_zmq_addr_pipe = context.Pipe(duplex=False)
        self.proc: multiprocessing.Process = context.Process(
            target=DPCoordinatorProc.run_coordinator,
            name="IRWKV_DP_Coordinator",
            kwargs={
                "engine_count": parallel_config.data_parallel_size,
                "front_publish_address": front_publish_address,
                "back_output_address": back_output_address,
                "back_publish_address": back_publish_address,
                "zmq_addr_pipe": child_zmq_addr_pipe,
                "logging_config": logging_config,
            },
            daemon=True,
        )
        self.proc.start()
        child_zmq_addr_pipe.close()
        (
            front_publish_address,
            back_output_address,
            back_publish_address,
        ) = self._wait_for_zmq_addrs(parent_zmq_addr_pipe)

        self.stats_publish_address = front_publish_address
        self.coord_in_address = back_publish_address
        self.coord_out_address = back_output_address
        self._finalizer = weakref.finalize(self, shutdown, [self.proc])

    def get_stats_publish_address(self) -> str:
        return self.stats_publish_address

    def get_engine_socket_addresses(self) -> tuple[str, str]:
        """Returns tuple of ZMQ input address, output address."""
        return self.coord_in_address, self.coord_out_address

    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown coordinator process with configurable timeout."""
        if self._finalizer.detach() is not None:
            shutdown([self.proc], timeout=timeout)


class EngineState:
    def __init__(self):
        # [waiting, running, kv_cache_usage]
        self.request_counts: list[int | float] = [0, 0, 0.0]


class DPCoordinatorProc:
    def __init__(
        self,
        engine_count: int,
        min_stats_update_interval_ms: int = 100,
    ):
        set_process_title("DPCoordinator")
        self.ctx = zmq.Context()

        self.engines = [EngineState() for _ in range(engine_count)]

        self.stats_update_interval_ms = min_stats_update_interval_ms

    @staticmethod
    def run_coordinator(
        engine_count: int,
        front_publish_address: str,
        back_output_address: str,
        back_publish_address: str,
        zmq_addr_pipe=None,
        min_stats_update_interval_ms: int = 100,
        logging_config: LoggingConfig | None = None,
    ):
        if logging_config is not None:
            configure_logging(logging_config)

        coordinator = DPCoordinatorProc(
            engine_count=engine_count,
            min_stats_update_interval_ms=min_stats_update_interval_ms,
        )
        try:
            coordinator.process_input_socket(
                front_publish_address,
                back_output_address,
                back_publish_address,
                zmq_addr_pipe,
            )
        except KeyboardInterrupt:
            logger.info("DP Coordinator process exiting")
        finally:
            if zmq_addr_pipe is not None:
                zmq_addr_pipe.close()

    def process_input_socket(
        self,
        front_publish_address: str,
        back_output_address: str,
        back_publish_address: str,
        zmq_addr_pipe=None,
    ):
        decoder = MsgpackDecoder(EngineCoreOutputs)

        stats_changed = False

        with (
            make_zmq_socket(
                path=front_publish_address,  # IPC
                ctx=self.ctx,
                socket_type=zmq.XPUB,
                bind=True,
            ) as publish_front,
            make_zmq_socket(
                path=back_output_address,  # IPC or TCP
                ctx=self.ctx,
                socket_type=zmq.PULL,
                bind=True,
            ) as output_back,
            make_zmq_socket(
                path=back_publish_address,  # IPC or TCP
                ctx=self.ctx,
                socket_type=zmq.XPUB,
                bind=True,
            ) as publish_back,
        ):
            if zmq_addr_pipe is not None:
                try:
                    zmq_addr_pipe.send(
                        (
                            publish_front.getsockopt(zmq.LAST_ENDPOINT).decode(),
                            output_back.getsockopt(zmq.LAST_ENDPOINT).decode(),
                            publish_back.getsockopt(zmq.LAST_ENDPOINT).decode(),
                        )
                    )
                finally:
                    zmq_addr_pipe.close()
            # Wait until all engines subscribe.
            for _ in self.engines:
                if publish_back.recv() != b"\x01":
                    logger.error(
                        "DP Coordinator received unexpected message while "
                        "waiting for engines to subscribe"
                    )
                    return
            # Send ready message to engines.
            publish_back.send(b"READY")

            logger.info("All engine subscriptions received by DP coordinator")

            poller = zmq.Poller()
            poller.register(publish_front, zmq.POLLIN)
            poller.register(publish_back, zmq.POLLIN)
            poller.register(output_back, zmq.POLLIN)
            last_publish_time = 0
            while True:
                elapsed = int(time.time() * 1000) - last_publish_time
                wait_for = self.stats_update_interval_ms if stats_changed else 5000
                events = poller.poll(timeout=max(0, wait_for - elapsed))
                if not events:
                    publish_front.send(
                        msgspec.msgpack.encode(self._get_engine_counts())
                    )
                    stats_changed = False
                    last_publish_time = int(time.time() * 1000)
                    continue

                events = dict(events)
                if publish_front in events:
                    publish_front.recv()
                if publish_back in events:
                    publish_back.recv()
                if output_back in events:
                    outputs = decoder.decode(output_back.recv())
                    assert not outputs.outputs
                    assert outputs.utility_output is None
                    scheduler_stats = outputs.scheduler_stats
                    if scheduler_stats is not None:
                        stats = self.engines[outputs.engine_index].request_counts
                        stats[0] = scheduler_stats.num_waiting_reqs
                        stats[1] = scheduler_stats.num_running_reqs
                        stats[2] = scheduler_stats.kv_cache_usage
                        stats_changed = True

    def _get_engine_counts(self) -> list[list[int | float]]:
        """Return list of [waiting, running] count lists for each engine."""
        return [e.request_counts for e in self.engines]
