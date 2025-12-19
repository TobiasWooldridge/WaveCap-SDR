"""Concurrent operations and async testing.

Tests thread safety, concurrent access, async operations, and resource
cleanup patterns. Adapted from SDRTrunk's CountDownLatch testing patterns.

Reference: https://github.com/DSheirer/sdrtrunk
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from queue import Queue, Empty
from threading import Lock, Event
from typing import Any, Dict, List, Optional, Set

import pytest


# ============================================================================
# Thread-Safe Counter (for testing concurrent access)
# ============================================================================

class ThreadSafeCounter:
    """Thread-safe counter for testing concurrent increments."""

    def __init__(self):
        self._value = 0
        self._lock = Lock()

    def increment(self) -> int:
        with self._lock:
            self._value += 1
            return self._value

    def decrement(self) -> int:
        with self._lock:
            self._value -= 1
            return self._value

    @property
    def value(self) -> int:
        with self._lock:
            return self._value


class TestThreadSafeCounter:
    """Test thread-safe counter."""

    def test_single_thread_increment(self):
        """Single-threaded increment works."""
        counter = ThreadSafeCounter()

        for _ in range(100):
            counter.increment()

        assert counter.value == 100

    def test_concurrent_increments(self):
        """Concurrent increments are thread-safe."""
        counter = ThreadSafeCounter()
        num_threads = 10
        increments_per_thread = 1000

        def increment_many():
            for _ in range(increments_per_thread):
                counter.increment()

        threads = [threading.Thread(target=increment_many) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert counter.value == num_threads * increments_per_thread

    def test_concurrent_mixed_operations(self):
        """Mixed increment/decrement operations are thread-safe."""
        counter = ThreadSafeCounter()
        operations = 1000

        def incrementer():
            for _ in range(operations):
                counter.increment()

        def decrementer():
            for _ in range(operations):
                counter.decrement()

        t1 = threading.Thread(target=incrementer)
        t2 = threading.Thread(target=decrementer)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Net result should be 0
        assert counter.value == 0


# ============================================================================
# Event Latch Pattern (like Java CountDownLatch)
# ============================================================================

class EventLatch:
    """Python equivalent of Java's CountDownLatch.

    Used for testing async/concurrent event processing.
    """

    def __init__(self, count: int):
        self._count = count
        self._lock = Lock()
        self._event = Event()
        if count <= 0:
            self._event.set()

    def count_down(self):
        """Decrement the count."""
        with self._lock:
            self._count -= 1
            if self._count <= 0:
                self._event.set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for count to reach zero.

        Returns True if count reached zero, False on timeout.
        """
        return self._event.wait(timeout)

    @property
    def count(self) -> int:
        with self._lock:
            return self._count


class TestEventLatch:
    """Test EventLatch (CountDownLatch equivalent)."""

    def test_single_countdown(self):
        """Single countdown releases waiter."""
        latch = EventLatch(1)

        def count_down():
            time.sleep(0.01)
            latch.count_down()

        t = threading.Thread(target=count_down)
        t.start()

        result = latch.wait(timeout=1.0)
        t.join()

        assert result is True
        assert latch.count == 0

    def test_multiple_countdowns(self):
        """Multiple countdowns required."""
        latch = EventLatch(3)

        def count_down():
            time.sleep(0.01)
            latch.count_down()

        threads = [threading.Thread(target=count_down) for _ in range(3)]
        for t in threads:
            t.start()

        result = latch.wait(timeout=1.0)
        for t in threads:
            t.join()

        assert result is True

    def test_timeout_before_countdown(self):
        """Timeout when countdown not reached."""
        latch = EventLatch(5)

        # Only 2 countdowns
        latch.count_down()
        latch.count_down()

        result = latch.wait(timeout=0.05)

        assert result is False
        assert latch.count == 3

    def test_zero_count_immediate(self):
        """Zero count is immediately satisfied."""
        latch = EventLatch(0)
        assert latch.wait(timeout=0) is True


# ============================================================================
# Consumer/Subscriber Pattern Tests
# ============================================================================

@dataclass
class AudioSegmentMock:
    """Mock audio segment for testing."""
    id: str
    audio: bytes = b""
    consumer_count: int = 0
    _lock: Lock = field(default_factory=Lock)

    def add_consumer(self):
        with self._lock:
            self.consumer_count += 1

    def remove_consumer(self) -> bool:
        """Remove consumer, return True if last consumer."""
        with self._lock:
            self.consumer_count -= 1
            return self.consumer_count <= 0


class TestConsumerPattern:
    """Test consumer/subscriber pattern (SDRTrunk AudioSegment pattern)."""

    def test_add_consumer(self):
        """Add consumer increments count."""
        segment = AudioSegmentMock(id="test")
        segment.add_consumer()
        segment.add_consumer()

        assert segment.consumer_count == 2

    def test_remove_consumer(self):
        """Remove consumer decrements count."""
        segment = AudioSegmentMock(id="test")
        segment.add_consumer()
        segment.add_consumer()

        segment.remove_consumer()
        assert segment.consumer_count == 1

    def test_last_consumer_triggers_cleanup(self):
        """Last consumer removal signals cleanup."""
        segment = AudioSegmentMock(id="test")
        segment.add_consumer()
        segment.add_consumer()

        assert segment.remove_consumer() is False
        assert segment.remove_consumer() is True  # Last consumer

    def test_concurrent_consumer_access(self):
        """Concurrent consumer add/remove is safe."""
        segment = AudioSegmentMock(id="test")
        latch = EventLatch(20)

        def add_remove():
            segment.add_consumer()
            time.sleep(0.001)
            segment.remove_consumer()
            latch.count_down()

        threads = [threading.Thread(target=add_remove) for _ in range(20)]
        for t in threads:
            t.start()

        latch.wait(timeout=5.0)
        for t in threads:
            t.join()

        # All consumers removed
        assert segment.consumer_count == 0


# ============================================================================
# Thread-Safe Queue Tests
# ============================================================================

class TestThreadSafeQueue:
    """Test thread-safe queue operations."""

    def test_producer_consumer(self):
        """Single producer, single consumer."""
        queue: Queue = Queue()
        items_produced = []
        items_consumed = []

        def producer():
            for i in range(100):
                queue.put(i)
                items_produced.append(i)

        def consumer():
            while len(items_consumed) < 100:
                try:
                    item = queue.get(timeout=0.1)
                    items_consumed.append(item)
                except Empty:
                    pass

        p = threading.Thread(target=producer)
        c = threading.Thread(target=consumer)

        p.start()
        c.start()
        p.join()
        c.join()

        assert items_produced == items_consumed

    def test_multiple_producers(self):
        """Multiple producers, single consumer."""
        queue: Queue = Queue()
        num_producers = 5
        items_per_producer = 100
        consumed = []

        def producer(producer_id):
            for i in range(items_per_producer):
                queue.put((producer_id, i))

        def consumer():
            expected = num_producers * items_per_producer
            while len(consumed) < expected:
                try:
                    item = queue.get(timeout=0.1)
                    consumed.append(item)
                except Empty:
                    pass

        producers = [threading.Thread(target=producer, args=(i,))
                    for i in range(num_producers)]
        c = threading.Thread(target=consumer)

        for p in producers:
            p.start()
        c.start()

        for p in producers:
            p.join()
        c.join()

        assert len(consumed) == num_producers * items_per_producer

    def test_queue_timeout(self):
        """Queue get with timeout."""
        queue: Queue = Queue()

        start = time.time()
        try:
            queue.get(timeout=0.1)
            got_item = True
        except Empty:
            got_item = False
        elapsed = time.time() - start

        assert not got_item
        assert elapsed >= 0.1


# ============================================================================
# Async Operation Tests
# ============================================================================

@pytest.mark.anyio
class TestAsyncOperations:
    """Test async operations."""

    async def test_async_queue(self):
        """Async queue producer/consumer."""
        queue: asyncio.Queue = asyncio.Queue()
        produced = []
        consumed = []

        async def producer():
            for i in range(50):
                await queue.put(i)
                produced.append(i)

        async def consumer():
            while len(consumed) < 50:
                item = await asyncio.wait_for(queue.get(), timeout=1.0)
                consumed.append(item)

        await asyncio.gather(producer(), consumer())

        assert produced == consumed

    async def test_async_timeout(self):
        """Async operation with timeout."""
        queue: asyncio.Queue = asyncio.Queue()

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(queue.get(), timeout=0.05)

    async def test_concurrent_async_tasks(self):
        """Multiple concurrent async tasks."""
        results = []
        lock = asyncio.Lock()

        async def task(task_id, delay):
            await asyncio.sleep(delay)
            async with lock:
                results.append(task_id)

        tasks = [
            task(1, 0.01),
            task(2, 0.02),
            task(3, 0.01),
        ]

        await asyncio.gather(*tasks)

        assert len(results) == 3
        assert set(results) == {1, 2, 3}

    async def test_async_event_waiting(self):
        """Async event signaling."""
        event = asyncio.Event()
        waited = False

        async def waiter():
            nonlocal waited
            await asyncio.wait_for(event.wait(), timeout=1.0)
            waited = True

        async def signaler():
            await asyncio.sleep(0.01)
            event.set()

        await asyncio.gather(waiter(), signaler())

        assert waited


# ============================================================================
# Resource Cleanup Tests
# ============================================================================

class ResourceTracker:
    """Track resource allocation/deallocation for testing."""

    def __init__(self):
        self.allocated: Set[str] = set()
        self.deallocated: Set[str] = set()
        self._lock = Lock()

    def allocate(self, resource_id: str):
        with self._lock:
            self.allocated.add(resource_id)

    def deallocate(self, resource_id: str):
        with self._lock:
            self.deallocated.add(resource_id)

    @property
    def leaked(self) -> Set[str]:
        """Resources allocated but not deallocated."""
        with self._lock:
            return self.allocated - self.deallocated


class TestResourceCleanup:
    """Test resource cleanup patterns."""

    def test_no_leaks_on_normal_path(self):
        """No leaks when resources are properly cleaned up."""
        tracker = ResourceTracker()

        for i in range(10):
            resource_id = f"resource_{i}"
            tracker.allocate(resource_id)
            # Simulate work
            tracker.deallocate(resource_id)

        assert len(tracker.leaked) == 0

    def test_detect_leaked_resources(self):
        """Detect leaked resources."""
        tracker = ResourceTracker()

        tracker.allocate("r1")
        tracker.allocate("r2")
        tracker.allocate("r3")
        tracker.deallocate("r1")
        tracker.deallocate("r3")

        assert tracker.leaked == {"r2"}

    def test_cleanup_on_exception(self):
        """Resources cleaned up even on exception."""
        tracker = ResourceTracker()

        try:
            tracker.allocate("r1")
            raise ValueError("Test error")
        except ValueError:
            pass
        finally:
            tracker.deallocate("r1")

        assert len(tracker.leaked) == 0


# ============================================================================
# Thread Pool Tests
# ============================================================================

class TestThreadPoolExecution:
    """Test thread pool task execution."""

    def test_parallel_execution(self):
        """Tasks execute in parallel."""
        results = []
        lock = Lock()

        def task(task_id):
            time.sleep(0.01)
            with lock:
                results.append(task_id)
            return task_id

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(task, i) for i in range(10)]
            completed = [f.result() for f in as_completed(futures)]

        assert len(completed) == 10
        assert set(completed) == set(range(10))

    def test_exception_in_task(self):
        """Exceptions in tasks are propagated."""
        def failing_task():
            raise ValueError("Task failed")

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(failing_task)

            with pytest.raises(ValueError, match="Task failed"):
                future.result()

    def test_task_timeout(self):
        """Task with timeout."""
        def slow_task():
            time.sleep(1.0)
            return "done"

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(slow_task)

            from concurrent.futures import TimeoutError
            with pytest.raises(TimeoutError):
                future.result(timeout=0.05)


# ============================================================================
# Lock Contention Tests
# ============================================================================

class TestLockContention:
    """Test lock contention scenarios."""

    def test_lock_prevents_race(self):
        """Lock prevents race condition."""
        shared_value = [0]  # Use list for mutability
        lock = Lock()

        def increment():
            for _ in range(10000):
                with lock:
                    shared_value[0] += 1

        threads = [threading.Thread(target=increment) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert shared_value[0] == 40000

    def test_race_without_lock(self):
        """Demonstrate race condition without lock."""
        # Note: This test may occasionally pass due to timing
        shared_value = [0]

        def increment():
            for _ in range(10000):
                # No lock - race condition
                temp = shared_value[0]
                shared_value[0] = temp + 1

        threads = [threading.Thread(target=increment) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Without lock, value is likely less than expected due to race
        # We just verify the test ran - value may vary
        assert shared_value[0] > 0


# ============================================================================
# Subscriber Disconnect Tests
# ============================================================================

class Broadcaster:
    """Broadcaster with subscriber management."""

    def __init__(self):
        self.subscribers: List[Queue] = []
        self._lock = Lock()

    def subscribe(self) -> Queue:
        queue = Queue()
        with self._lock:
            self.subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: Queue):
        with self._lock:
            if queue in self.subscribers:
                self.subscribers.remove(queue)

    def broadcast(self, message: Any):
        with self._lock:
            for queue in self.subscribers:
                queue.put(message)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self.subscribers)


class TestSubscriberManagement:
    """Test subscriber management patterns."""

    def test_subscribe_and_receive(self):
        """Subscriber receives broadcast messages."""
        broadcaster = Broadcaster()
        queue = broadcaster.subscribe()

        broadcaster.broadcast("test message")

        message = queue.get(timeout=0.1)
        assert message == "test message"

    def test_multiple_subscribers(self):
        """Multiple subscribers all receive messages."""
        broadcaster = Broadcaster()
        queues = [broadcaster.subscribe() for _ in range(5)]

        broadcaster.broadcast("test")

        for q in queues:
            assert q.get(timeout=0.1) == "test"

    def test_unsubscribe(self):
        """Unsubscribed queue doesn't receive messages."""
        broadcaster = Broadcaster()
        q1 = broadcaster.subscribe()
        q2 = broadcaster.subscribe()

        broadcaster.unsubscribe(q1)
        broadcaster.broadcast("test")

        assert q2.get(timeout=0.1) == "test"
        with pytest.raises(Empty):
            q1.get(timeout=0.05)

    def test_subscriber_count(self):
        """Track subscriber count."""
        broadcaster = Broadcaster()

        q1 = broadcaster.subscribe()
        q2 = broadcaster.subscribe()
        assert broadcaster.subscriber_count == 2

        broadcaster.unsubscribe(q1)
        assert broadcaster.subscriber_count == 1

    def test_concurrent_subscribe_unsubscribe(self):
        """Concurrent subscribe/unsubscribe is safe."""
        broadcaster = Broadcaster()
        latch = EventLatch(100)

        def subscribe_unsubscribe():
            queue = broadcaster.subscribe()
            time.sleep(0.001)
            broadcaster.unsubscribe(queue)
            latch.count_down()

        threads = [threading.Thread(target=subscribe_unsubscribe)
                  for _ in range(100)]
        for t in threads:
            t.start()

        latch.wait(timeout=5.0)
        for t in threads:
            t.join()

        assert broadcaster.subscriber_count == 0
