import pytest


class TestEpisodicMemory:
    def test_init_with_capacity(self):
        from universal_ai_optimizer.modules.memory_manager import EpisodicMemory
        mem = EpisodicMemory(capacity=5)
        assert mem.capacity == 5
        assert len(mem.memories) == 0

    def test_add_memory(self):
        from universal_ai_optimizer.modules.memory_manager import EpisodicMemory
        mem = EpisodicMemory(capacity=5)
        mem.add("test_entry")
        assert len(mem.memories) == 1

    def test_capacity_enforced(self):
        from universal_ai_optimizer.modules.memory_manager import EpisodicMemory
        mem = EpisodicMemory(capacity=3)
        for i in range(5):
            mem.add(f"entry_{i}")
        assert len(mem.memories) == 3
        items = list(mem.memories)
        assert "entry_0" not in items
        assert "entry_4" in items

    def test_get_all(self):
        from universal_ai_optimizer.modules.memory_manager import EpisodicMemory
        mem = EpisodicMemory(capacity=10)
        for i in range(3):
            mem.add(f"entry_{i}")
        items = mem.get_all()
        assert len(items) == 3


class TestSemanticMemory:
    def test_init_with_capacity(self):
        from universal_ai_optimizer.modules.memory_manager import SemanticMemory
        mem = SemanticMemory(capacity=5)
        assert mem.capacity == 5

    def test_add_and_retrieve(self):
        from universal_ai_optimizer.modules.memory_manager import SemanticMemory
        mem = SemanticMemory(capacity=10)
        mem.add({"key": "concept", "info": "test"})
        items = mem.get_all()
        assert len(items) == 1

    def test_capacity_enforced(self):
        from universal_ai_optimizer.modules.memory_manager import SemanticMemory
        mem = SemanticMemory(capacity=3)
        for i in range(5):
            mem.add({"key": f"concept_{i}"})
        assert len(mem.memories) <= 3

    def test_clear_reassign(self):
        from universal_ai_optimizer.modules.memory_manager import SemanticMemory
        mem = SemanticMemory(capacity=10)
        mem.add({"key": "test"})
        mem.memories = []
        assert len(mem.memories) == 0


class TestWorkingMemory:
    def test_init_with_capacity(self):
        from universal_ai_optimizer.modules.memory_manager import WorkingMemory
        mem = WorkingMemory(capacity=5)
        assert len(mem.memories) == 0

    def test_add(self):
        from universal_ai_optimizer.modules.memory_manager import WorkingMemory
        mem = WorkingMemory(capacity=5)
        mem.add("item")
        assert len(mem.memories) == 1

    def test_capacity_enforced(self):
        from universal_ai_optimizer.modules.memory_manager import WorkingMemory
        mem = WorkingMemory(capacity=2)
        mem.add("a")
        mem.add("b")
        mem.add("c")
        assert len(mem.memories) == 2
        items = list(mem.memories)
        assert "a" not in items
        assert "c" in items

    def test_get_all(self):
        from universal_ai_optimizer.modules.memory_manager import WorkingMemory
        mem = WorkingMemory(capacity=10)
        mem.add("x")
        mem.add("y")
        assert mem.get_all() == ["x", "y"]


class TestLongTermMemory:
    def test_init_with_capacity(self):
        from universal_ai_optimizer.modules.memory_manager import LongTermMemory
        mem = LongTermMemory(capacity=5)
        assert mem.capacity == 5
        assert len(mem.memories) == 0

    def test_add_and_retrieve(self):
        from universal_ai_optimizer.modules.memory_manager import LongTermMemory
        mem = LongTermMemory(capacity=10)
        mem.add({"key": "k1", "data": "value"})
        items = mem.get_all()
        assert len(items) == 1

    def test_capacity_enforced(self):
        from universal_ai_optimizer.modules.memory_manager import LongTermMemory
        mem = LongTermMemory(capacity=2)
        for i in range(4):
            mem.add({"key": f"k{i}"})
        assert len(mem.memories) <= 2


class TestMemoryManager:
    def test_init_with_defaults(self):
        from universal_ai_optimizer.modules.memory_manager import MemoryManager
        mm = MemoryManager()
        assert mm.enabled is True
        assert mm.consolidation_threshold == 0.8

    def test_init_with_config(self):
        from universal_ai_optimizer.modules.memory_manager import MemoryManager
        mm = MemoryManager({"consolidation_threshold": 0.5})
        assert mm.consolidation_threshold == 0.5

    def test_process_disabled(self):
        from universal_ai_optimizer.modules.memory_manager import MemoryManager
        mm = MemoryManager({"enabled": False})
        result = mm.process("prompt", {}, None, {})
        assert result == {}

    def test_process_adds_to_episodic(self):
        from universal_ai_optimizer.modules.memory_manager import MemoryManager
        mm = MemoryManager()
        result = mm.process("test prompt", {"task_type": "test"}, None, {})
        assert "memory_stats" in result
        assert "stored_memory_id" in result

    def test_get_metrics(self):
        from universal_ai_optimizer.modules.memory_manager import MemoryManager
        mm = MemoryManager()
        metrics = mm.get_metrics()
        assert metrics["module"] == "MemoryManager"

    def test_lock_used(self):
        """Ensure RLock doesn't deadlock on recursive calls."""
        from universal_ai_optimizer.modules.memory_manager import MemoryManager
        mm = MemoryManager()
        with mm._lock:
            with mm._lock:
                assert True
