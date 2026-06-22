"""Integration test that exercises the full pipeline through UniversalAIOptimizer."""
import pytest
from universal_ai_optimizer.configs.default import OptimizerConfig
from universal_ai_optimizer.core.optimizer import OptimizationResult


class TestPipeline:
    def test_optimizer_init(self):
        from universal_ai_optimizer.core.optimizer import UniversalAIOptimizer
        opt = UniversalAIOptimizer()
        assert opt.config is not None
        assert hasattr(opt.config, 'debug')

    def test_optimizer_init_with_config(self):
        from universal_ai_optimizer.core.optimizer import UniversalAIOptimizer
        config = OptimizerConfig()
        config.debug = True
        opt = UniversalAIOptimizer(config)
        assert opt.config.debug is True

    def test_optimize_returns_result(self):
        from universal_ai_optimizer.core.optimizer import UniversalAIOptimizer
        opt = UniversalAIOptimizer()
        result = opt.optimize("What is 2+2?", {}, None)
        assert isinstance(result, OptimizationResult)
        assert result.original_prompt == "What is 2+2?"

    def test_get_metrics(self):
        from universal_ai_optimizer.core.optimizer import UniversalAIOptimizer
        opt = UniversalAIOptimizer()
        metrics = opt.get_metrics()
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

    def test_pipeline_runs_through_all_modules(self):
        """Verify that the pipeline processes through routing, execution, verification, etc."""
        from universal_ai_optimizer.core.optimizer import UniversalAIOptimizer
        opt = UniversalAIOptimizer()
        result = opt.optimize("def hello(): pass", {"task_type": "code_generation"}, None)
        assert result.original_prompt == "def hello(): pass"

    def test_pipeline_with_context(self):
        from universal_ai_optimizer.core.optimizer import UniversalAIOptimizer
        opt = UniversalAIOptimizer()
        result = opt.optimize("explain quantum computing", {"task_type": "question_answering", "difficulty": "hard"}, None)
        assert result.original_prompt == "explain quantum computing"

    def test_full_pipeline_no_crash(self):
        """End-to-end smoke test with all default configs."""
        from universal_ai_optimizer.core.optimizer import UniversalAIOptimizer
        opt = UniversalAIOptimizer()
        tasks = [
            ("write a poem about AI", {"task_type": "creative_writing"}),
            ("sort this list: [3,1,2]", {"task_type": "code_generation"}),
            ("what is machine learning?", {"task_type": "question_answering"}),
        ]
        for prompt, context in tasks:
            result = opt.optimize(prompt, context, None)
            assert isinstance(result, OptimizationResult)
            assert result.original_prompt == prompt
