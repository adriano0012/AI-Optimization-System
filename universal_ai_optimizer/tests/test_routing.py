import pytest


class TestLearningRouter:
    def test_init_with_defaults(self):
        from universal_ai_optimizer.modules.routing.learning_router import LearningRouter
        router = LearningRouter()
        assert router.enabled is True
        assert router.routing_type == "model"
        assert router.exploration_rate == 0.1
        assert router.learning_rate == 0.01

    def test_init_with_config(self, routing_config):
        from universal_ai_optimizer.modules.routing.learning_router import LearningRouter
        router = LearningRouter(routing_config)
        assert router.exploration_rate == 0.1
        assert len(router.available_options) == 3

    def test_route_empty_disabled(self):
        from universal_ai_optimizer.modules.routing.learning_router import LearningRouter
        router = LearningRouter({"enabled": False})
        assert router.route({"task_type": "test"}) is None

    def test_route_with_disabled_returns_default(self):
        from universal_ai_optimizer.modules.routing.learning_router import LearningRouter
        router = LearningRouter({"enabled": False, "available_options": ["gpt-4"]})
        assert router.route({"task_type": "test"}) == "gpt-4"

    def test_update_performance(self, routing_config):
        from universal_ai_optimizer.modules.routing.learning_router import LearningRouter
        router = LearningRouter(routing_config)
        router.update_performance("gpt-4", {"task_type": "code_generation"}, 0.9)
        key = ("gpt-4", "code_generation")
        assert router.action_counts[key] == 1
        assert router.total_rewards[key] == 0.9
        assert router.average_rewards[key] == 0.9

    def test_update_performance_multiple(self, routing_config):
        from universal_ai_optimizer.modules.routing.learning_router import LearningRouter
        router = LearningRouter(routing_config)
        for _ in range(5):
            router.update_performance("gpt-4", {"task_type": "code_generation"}, 1.0)
        key = ("gpt-4", "code_generation")
        assert router.action_counts[key] == 5
        assert router.average_rewards[key] == 1.0

    def test_update_performance_disabled(self):
        from universal_ai_optimizer.modules.routing.learning_router import LearningRouter
        router = LearningRouter({"enabled": False})
        router.update_performance("gpt-4", {"task_type": "test"}, 1.0)
        assert len(router.action_counts) == 0

    def test_get_performance_summary(self, routing_config):
        from universal_ai_optimizer.modules.routing.learning_router import LearningRouter
        router = LearningRouter(routing_config)
        router.update_performance("gpt-4", {"task_type": "code_generation"}, 0.8)
        router.update_performance("claude-3", {"task_type": "creative_writing"}, 0.9)
        summary = router.get_performance_summary()
        assert "gpt-4" in summary
        assert "claude-3" in summary
        assert summary["gpt-4"]["code_generation"]["average_reward"] == 0.8
        assert summary["claude-3"]["creative_writing"]["average_reward"] == 0.9

    def test_route_returns_option(self, routing_config):
        from universal_ai_optimizer.modules.routing.learning_router import LearningRouter
        router = LearningRouter(routing_config)
        result = router.route({"task_type": "code_generation", "difficulty": "medium", "prompt": "test"})
        assert result in routing_config["available_options"]

    def test_get_metrics(self, routing_config):
        from universal_ai_optimizer.modules.routing.learning_router import LearningRouter
        router = LearningRouter(routing_config)
        metrics = router.get_metrics()
        assert metrics["enabled"] is True
        assert metrics["routing_type"] == "model"
        assert metrics["available_options_count"] == 3

    def test_thompson_sampling_select(self, routing_config):
        from universal_ai_optimizer.modules.routing.learning_router import LearningRouter
        from collections import defaultdict
        # Update all 3 options with both alpha > 0 and beta > 0 to avoid beta(<=0) errors
        config = {**routing_config, "available_options": ["gpt-4", "claude-3", "gemini-pro"]}
        router = LearningRouter(config)
        # Manually set alpha and beta to ensure both > 0 for all options
        for opt in router.available_options:
            router.alpha[opt] = defaultdict(float)
            router.beta[opt] = defaultdict(float)
            router.alpha[opt]["code_generation"] = 2.0
            router.beta[opt]["code_generation"] = 2.0
        result = router._thompson_sampling_select({"task_type": "code_generation"})
        assert result in ["gpt-4", "claude-3", "gemini-pro"]

    def test_save_state(self, routing_config, tmp_path):
        from universal_ai_optimizer.modules.routing.learning_router import LearningRouter
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            router = LearningRouter(routing_config)
            router.update_performance("gpt-4", {"task_type": "code_generation"}, 0.9)
            router.save_state()
            assert os.path.exists("learning_router/state.json")
        finally:
            os.chdir(original_cwd)

    def test_extract_task_features(self, routing_config):
        from universal_ai_optimizer.modules.routing.learning_router import LearningRouter
        router = LearningRouter(routing_config)
        features = router._extract_task_features({
            "task_type": "code_generation",
            "difficulty": "hard",
            "prompt": "def hello(): pass",
            "context": {"lang": "python"}
        })
        assert len(features) == 10

    def test_process_returns_empty(self, routing_config):
        from universal_ai_optimizer.modules.routing.learning_router import LearningRouter
        router = LearningRouter(routing_config)
        result = router.process("hello", {}, None, None)
        assert result == {}


class TestSemanticRouter:
    def test_init_with_defaults(self):
        from universal_ai_optimizer.modules.routing.semantic_router import SemanticRouter
        router = SemanticRouter()
        assert router.enabled is True
        assert router.similarity_threshold == 0.7

    def test_init_with_categories(self):
        from universal_ai_optimizer.modules.routing.semantic_router import SemanticRouter
        router = SemanticRouter({
            "categories": {
                "tech": {"description": "Technology and programming topics"}
            }
        })
        assert "tech" in router.categories

    def test_process_disabled(self):
        from universal_ai_optimizer.modules.routing.semantic_router import SemanticRouter
        router = SemanticRouter({"enabled": False})
        assert router.process("hello", {}, None, None) == {}

    def test_process_without_categories(self):
        from universal_ai_optimizer.modules.routing.semantic_router import SemanticRouter
        router = SemanticRouter()
        result = router.process("hello world", {}, None, None)
        assert result["semantic_route"] is None

    def test_process_with_category_match(self):
        from universal_ai_optimizer.modules.routing.semantic_router import SemanticRouter
        router = SemanticRouter({
            "categories": {
                "tech": {"description": "Technology and programming topics"},
                "science": {"description": "Scientific discussions"},
            }
        })
        result = router.process("python programming", {}, None, None)
        assert "semantic_route" in result
        assert "semantic_similarity" in result

    def test_add_category(self):
        from universal_ai_optimizer.modules.routing.semantic_router import SemanticRouter
        router = SemanticRouter()
        router.add_category("new_cat", "A new category")
        assert "new_cat" in router.categories
        assert router.categories["new_cat"]["description"] == "A new category"

    def test_add_category_with_preferred_models(self):
        from universal_ai_optimizer.modules.routing.semantic_router import SemanticRouter
        router = SemanticRouter()
        router.add_category("new_cat", "A new category", preferred_models=["gpt-4"])
        assert router.categories["new_cat"]["preferred_models"] == ["gpt-4"]

    def test_get_metrics(self):
        from universal_ai_optimizer.modules.routing.semantic_router import SemanticRouter
        router = SemanticRouter()
        metrics = router.get_metrics()
        assert metrics["enabled"] is True
        assert metrics["similarity_threshold"] == 0.7

    def test_cosine_similarity_identical(self):
        from universal_ai_optimizer.modules.routing.semantic_router import SemanticRouter
        router = SemanticRouter()
        v = [1.0, 0.0, 0.0]
        assert router._cosine_similarity(v, v) == 1.0

    def test_cosine_similarity_orthogonal(self):
        from universal_ai_optimizer.modules.routing.semantic_router import SemanticRouter
        router = SemanticRouter()
        assert router._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_get_embedding_returns_384d(self):
        from universal_ai_optimizer.modules.routing.semantic_router import SemanticRouter
        router = SemanticRouter()
        emb = router._get_embedding("test")
        assert len(emb) == 384

    def test_get_embedding_normalized(self):
        from universal_ai_optimizer.modules.routing.semantic_router import SemanticRouter
        import math
        router = SemanticRouter()
        emb = router._get_embedding("hello world test")
        norm = math.sqrt(sum(v * v for v in emb))
        assert abs(norm - 1.0) < 1e-6

    def test_get_embedding_empty_string(self):
        from universal_ai_optimizer.modules.routing.semantic_router import SemanticRouter
        import math
        router = SemanticRouter()
        emb = router._get_embedding("")
        norm = math.sqrt(sum(v * v for v in emb))
        assert norm == 0.0
