"""
Orchestrator Agent
Coordinates all other agents and manages the multi-agent workflow
"""

from typing import Dict, Any, Optional, List
import logging
from .base_agent import BaseAgent
from .planner_agent import PlannerAgent
from .coding_agent import CodingAgent
from .critic_agent import CriticAgent
from .consensus_agent import ConsensusAgent
from .research_agent import ResearchAgent
from .verification_agent import VerificationAgent

logger = logging.getLogger(__name__)

class OrchestratorAgent(BaseAgent):
    """
    Central orchestrator that coordinates multiple agents to solve complex tasks.
    Decomposes tasks, dispatches to specialized agents, and synthesizes results.
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        self.agents = {}
        self.max_retries = self.config.get('max_retries', 2)
        self.consensus_threshold = self.config.get('consensus_threshold', 0.7)
        
        # Auto-register all available agents
        self._register_all_agents()
    
    def _register_all_agents(self):
        """Register default agent instances"""
        agent_classes = [
            ('PlannerAgent', PlannerAgent),
            ('CodingAgent', CodingAgent),
            ('CriticAgent', CriticAgent),
            ('ConsensusAgent', ConsensusAgent),
            ('ResearchAgent', ResearchAgent),
            ('VerificationAgent', VerificationAgent),
        ]
        for name, cls in agent_classes:
            try:
                agent = cls(self.config)
                self.register_agent(name, agent)
            except Exception as e:
                self.logger.warning(f"Failed to register {name}: {e}")
    
    def register_agent(self, name: str, agent: BaseAgent):
        """Register a specialized agent"""
        self.agents[name] = agent
        self.logger.debug(f"Registered agent: {name}")
    
    def process(self, input_data: Any, context=None, model_adapter=None, pipeline_state=None) -> Dict[str, Any]:
        """
        Orchestrate multi-agent processing workflow
        
        Args:
            input_data: The task/prompt to process
            context: Optional context
            
        Returns:
            Dict with synthesized result, agent outputs, and metadata
        """
        context = context or {}
        
        if not self.validate_input(input_data):
            return {'error': 'Invalid input', 'result': None}
        
        task_type = context.get('task_type', self._classify_task(input_data))
        
        agent_outputs = {}
        agent_scores = {}
        
        # Dispatch to relevant agents based on task type
        relevant_agents = self._select_agents(task_type, context)
        
        for agent_name in relevant_agents:
            if agent_name in self.agents:
                try:
                    agent = self.agents[agent_name]
                    output = agent.process(input_data, context)
                    agent_outputs[agent_name] = output
                    agent_scores[agent_name] = self._score_output(output, context)
                except Exception as e:
                    self.logger.warning(f"Agent {agent_name} failed: {e}")
                    agent_outputs[agent_name] = {'error': str(e)}
                    agent_scores[agent_name] = 0.0
        
        # Synthesize results
        result = self._synthesize_results(agent_outputs, agent_scores, context)
        
        return {
            'result': result,
            'agent_outputs': agent_outputs,
            'agent_scores': agent_scores,
            'task_type': task_type,
            'agents_used': list(agent_outputs.keys()),
            'consensus_reached': max(agent_scores.values(), default=0) >= self.consensus_threshold if agent_scores else False
        }
    
    def _classify_task(self, input_data: Any) -> str:
        """Classify the task type based on input"""
        text = str(input_data).lower()
        
        if any(kw in text for kw in ['code', 'function', 'class', 'debug', 'implement']):
            return 'coding'
        elif any(kw in text for kw in ['research', 'find', 'search', 'analyze', 'compare']):
            return 'research'
        elif any(kw in text for kw in ['verify', 'check', 'validate', 'fact']):
            return 'verification'
        elif any(kw in text for kw in ['plan', 'strategy', 'approach', 'design']):
            return 'planning'
        else:
            return 'general'
    
    def _select_agents(self, task_type: str, context: Dict[str, Any]) -> List[str]:
        """Select which agents to use based on task type"""
        agent_map = {
            'coding': ['CodingAgent', 'CriticAgent', 'VerificationAgent'],
            'research': ['ResearchAgent', 'CriticAgent'],
            'verification': ['VerificationAgent', 'ConsensusAgent'],
            'planning': ['PlannerAgent', 'CriticAgent'],
            'general': ['PlannerAgent', 'CriticAgent', 'ConsensusAgent']
        }
        
        selected = agent_map.get(task_type, ['PlannerAgent', 'CriticAgent'])
        return [a for a in selected if a in self.agents]
    
    def _score_output(self, output: Any, context: Dict[str, Any]) -> float:
        """Score an agent's output quality"""
        if output is None:
            return 0.0
        if isinstance(output, dict) and 'error' in output:
            return 0.1
        if isinstance(output, dict) and 'confidence' in output:
            return float(output['confidence'])
        if isinstance(output, str) and len(output) > 0:
            return 0.7
        return 0.5
    
    def _synthesize_results(self, outputs: Dict[str, Any], 
                          scores: Dict[str, Any],
                          context: Dict[str, Any]) -> Any:
        """Synthesize results from multiple agents"""
        if not outputs:
            return None
        
        if len(outputs) == 1:
            return list(outputs.values())[0]
        
        # Weight by score
        best_agent = max(scores, key=scores.get) if scores else list(outputs.keys())[0]
        best_output = outputs[best_agent]
        
        # If outputs are strings, concatenate with priority
        if all(isinstance(v, str) for v in outputs.values()):
            sorted_outputs = sorted(outputs.items(), key=lambda x: scores.get(x[0], 0), reverse=True)
            return '\n\n'.join(f"## {name}\n{output}" for name, output in sorted_outputs)
        
        # For dict outputs, return the highest-scored one
        return best_output
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            'registered_agents': list(self.agents.keys()),
            'agent_count': len(self.agents)
        }
