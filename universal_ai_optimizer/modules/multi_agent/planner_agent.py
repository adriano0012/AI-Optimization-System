"""
Planner Agent
Responsible for breaking down complex tasks into manageable steps
"""

from typing import Dict, Any, Optional, List
import logging
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class PlannerAgent(BaseAgent):
    """
    Agent that plans and decomposes complex tasks
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_steps = self.config.get('max_steps', 10)
        self.depth = self.config.get('depth', 2)
        
        self.logger.info("PlannerAgent initialized")
    
    def process(self, input_data: Any,
               context: Optional[Dict[str, Any]] = None,
               model_adapter: Optional[Any] = None,
               pipeline_state: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create a plan for accomplishing the task described in input_data
        
        Args:
            input_data: Task description or goal
            context: Optional context (previous attempts, resources, etc.)
            
        Returns:
            A plan consisting of steps to achieve the goal
        """
        if not self.validate_input(input_data):
            raise ValueError("Invalid input for PlannerAgent")
        
        task_description = str(input_data) if not isinstance(input_data, str) else input_data
        
        self.logger.info(f"Creating plan for task: {task_description[:100]}...")
        
        # In a real implementation, this would use LLM reasoning or specialized planning algorithms
        # For now, we'll create a simple placeholder plan
        
        plan = {
            'goal': task_description,
            'steps': self._decompose_task(task_description, context),
            'estimated_complexity': self._estimate_complexity(task_description),
            'required_resources': self._estimate_resources(task_description),
            'contingency_plans': self._generate_contingencies(task_description)
        }
        
        return plan
    
    def _decompose_task(self, task_description: str, 
                       context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Break down the task into smaller steps"""
        # Placeholder implementation
        # In reality, this would use NLP or LLM to understand the task and break it down
        
        steps = []
        
        # Simple heuristic: if task contains certain keywords, break it down accordingly
        if any(keyword in task_description.lower() for keyword in ['research', 'investigate', 'analyze']):
            steps = [
                {'id': 1, 'description': 'Define research scope and objectives', 'type': 'planning'},
                {'id': 2, 'description': 'Gather initial information and resources', 'type': 'information_gathering'},
                {'id': 3, 'description': 'Analyze collected information', 'type': 'analysis'},
                {'id': 4, 'description': 'Synthesize findings and draw conclusions', 'type': 'synthesis'},
                {'id': 5, 'description': 'Prepare final report or presentation', 'type': 'reporting'}
            ]
        elif any(keyword in task_description.lower() for keyword in ['code', 'develop', 'build', 'create']):
            steps = [
                {'id': 1, 'description': 'Requirements gathering and analysis', 'type': 'planning'},
                {'id': 2, 'description': 'System design and architecture', 'type': 'design'},
                {'id': 3, 'description': 'Implementation and coding', 'type': 'implementation'},
                {'id': 4, 'description': 'Testing and quality assurance', 'type': 'testing'},
                {'id': 5, 'description': 'Deployment and release', 'type': 'deployment'}
            ]
        else:
            # Generic task decomposition
            steps = [
                {'id': 1, 'description': 'Understand the task requirements', 'type': 'analysis'},
                {'id': 2, 'description': 'Identify necessary resources and constraints', 'type': 'planning'},
                {'id': 3, 'description': 'Execute the core task activities', 'type': 'execution'},
                {'id': 4, 'description': 'Review and validate results', 'type': 'validation'},
                {'id': 5, 'description': 'Finalize and deliver output', 'type': 'completion'}
            ]
        
        # Limit steps based on configuration
        return steps[:self.max_steps]
    
    def _estimate_complexity(self, task_description: str) -> str:
        """Estimate the complexity of the task"""
        # Simple heuristic based on length and keywords
        length_factor = min(len(task_description) / 100, 3.0)  # normalize length
        
        complexity_keywords = {
            'low': ['simple', 'basic', 'easy', 'straightforward'],
            'medium': ['moderate', 'intermediate', 'standard'],
            'high': ['complex', 'advanced', 'sophisticated', 'intricate', 'challenging']
        }
        
        task_lower = task_description.lower()
        complexity_score = 1.0  # default medium
        
        for level, keywords in complexity_keywords.items():
            if any(keyword in task_lower for keyword in keywords):
                if level == 'low':
                    complexity_score = 0.5
                elif level == 'medium':
                    complexity_score = 1.0
                else:  # high
                    complexity_score = 2.0
                break
        
        # Combine length and keyword factors
        final_score = (length_factor + complexity_score) / 2
        
        if final_score < 1.0:
            return 'low'
        elif final_score < 2.0:
            return 'medium'
        else:
            return 'high'
    
    def _estimate_resources(self, task_description: str) -> Dict[str, Any]:
        """Estimate resources needed for the task"""
        # Placeholder implementation
        complexity = self._estimate_complexity(task_description)
        
        base_resources = {
            'low': {'time_estimate_minutes': 15, 'compute_level': 'low'},
            'medium': {'time_estimate_minutes': 60, 'compute_level': 'medium'},
            'high': {'time_estimate_minutes': 240, 'compute_level': 'high'}
        }
        
        return base_resources.get(complexity, base_resources['medium'])
    
    def _generate_contingencies(self, task_description: str) -> List[Dict[str, Any]]:
        """Generate contingency plans for potential issues"""
        # Placeholder implementation
        return [
            {
                'scenario': 'Insufficient information or resources',
                'response': 'Gather additional data or adjust scope'
            },
            {
                'scenario': 'Unexpected complexity in execution',
                'response': 'Break down further or seek expert assistance'
            },
            {
                'scenario': 'Time constraints',
                'response': 'Prioritize critical path elements'
            }
        ]
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate that we have a task to plan for"""
        return input_data is not None and str(input_data).strip() != ''
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get planner agent metrics"""
        base_metrics = super().get_metrics() if hasattr(super(), 'get_metrics') else {}
        base_metrics.update({
            'max_steps': self.max_steps,
            'planning_depth': self.depth
        })
        return base_metrics