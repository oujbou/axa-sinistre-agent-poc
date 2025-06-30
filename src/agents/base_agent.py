from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import json
from loguru import logger

class AgentStatus(Enum):
    """Agent execution status"""
    IDLE = "idle"
    REASONING = "reasoning"
    EXECUTING = "executing"
    ACTING = "acting"
    CRITICIZING = "criticizing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class AgentState:
    """Shared state between agents"""
    agent_name: str
    status: AgentStatus
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    reasoning: str
    actions_taken: List[str]
    critic_feedback: str
    confidence_score: float
    errors: List[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for LangGraph"""
        return {
            "agent_name": self.agent_name,
            "status": self.status.value,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "reasoning": self.reasoning,
            "actions_taken": self.actions_taken,
            "critic_feedback": self.critic_feedback,
            "confidence_score": self.confidence_score,
            "errors": self.errors,
            "metadata": self.metadata
        }


class BaseAgent(ABC):
    """
    Base class for all agents implementing REACT pattern

    REACT Pattern:
    - Reasoning: Analyze input and plan approach
    - Execution: Perform core task logic
    - Action: Execute specific operations
    - Critic: Evaluate results and provide feedback
    - Termination: Complete with validated output
    """

    def __init__(self, name: str, llm_client, config: Dict[str, Any] = None):
        self.name = name
        self.llm_client = llm_client
        self.config = config or {}
        self.state = None

        logger.info(f"Initialized {name} agent with REACT pattern")

    def initialize_state(self, input_data: Dict[str, Any]) -> AgentState:
        """Initialize agent state with input data"""
        self.state = AgentState(
            agent_name=self.name,
            status=AgentStatus.IDLE,
            input_data=input_data,
            output_data={},
            reasoning="",
            actions_taken=[],
            critic_feedback="",
            confidence_score=0.0,
            errors=[],
            metadata={}
        )
        return self.state

    def execute_react_cycle(self, input_data: Dict[str, Any]) -> AgentState:
        """
        Execute complete REACT cycle

        Returns:
            AgentState: Final state after REACT execution
        """
        try:
            # Initialize state
            self.initialize_state(input_data)

            # REACT Pattern execution
            self._reasoning_phase()
            self._execution_phase()
            self._action_phase()
            self._critic_phase()
            self._termination_phase()

            logger.info(f"Agent {self.name} completed REACT cycle successfully")
            return self.state

        except Exception as e:
            logger.error(f"Agent {self.name} failed during REACT cycle: {str(e)}")
            if self.state:
                self.state.status = AgentStatus.FAILED
                self.state.errors.append(str(e))
            return self.state or AgentState(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                input_data=input_data,
                output_data={},
                reasoning="",
                actions_taken=[],
                critic_feedback="",
                confidence_score=0.0,
                errors=[str(e)],
                metadata={}
            )

    def _reasoning_phase(self):
        """Phase 1: Reasoning - Analyze input and plan approach"""
        self.state.status = AgentStatus.REASONING
        logger.debug(f"{self.name}: Starting reasoning phase")

        reasoning = self.reason(self.state.input_data)
        self.state.reasoning = reasoning
        self.state.actions_taken.append("reasoning_completed")

        logger.debug(f"{self.name}: Reasoning completed")

    def _execution_phase(self):
        """Phase 2: Execution - Perform core task logic"""
        self.state.status = AgentStatus.EXECUTING
        logger.debug(f"{self.name}: Starting execution phase")

        output = self.execute(self.state.input_data, self.state.reasoning)
        self.state.output_data.update(output)
        self.state.actions_taken.append("execution_completed")

        logger.debug(f"{self.name}: Execution completed")

    def _action_phase(self):
        """Phase 3: Action - Execute specific operations"""
        self.state.status = AgentStatus.ACTING
        logger.debug(f"{self.name}: Starting action phase")

        actions_result = self.act(self.state.output_data)
        if actions_result:
            self.state.output_data.update(actions_result)
        self.state.actions_taken.append("actions_completed")

        logger.debug(f"{self.name}: Action completed")

    def _critic_phase(self):
        """Phase 4: Critic - Evaluate results and provide feedback"""
        self.state.status = AgentStatus.CRITICIZING
        logger.debug(f"{self.name}: Starting critic phase")

        critic_result = self.criticize(self.state.output_data)
        self.state.critic_feedback = critic_result.get("feedback", "")
        self.state.confidence_score = critic_result.get("confidence", 0.0)

        # Check if retry is needed
        if critic_result.get("needs_retry", False) and self.state.confidence_score < 0.7:
            logger.warning(f"{self.name}: Low confidence, considering retry")

        self.state.actions_taken.append("critic_completed")
        logger.debug(f"{self.name}: Critic completed")

    def _termination_phase(self):
        """Phase 5: Termination - Complete with validated output"""
        self.state.status = AgentStatus.COMPLETED
        logger.debug(f"{self.name}: Starting termination phase")

        # Final validation and cleanup
        final_output = self.terminate(self.state.output_data, self.state.critic_feedback)
        self.state.output_data = final_output
        self.state.actions_taken.append("termination_completed")

        logger.info(f"{self.name}: REACT cycle terminated successfully with confidence {self.state.confidence_score}")

    @abstractmethod
    def reason(self, input_data: Dict[str, Any]) -> str:
        """
        REASONING: Analyze input and determine strategy

        Args:
            input_data: Input data to reason about

        Returns:
            str: Reasoning explanation and strategy
        """
        pass

    @abstractmethod
    def execute(self, input_data: Dict[str, Any], reasoning: str) -> Dict[str, Any]:
        """
        EXECUTION: Perform the core task based on reasoning

        Args:
            input_data: Original input data
            reasoning: Strategy from reasoning phase

        Returns:
            Dict[str, Any]: Execution results
        """
        pass

    @abstractmethod
    def act(self, execution_output: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        ACTION: Execute specific operations based on execution results

        Args:
            execution_output: Results from execution phase

        Returns:
            Optional[Dict[str, Any]]: Additional action results
        """
        pass

    @abstractmethod
    def criticize(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        CRITIC: Evaluate results and provide feedback

        Args:
            output_data: All output data to evaluate

        Returns:
            Dict[str, Any]: Critic feedback with confidence score and retry flag
        """
        pass

    @abstractmethod
    def terminate(self, output_data: Dict[str, Any], critic_feedback: str) -> Dict[str, Any]:
        """
        TERMINATION: Final validation and output preparation

        Args:
            output_data: All output data
            critic_feedback: Feedback from critic phase

        Returns:
            Dict[str, Any]: Final validated output
        """
        pass