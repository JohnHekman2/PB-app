"""
Token usage tracking and estimation.
Tracks AI API token consumption for rate limiting and cost analysis.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Dict
from datetime import datetime

from services.logging_config import get_logger, get_token_logger

logger = get_logger()
token_logger = get_token_logger()


@dataclass
class TokenUsage:
    """
    Captures token usage for a single API call.
    """
    provider: str  # "openai" or "gemini"
    operation: str  # "analyze_pdf", "batch_analysis", "generate_conclusion", etc.
    input_tokens: int
    output_tokens: int
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used in this operation."""
        return self.input_tokens + self.output_tokens
    
    def __str__(self) -> str:
        return (
            f"{self.provider.upper()} | {self.operation} | "
            f"IN: {self.input_tokens:,} | OUT: {self.output_tokens:,} | "
            f"TOTAL: {self.total_tokens:,}"
        )


@dataclass
class SessionTokenStats:
    """
    Aggregated token statistics for a session.
    """
    session_id: str
    start_time: datetime = field(default_factory=datetime.now)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    operations_count: int = 0
    operations_by_type: Dict[str, int] = field(default_factory=dict)
    operations_by_provider: Dict[str, int] = field(default_factory=dict)
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used in this session."""
        return self.total_input_tokens + self.total_output_tokens
    
    @property
    def elapsed_seconds(self) -> float:
        """Seconds elapsed since session start."""
        return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def tokens_per_second(self) -> float:
        """Average token consumption rate."""
        if self.elapsed_seconds == 0:
            return 0
        return self.total_tokens / self.elapsed_seconds
    
    def to_dict(self) -> dict:
        """Convert to dictionary for easy serialization."""
        return {
            "session_id": self.session_id,
            "total_tokens": self.total_tokens,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "operations_count": self.operations_count,
            "elapsed_seconds": self.elapsed_seconds,
            "tokens_per_second": self.tokens_per_second,
            "operations_by_type": self.operations_by_type,
            "operations_by_provider": self.operations_by_provider,
        }
    
    def __str__(self) -> str:
        return (
            f"Session {self.session_id} | {self.operations_count} ops | "
            f"IN: {self.total_input_tokens:,} | OUT: {self.total_output_tokens:,} | "
            f"TOTAL: {self.total_tokens:,} tokens in {self.elapsed_seconds:.1f}s"
        )


class TokenUsageTracker:
    """
    Tracks token usage across the application session.
    Provides methods for logging, aggregation, and rate limit checking.
    """
    
    def __init__(self, session_id: str):
        """
        Initialize token tracker for a session.
        
        Args:
            session_id: Unique identifier for this session (e.g., session hash)
        """
        self.session_id = session_id
        self.stats = SessionTokenStats(session_id=session_id)
        self.usage_history = []
        logger.info(f"Token tracker initialized for session {session_id}")
    
    def log_token_usage(
        self,
        provider: str,
        operation: str,
        input_tokens: int,
        output_tokens: int
    ) -> TokenUsage:
        """
        Log token usage for a single API call.
        
        Args:
            provider: "openai" or "gemini"
            operation: Description of what was done (e.g., "analyze_pdf")
            input_tokens: Estimated or actual input tokens
            output_tokens: Estimated or actual output tokens
            
        Returns:
            TokenUsage object with the recorded usage
        """
        usage = TokenUsage(
            provider=provider,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
        
        # Update aggregated stats
        self.stats.total_input_tokens += input_tokens
        self.stats.total_output_tokens += output_tokens
        self.stats.operations_count += 1
        self.stats.operations_by_type[operation] = self.stats.operations_by_type.get(operation, 0) + 1
        self.stats.operations_by_provider[provider] = self.stats.operations_by_provider.get(provider, 0) + 1
        
        # Store in history
        self.usage_history.append(usage)
        
        # Log to token logger
        token_logger.info(str(usage))
        
        # Also log to main logger at DEBUG level
        logger.debug(f"Token usage recorded: {usage}")
        
        return usage
    
    def estimate_tokens(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        """
        Estimate token count for text using tiktoken.
        
        Args:
            text: Text to estimate tokens for
            model: Model name for tiktoken encoding (default: "gpt-3.5-turbo")
            
        Returns:
            Estimated number of tokens
        """
        try:
            import tiktoken
            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fallback to cl100k_base encoding (most common)
                encoding = tiktoken.get_encoding("cl100k_base")
            tokens = len(encoding.encode(text))
            return tokens
        except ImportError:
            logger.warning("tiktoken not available; using rough estimation")
            # Rough fallback: ~1 token per 4 characters
            return len(text) // 4
    
    def get_stats(self) -> SessionTokenStats:
        """
        Get current session statistics.
        
        Returns:
            SessionTokenStats object
        """
        return self.stats
    
    def get_stats_summary(self) -> str:
        """
        Get human-readable summary of token usage.
        
        Returns:
            Formatted string with session statistics
        """
        return str(self.stats)
    
    def check_rate_limit(
        self,
        max_tokens_per_session: Optional[int] = None,
        max_tokens_per_minute: Optional[int] = None
    ) -> tuple[bool, str]:
        """
        Check if current usage exceeds rate limits.
        
        Args:
            max_tokens_per_session: Maximum total tokens allowed per session
            max_tokens_per_minute: Maximum tokens allowed per minute
            
        Returns:
            Tuple of (is_within_limits, reason_if_exceeded)
        """
        if max_tokens_per_session and self.stats.total_tokens > max_tokens_per_session:
            reason = f"Session token limit exceeded: {self.stats.total_tokens:,} > {max_tokens_per_session:,}"
            logger.warning(reason)
            return False, reason
        
        if max_tokens_per_minute:
            minute_tokens = sum(
                u.total_tokens for u in self.usage_history
                if (datetime.now() - u.timestamp).total_seconds() < 60
            )
            if minute_tokens > max_tokens_per_minute:
                reason = f"Per-minute token limit exceeded: {minute_tokens:,} > {max_tokens_per_minute:,}"
                logger.warning(reason)
                return False, reason
        
        return True, "OK"
    
    def reset(self):
        """Reset the tracker for a new session."""
        self.stats = SessionTokenStats(session_id=self.session_id)
        self.usage_history = []
        logger.info(f"Token tracker reset for session {self.session_id}")
    
    def to_dict(self) -> dict:
        """Export stats as dictionary."""
        return self.stats.to_dict()
