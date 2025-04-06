"""Severity levels for ML static analysis framework."""

from enum import Enum, auto


class Severity(Enum):
    """Severity levels for findings in the ML static analysis framework."""
    
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    
    def __str__(self) -> str:
        """Return a string representation of the severity level."""
        return self.name.lower()
    
    @classmethod
    def from_str(cls, severity_str: str) -> 'Severity':
        """Create a Severity from a string.
        
        Args:
            severity_str: String representation of a severity level.
            
        Returns:
            A Severity instance.
        """
        severity_str = severity_str.upper()
        
        if severity_str == "INFO":
            return cls.INFO
        elif severity_str == "WARNING":
            return cls.WARNING
        elif severity_str == "ERROR":
            return cls.ERROR
        else:
            raise ValueError(f"Invalid severity: {severity_str}")
