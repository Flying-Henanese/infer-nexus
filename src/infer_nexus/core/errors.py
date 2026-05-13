class InferNexusError(Exception):
    """Base error for infer-nexus."""


class ConfigError(InferNexusError):
    """Raised when configuration is missing or invalid."""


class ModelNotFoundError(InferNexusError):
    """Raised when a requested model is not registered."""


class AdmissionRejectedError(InferNexusError):
    """Raised when admission control rejects a request."""

    def __init__(self, message: str, code: str = "overloaded") -> None:
        super().__init__(message)
        self.code = code
