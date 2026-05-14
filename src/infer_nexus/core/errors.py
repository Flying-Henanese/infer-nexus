class InferNexusError(Exception):
    """Base error for infer-nexus."""


class ConfigError(InferNexusError):
    """Raised when configuration is missing or invalid."""


class ModelNotFoundError(InferNexusError):
    """Raised when a requested model is not registered."""


class ModelArtifactMissingError(InferNexusError):
    """Raised when a configured model's local artifact path is missing."""

    def __init__(self, model_name: str, model_path: str, resolved_path: str) -> None:
        message = (
            f"Model '{model_name}' is registered but missing from local storage. "
            f"Configured model_path='{model_path}', resolved_path='{resolved_path}'."
        )
        super().__init__(message)
        self.model_name = model_name
        self.model_path = model_path
        self.resolved_path = resolved_path


class RuntimeNotConnectedError(InferNexusError):
    """Raised when a runtime dispatch path exists but no executor is wired yet."""

    def __init__(self, message: str, code: str = "runtime_not_connected") -> None:
        super().__init__(message)
        self.code = code


class BackendRequestValidationError(InferNexusError):
    """Raised when a backend cannot serve a valid API request within current support bounds."""

    def __init__(self, message: str, code: str = "unsupported_parameter") -> None:
        super().__init__(message)
        self.code = code


class BackendConfigurationError(InferNexusError):
    """Raised when a model/backend/runtime configuration is internally inconsistent."""

    def __init__(self, message: str, code: str = "backend_configuration_error") -> None:
        super().__init__(message)
        self.code = code


class AdmissionRejectedError(InferNexusError):
    """Raised when admission control rejects a request."""

    def __init__(self, message: str, code: str = "overloaded") -> None:
        super().__init__(message)
        self.code = code
