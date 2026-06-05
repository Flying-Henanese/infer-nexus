"""infer-nexus 的共享异常层级。"""


class InferNexusError(Exception):
    """infer-nexus 所有自定义异常的基类。"""


class ConfigError(InferNexusError):
    """配置缺失或配置内容无效时抛出。"""


class ModelNotFoundError(InferNexusError):
    """请求的模型未在模型目录中注册时抛出。"""


class ModelArtifactMissingError(InferNexusError):
    """已配置模型的本地产物路径不存在时抛出。"""

    def __init__(self, model_name: str, model_path: str, resolved_path: str) -> None:
        """构造模型产物缺失错误，并附加定位信息。"""
        message = (
            f"Model '{model_name}' is registered but missing from local storage. "
            f"Configured model_path='{model_path}', resolved_path='{resolved_path}'."
        )
        super().__init__(message)
        self.model_name = model_name
        self.model_path = model_path
        self.resolved_path = resolved_path


class RuntimeNotConnectedError(InferNexusError):
    """运行时分发路径存在但尚未连接执行器时抛出。"""

    def __init__(self, message: str, code: str = "runtime_not_connected") -> None:
        """构造运行时未连接错误，并保存 API 面向的错误码。"""
        super().__init__(message)
        self.code = code


class RuntimeExecutionError(InferNexusError):
    """运行时目标已连接但后端执行失败时抛出。"""

    def __init__(self, message: str, code: str = "runtime_execution_failed") -> None:
        """构造运行时执行错误，并保存 API 面向的错误码。"""
        super().__init__(message)
        self.code = code


class BackendRequestValidationError(InferNexusError):
    """后端无法在当前能力范围内服务某个有效 API 请求时抛出。"""

    def __init__(self, message: str, code: str = "unsupported_parameter") -> None:
        """构造后端请求校验错误，并保存 API 面向的错误码。"""
        super().__init__(message)
        self.code = code


class BackendConfigurationError(InferNexusError):
    """模型、后端或运行时配置内部不一致时抛出。"""

    def __init__(self, message: str, code: str = "backend_configuration_error") -> None:
        """构造后端配置错误，并保存 API 面向的错误码。"""
        super().__init__(message)
        self.code = code


class AdmissionRejectedError(InferNexusError):
    """准入控制拒绝请求时抛出。"""

    def __init__(self, message: str, code: str = "overloaded") -> None:
        """构造准入拒绝错误，并保存 API 面向的错误码。"""
        super().__init__(message)
        self.code = code
