"""FastAPI 依赖注入函数：从 app.state 提取运行期组件。"""

from fastapi import Request

from infer_nexus.catalog.registry import ModelRegistry
from infer_nexus.control.admission import AdmissionController
from infer_nexus.control.load_inspector import LoadInspector
from infer_nexus.model_store import LocalModelStore
from infer_nexus.runtime.dispatcher import RuntimeDispatcher


def get_registry(request: Request) -> ModelRegistry:
    """返回模型注册表。"""
    return request.app.state.registry


def get_load_inspector(request: Request) -> LoadInspector:
    """返回集群负载快照采集器。"""
    return request.app.state.load_inspector


def get_admission_controller(request: Request) -> AdmissionController:
    """返回请求准入控制器。"""
    return request.app.state.admission


def get_model_store(request: Request) -> LocalModelStore:
    """返回本地模型仓库。"""
    return request.app.state.model_store


def get_runtime_dispatcher(request: Request) -> RuntimeDispatcher:
    """返回运行时请求分发器。"""
    return request.app.state.runtime_dispatcher
