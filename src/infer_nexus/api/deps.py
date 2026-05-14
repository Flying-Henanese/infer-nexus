from fastapi import Request

from infer_nexus.catalog.registry import ModelRegistry
from infer_nexus.control.admission import AdmissionController
from infer_nexus.control.load_inspector import LoadInspector
from infer_nexus.model_store import LocalModelStore
from infer_nexus.runtime.dispatcher import RuntimeDispatcher


def get_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry


def get_load_inspector(request: Request) -> LoadInspector:
    return request.app.state.load_inspector


def get_admission_controller(request: Request) -> AdmissionController:
    return request.app.state.admission


def get_model_store(request: Request) -> LocalModelStore:
    return request.app.state.model_store


def get_runtime_dispatcher(request: Request) -> RuntimeDispatcher:
    return request.app.state.runtime_dispatcher
