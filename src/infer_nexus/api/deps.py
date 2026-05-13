from fastapi import Request

from infer_nexus.catalog.registry import ModelRegistry
from infer_nexus.control.admission import AdmissionController
from infer_nexus.control.load_inspector import LoadInspector


def get_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry


def get_load_inspector(request: Request) -> LoadInspector:
    return request.app.state.load_inspector


def get_admission_controller(request: Request) -> AdmissionController:
    return request.app.state.admission
