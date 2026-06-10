"""Prometheus 指标导出路由。"""

from fastapi import APIRouter
from fastapi.responses import Response

from infer_nexus.observability.metrics import render_prometheus_metrics

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def metrics() -> Response:
    """导出 Prometheus text format 实时指标。"""
    body, content_type = render_prometheus_metrics()
    return Response(content=body, headers={"Content-Type": content_type})
