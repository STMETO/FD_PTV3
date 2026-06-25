"""服务端构建器。"""

from ..utils.config import _get_cfg

from .orchestrator import DefaultFederatedServer


_SERVER_REGISTRY = {
    "defaultfederatedserver": DefaultFederatedServer,
    "default": DefaultFederatedServer,
}


def build_server(cfg):
    fed_cfg = _get_cfg(cfg, "federated", {})
    server_cfg = fed_cfg.get("server", {}) if isinstance(fed_cfg, dict) else {}
    server_type = server_cfg.get("type", "DefaultFederatedServer")
    server_cls = _SERVER_REGISTRY.get(server_type.lower(), DefaultFederatedServer)
    return server_cls(cfg)
