"""agent-tap: Proxy to trace Claude Code API requests.

A CLI tool that wraps Claude Code with a local proxy (reverse or forward)
to intercept and record all API requests. Useful for studying Claude Code's
Context Engineering.
"""

from __future__ import annotations

from agent_tap.certs import CertificateAuthority, ensure_ca
from agent_tap.cli import (
    __version__,
    _build_update_command,
    _detect_installer,
    async_main,
    dashboard_main,
    main_entry,
    parse_args,
    parse_dashboard_args,
    parse_trust_ca_args,
    parse_update_args,
    trust_ca_main,
    update_main,
)
from agent_tap.forward_proxy import ForwardProxyServer
from agent_tap.history import cleanup_trace_sessions, delete_trace_history, migrate_legacy_traces
from agent_tap.live import LiveViewerServer
from agent_tap.proxy import filter_headers
from agent_tap.sse import SSEReassembler
from agent_tap.trace import TraceWriter
from agent_tap.trace_store import get_trace_store, reset_trace_store, resolve_db_path
from agent_tap.viewer import _generate_html_viewer

__all__ = [
    "__version__",
    "_build_update_command",
    "_detect_installer",
    "main_entry",
    "parse_args",
    "parse_dashboard_args",
    "parse_trust_ca_args",
    "parse_update_args",
    "trust_ca_main",
    "update_main",
    "async_main",
    "dashboard_main",
    "CertificateAuthority",
    "ensure_ca",
    "ForwardProxyServer",
    "SSEReassembler",
    "TraceWriter",
    "LiveViewerServer",
    "filter_headers",
    "_generate_html_viewer",
    "cleanup_trace_sessions",
    "delete_trace_history",
    "migrate_legacy_traces",
    "get_trace_store",
    "reset_trace_store",
    "resolve_db_path",
]
