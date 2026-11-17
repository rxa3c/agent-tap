from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_tap import cli_clients
from agent_tap.cli import CLIENT_CONFIGS, parse_args, run_client
from agent_tap.forward_proxy import ForwardProxyServer

_CA_PEM = "-----BEGIN CERTIFICATE-----\nTAPCA\nLINE2\n-----END CERTIFICATE-----\n"


class _DummyProc:
    def __init__(self) -> None:
        self.pid = 12345
        self.returncode: int | None = None

    async def wait(self) -> int:
        self.returncode = 0
        return 0

    def terminate(self) -> None:
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9


def _write_app(app_root: Path, bundle_id: str) -> Path:
    (app_root / "Contents" / "MacOS").mkdir(parents=True, exist_ok=True)
    (app_root / "Contents" / "Info.plist").write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleIdentifier</key><string>{bundle_id}</string>
</dict></plist>
"""
    )
    executable = app_root / "Contents" / "MacOS" / "Electron"
    executable.write_text("")
    return executable


def _forward_server(client: str) -> ForwardProxyServer:
    cfg = CLIENT_CONFIGS[client]
    return ForwardProxyServer(
        host="127.0.0.1",
        port=0,
        ca=None,  # type: ignore[arg-type]
        writer=None,  # type: ignore[arg-type]
        session=None,  # type: ignore[arg-type]
        trace_methods=cfg.forward_trace_methods,
        trace_path_prefixes=cfg.forward_trace_path_prefixes,
        trace_path_suffixes=cfg.forward_trace_path_suffixes,
    )


def test_workbuddy_config_is_forward_only_with_keychain_trust() -> None:
    cfg = CLIENT_CONFIGS["workbuddy"]

    assert cfg.default_proxy_mode == "forward"
    assert cfg.auto_trust_ca_macos is True
    assert cfg.default_target == "https://www.workbuddy.cn/v2"


def test_workbuddy_trace_filter_keeps_only_model_calls() -> None:
    server = _forward_server("workbuddy")

    assert server._should_trace_request("POST", "/v2/chat/completions")
    assert server._should_trace_request("post", "/v2/chat/completions")
    assert not server._should_trace_request("GET", "/v2/chat/completions")
    assert not server._should_trace_request("POST", "/v2/plugin/login/account")
    assert not server._should_trace_request("POST", "/v2/update/check")
    assert not server._should_trace_request("GET", "/v2/config")


def test_parse_args_workbuddy_defaults_to_forward() -> None:
    args = parse_args(["--tap-client", "workbuddy"])

    assert args.client == "workbuddy"
    assert args.proxy_mode == "forward"
    assert args.target == "https://www.workbuddy.cn/v2"


def test_parse_args_workbuddy_rejects_reverse_mode() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--tap-client", "workbuddy", "--tap-proxy-mode", "reverse"])


def test_workbuddy_executable_requires_workbuddy_bundle_id(tmp_path: Path) -> None:
    good = _write_app(tmp_path / "WorkBuddy.app", "com.tencent.workbuddy.mac")
    other = _write_app(tmp_path / "Other.app", "com.example.electron")
    bare = tmp_path / "bin" / "workbuddy-stub"
    bare.parent.mkdir()
    bare.write_text("")

    assert cli_clients._is_workbuddy_executable(good)
    assert not cli_clients._is_workbuddy_executable(other)
    assert cli_clients._is_workbuddy_executable(bare)
    assert not cli_clients._is_workbuddy_executable(tmp_path / "missing")


def test_resolve_workbuddy_executable_prefers_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    override = _write_app(tmp_path / "Custom" / "WorkBuddy.app", "com.tencent.workbuddy.mac")
    monkeypatch.setenv("WORKBUDDY_EXECUTABLE", str(override))
    monkeypatch.setattr(cli_clients.sys, "platform", "darwin")

    resolved = cli_clients._resolve_client_executable("workbuddy", CLIENT_CONFIGS["workbuddy"], None)

    assert resolved == str(override)
    assert cli_clients._workbuddy_executable_candidates()[1] == Path(
        "/Applications/WorkBuddy.app/Contents/MacOS/Electron"
    )


def test_resolve_workbuddy_executable_skips_wrong_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wrong = _write_app(tmp_path / "WorkBuddy.app", "com.example.other")
    monkeypatch.setattr(cli_clients, "_workbuddy_executable_candidates", lambda: (wrong,))

    assert cli_clients._resolve_client_executable("workbuddy", CLIENT_CONFIGS["workbuddy"], None) is None


def test_workbuddy_existing_processes_filters_current_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    current_pid = os.getpid()

    def fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        captured["cmd"] = cmd
        return SimpleNamespace(
            returncode=0,
            stdout=(
                f"{current_pid} agent-tap --tap-client workbuddy\n"
                "201 /Applications/WorkBuddy.app/Contents/MacOS/Electron\n"
                "202 /Applications/WorkBuddy.app/Contents/Resources/app.asar.unpacked/cli/bin/codebuddy\n"
            ),
        )

    monkeypatch.setattr(cli_clients.sys, "platform", "darwin")
    monkeypatch.delenv("WORKBUDDY_EXECUTABLE", raising=False)
    monkeypatch.setattr(cli_clients.subprocess, "run", fake_run)

    lines = cli_clients._workbuddy_existing_processes()

    assert captured["cmd"] == ["pgrep", "-fl", r"(WorkBuddy\.app/Contents/)"]
    assert lines == [
        "201 /Applications/WorkBuddy.app/Contents/MacOS/Electron",
        "202 /Applications/WorkBuddy.app/Contents/Resources/app.asar.unpacked/cli/bin/codebuddy",
    ]


def test_workbuddy_existing_processes_ignores_pgrep_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_clients.sys, "platform", "darwin")
    monkeypatch.setattr(
        cli_clients.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=2, stdout="garbage"),
    )
    assert cli_clients._workbuddy_existing_processes() == []

    monkeypatch.setattr(cli_clients.sys, "platform", "linux")
    assert cli_clients._workbuddy_existing_processes() == []


@pytest.mark.asyncio
async def test_run_client_workbuddy_refuses_when_app_running(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fail_create_subprocess_exec(*_cmd: str, **_kwargs: object) -> _DummyProc:
        raise AssertionError("WorkBuddy must not be launched while it is running")

    monkeypatch.setattr(
        "agent_tap.cli_clients._resolve_client_executable",
        lambda client, cfg, client_cmd: "/Applications/WorkBuddy.app/Contents/MacOS/Electron",
    )
    monkeypatch.setattr(
        "agent_tap.cli_clients._workbuddy_existing_processes",
        lambda: ["201 /Applications/WorkBuddy.app/Contents/MacOS/Electron"],
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail_create_subprocess_exec)

    code = await run_client(43123, [], client="workbuddy", proxy_mode="forward", ca_cert_path=None)

    assert code == 1
    out = capsys.readouterr().err
    assert "WorkBuddy is already running" in out
    assert "Quit WorkBuddy completely" in out


@pytest.mark.asyncio
async def test_run_client_workbuddy_forward_launches_with_proxy_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    ca_path = tmp_path / "ca.pem"
    ca_path.write_text(_CA_PEM)
    monkeypatch.setenv("WORKBUDDY_CONFIG_DIR", str(tmp_path / "workbuddy-home"))
    monkeypatch.delenv("CODEBUDDY_BASE_URL", raising=False)

    async def fake_create_subprocess_exec(*cmd: str, **kwargs: object) -> _DummyProc:
        captured["cmd"] = cmd
        captured.update(kwargs)
        return _DummyProc()

    monkeypatch.setattr(
        "agent_tap.cli_clients._resolve_client_executable",
        lambda client, cfg, client_cmd: "/Applications/WorkBuddy.app/Contents/MacOS/Electron",
    )
    monkeypatch.setattr("agent_tap.cli_clients._workbuddy_existing_processes", lambda: [])
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("agent_tap.cli_clients._ensure_app_frontmost", lambda pid: True)
    monkeypatch.setattr("agent_tap.cli_clients._workbuddy_host_pids", lambda: set())
    monkeypatch.setattr("agent_tap.cli_clients._WORKBUDDY_RELAUNCH_WAIT_SECONDS", 0)

    code = await run_client(43123, [], client="workbuddy", proxy_mode="forward", ca_cert_path=ca_path)

    env = captured["env"]
    assert code == 0
    assert captured["cmd"] == ("/Applications/WorkBuddy.app/Contents/MacOS/Electron",)
    assert env["HTTP_PROXY"] == "http://127.0.0.1:43123"
    assert env["HTTPS_PROXY"] == "http://127.0.0.1:43123"
    assert "127.0.0.1" in env["NO_PROXY"].split(",")
    assert env["NODE_EXTRA_CA_CERTS"] == str(ca_path)
    # Forward mode must not redirect the model endpoint itself.
    assert "CODEBUDDY_BASE_URL" not in env
    assert captured["stdin"] == subprocess.DEVNULL
    assert captured["stdout"] == subprocess.DEVNULL
    assert captured["stderr"] == subprocess.DEVNULL


def test_ensure_app_frontmost_targets_pid_and_never_launches(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="front\n", stderr="")

    monkeypatch.setattr(cli_clients.subprocess, "run", fake_run)
    assert cli_clients._ensure_app_frontmost(4321) is True
    script = calls[0][-1]
    assert calls[0][:4] == ["osascript", "-l", "JavaScript", "-e"]
    assert "var pid = 4321;" in script
    assert "runningApplicationWithProcessIdentifier(pid)" in script
    # `tell application ... to activate` would launch an unproxied instance if the app had exited.
    assert "tell application" not in script

    # An activation request alone is not success; a later call must see it in front.
    for not_front in ("activated\n", "missing\n"):
        monkeypatch.setattr(
            cli_clients.subprocess,
            "run",
            lambda cmd, out=not_front, **kwargs: SimpleNamespace(returncode=0, stdout=out),
        )
        assert cli_clients._ensure_app_frontmost(4321) is False

    def raise_os_error(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        raise OSError("no osascript")

    monkeypatch.setattr(cli_clients.subprocess, "run", raise_os_error)
    assert cli_clients._ensure_app_frontmost(4321) is False


@pytest.mark.asyncio
async def test_bring_workbuddy_to_front_retries_until_activated(monkeypatch: pytest.MonkeyPatch) -> None:
    results = iter([False, False, True])
    seen: list[int] = []

    def fake_activate(pid: int) -> bool:
        seen.append(pid)
        return next(results)

    monkeypatch.setattr(cli_clients, "_WORKBUDDY_ACTIVATE_INITIAL_DELAY_SECONDS", 0)
    monkeypatch.setattr(cli_clients, "_WORKBUDDY_ACTIVATE_RETRY_SECONDS", 0)
    monkeypatch.setattr(cli_clients, "_ensure_app_frontmost", fake_activate)

    await cli_clients._bring_workbuddy_to_front(_DummyProc())  # type: ignore[arg-type]

    assert seen == [12345, 12345, 12345]


@pytest.mark.asyncio
async def test_bring_workbuddy_to_front_skips_exited_process(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = _DummyProc()
    proc.returncode = 0
    monkeypatch.setattr(cli_clients, "_WORKBUDDY_ACTIVATE_INITIAL_DELAY_SECONDS", 0)
    monkeypatch.setattr(cli_clients, "_ensure_app_frontmost", lambda pid: pytest.fail("must not activate"))

    await cli_clients._bring_workbuddy_to_front(proc)  # type: ignore[arg-type]


def test_workbuddy_host_pids_ignore_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli_clients,
        "_workbuddy_existing_processes",
        lambda: [
            "111 /Applications/WorkBuddy.app/Contents/MacOS/Electron",
            "112 /Applications/WorkBuddy.app/Contents/MacOS/Electron --disable-warning=ExperimentalWarning daemon.js",
            "113 /Applications/WorkBuddy.app/Contents/Frameworks/WorkBuddy Helper.app/Contents/MacOS/WorkBuddy Helper",
        ],
    )
    assert cli_clients._workbuddy_host_pids() == {111}


@pytest.mark.asyncio
async def test_workbuddy_relaunch_detected_after_fast_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    # The updater takes a few seconds to bring the new host up.
    snapshots = iter([set(), {4321}, {4321, 9999}])
    monkeypatch.setattr(cli_clients, "_WORKBUDDY_RELAUNCH_POLL_SECONDS", 0)
    monkeypatch.setattr(cli_clients, "_workbuddy_host_pids", lambda: next(snapshots))

    assert await cli_clients._workbuddy_relaunched_outside_tap(4321, elapsed=0.5) is True


@pytest.mark.asyncio
async def test_workbuddy_relaunch_not_reported_after_normal_quit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def host_pids() -> set[int]:
        nonlocal calls
        calls += 1
        return {4321}  # only the host agent-tap spawned, still shutting down

    monkeypatch.setattr(cli_clients, "_workbuddy_host_pids", host_pids)

    assert await cli_clients._workbuddy_relaunched_outside_tap(4321, elapsed=600.0) is False
    assert calls == 1  # a long session must not delay the summary


@pytest.mark.asyncio
async def test_run_client_workbuddy_warns_when_relaunched_outside_tap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fake_create_subprocess_exec(*cmd: str, **kwargs: object) -> _DummyProc:
        return _DummyProc()

    monkeypatch.setenv("WORKBUDDY_CONFIG_DIR", str(tmp_path / "workbuddy-home"))
    monkeypatch.setattr(
        "agent_tap.cli_clients._resolve_client_executable",
        lambda client, cfg, client_cmd: "/Applications/WorkBuddy.app/Contents/MacOS/Electron",
    )
    monkeypatch.setattr("agent_tap.cli_clients._workbuddy_existing_processes", lambda: [])
    monkeypatch.setattr("agent_tap.cli_clients._workbuddy_host_pids", lambda: {777})
    monkeypatch.setattr("agent_tap.cli_clients._ensure_app_frontmost", lambda pid: True)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    code = await run_client(43123, [], client="workbuddy", proxy_mode="forward", ca_cert_path=None)

    assert code == 0
    assert "started again outside agent-tap" in capsys.readouterr().err


def test_workbuddy_stale_ca_bundle_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_dir = tmp_path / "workbuddy-home"
    config_dir.mkdir()
    monkeypatch.setenv("WORKBUDDY_CONFIG_DIR", str(config_dir))
    ca_path = tmp_path / "ca.pem"
    ca_path.write_text(_CA_PEM)
    bundle = config_dir / "system-ca-bundle.pem"

    cli_clients._warn_if_workbuddy_ca_bundle_is_stale(ca_path)
    assert capsys.readouterr().err == ""

    bundle.write_text("-----BEGIN CERTIFICATE-----\nOTHER\n-----END CERTIFICATE-----\n")
    cli_clients._warn_if_workbuddy_ca_bundle_is_stale(ca_path)
    out = capsys.readouterr().err
    assert "does not include the agent-tap CA" in out
    assert str(bundle) in out
    assert bundle.exists(), "agent-tap must not modify WorkBuddy's cache"

    # WorkBuddy joins certificates with its own line breaks; compare ignoring whitespace.
    bundle.write_text(
        "-----BEGIN CERTIFICATE-----\r\nOTHER\r\n-----END CERTIFICATE-----\n" + _CA_PEM.replace("\n", "\r\n")
    )
    cli_clients._warn_if_workbuddy_ca_bundle_is_stale(ca_path)
    assert capsys.readouterr().err == ""


def test_workbuddy_config_dir_follows_workbuddy_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("WORKBUDDY_CONFIG_DIR", raising=False)
    monkeypatch.delenv("CODEBUDDY_CONFIG_DIR", raising=False)
    assert cli_clients._workbuddy_config_dir() == Path.home() / ".workbuddy"

    monkeypatch.setenv("CODEBUDDY_CONFIG_DIR", str(tmp_path / "cb"))
    assert cli_clients._workbuddy_config_dir() == tmp_path / "cb"

    monkeypatch.setenv("WORKBUDDY_CONFIG_DIR", str(tmp_path / "wb"))
    assert cli_clients._workbuddy_config_dir() == tmp_path / "wb"


_WORKBUDDY_SSE = (
    b'data: {"id":"c1","model":"wb-model","choices":[{"delta":{"role":"assistant","content":"Checking "}}]}\n\n'
    b'data: {"id":"c1","choices":[{"delta":{"tool_calls":'
    b'[{"index":0,"id":"call_1","type":"function","function":{"name":"read_file","arguments":""}}]}}]}\n\n'
    b'data: {"id":"c1","choices":[{"delta":{"tool_calls":'
    b'[{"index":0,"function":{"arguments":"{\\"path\\":\\"a.md\\"}"}}]}}]}\n\n'
    b'data: {"id":"c1","choices":[{"delta":{},"finish_reason":"tool_calls"}],'
    b'"usage":{"prompt_tokens":12,"completion_tokens":5}}\n\n'
    b"data: [DONE]\n\n"
)


async def _start_fake_workbuddy_upstream(tmp_path: Path):
    from aiohttp import web

    from agent_tap.certs import CertificateAuthority, ensure_ca

    upstream_ca = CertificateAuthority(*ensure_ca(tmp_path / "upstream-ca"))
    seen: list[dict] = []

    async def chat_completions(request: web.Request) -> web.StreamResponse:
        seen.append(await request.json())
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for chunk in _WORKBUDDY_SSE.split(b"\n\n")[:-1]:
            await resp.write(chunk + b"\n\n")
        await resp.write_eof()
        return resp

    async def account(_request: web.Request) -> web.Response:
        return web.json_response({"nickname": "secret-account-name"})

    app = web.Application()
    app.router.add_post("/v2/chat/completions", chat_completions)
    app.router.add_post("/v2/plugin/accounts", account)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0, ssl_context=upstream_ca.make_ssl_context("127.0.0.1"))
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return port, runner, seen


@pytest.mark.asyncio
async def test_workbuddy_forward_capture_decodes_gzip_and_skips_account_traffic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """WorkBuddy gzips model request bodies; only the model call may be persisted."""
    import gzip
    import json
    import ssl

    import aiohttp
    from yarl import URL

    from agent_tap.certs import CertificateAuthority, ensure_ca
    from agent_tap.trace import TraceWriter
    from agent_tap.trace_store import TraceStore

    ca_cert_path, ca_key_path = ensure_ca(tmp_path / "ca")
    upstream_port, upstream_runner, seen = await _start_fake_workbuddy_upstream(tmp_path)
    store = TraceStore(tmp_path / "traces.sqlite3")
    session_id = store.create_session(client="workbuddy", proxy_mode="forward")
    writer = TraceWriter(session_id, store=store)

    upstream_ssl = ssl.create_default_context()
    upstream_ssl.check_hostname = False
    upstream_ssl.verify_mode = ssl.CERT_NONE
    session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=upstream_ssl), auto_decompress=False)
    original_request = session.request

    async def _to_fake_upstream(method, url, **kwargs):
        return await original_request(method=method, url=URL(str(url)).with_host("127.0.0.1"), **kwargs)

    monkeypatch.setattr(session, "request", _to_fake_upstream)
    cfg = CLIENT_CONFIGS["workbuddy"]
    server = ForwardProxyServer(
        host="127.0.0.1",
        port=0,
        ca=CertificateAuthority(ca_cert_path, ca_key_path),
        writer=writer,
        session=session,
        trace_methods=cfg.forward_trace_methods,
        trace_path_prefixes=cfg.forward_trace_path_prefixes,
        trace_path_suffixes=cfg.forward_trace_path_suffixes,
    )
    proxy_port = await server.start()

    request_body = {
        "model": "wb-model",
        "stream": True,
        "messages": [
            {"role": "system", "content": "You are WorkBuddy."},
            {"role": "user", "content": "summarize a.md"},
        ],
        "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}],
    }
    try:
        client_ssl = ssl.create_default_context(cafile=str(ca_cert_path))
        proxy_url = f"http://127.0.0.1:{proxy_port}"
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=client_ssl)) as client:
            async with client.post(
                f"https://www.workbuddy.cn:{upstream_port}/v2/plugin/accounts",
                proxy=proxy_url,
                json={},
            ) as resp:
                assert (await resp.json())["nickname"] == "secret-account-name"
            async with client.post(
                f"https://www.workbuddy.cn:{upstream_port}/v2/chat/completions",
                proxy=proxy_url,
                data=gzip.compress(json.dumps(request_body).encode()),
                headers={"Content-Type": "application/json", "Content-Encoding": "gzip"},
            ) as resp:
                assert resp.status == 200
                streamed = await resp.read()
        writer.close()
        records = [json.loads(line) for line in store.export_jsonl(session_id).splitlines()]
    finally:
        await server.stop()
        await session.close()
        await upstream_runner.cleanup()

    # Compressed bytes are forwarded with their Content-Encoding; the upstream inflates them itself.
    assert seen == [request_body]
    assert streamed.endswith(b"data: [DONE]\n\n")
    assert len(records) == 1
    record = records[0]
    assert record["request"]["path"] == "/v2/chat/completions"
    assert record["request"]["body"]["messages"][1]["content"] == "summarize a.md"
    assert record["request"]["body"]["tools"][0]["function"]["name"] == "read_file"
    message = record["response"]["body"]["choices"][0]["message"]
    assert message["content"] == "Checking "
    assert message["tool_calls"][0]["function"] == {"name": "read_file", "arguments": '{"path":"a.md"}'}
    assert "secret-account-name" not in store.export_jsonl(session_id)
