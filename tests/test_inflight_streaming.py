"""In-flight (still streaming) trace snapshots: store, forward proxy, dashboard."""

from __future__ import annotations

import asyncio
import json
import shutil
import ssl
import subprocess
import textwrap
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web
from yarl import URL

from claude_tap import forward_proxy
from claude_tap.certs import CertificateAuthority, ensure_ca
from claude_tap.forward_proxy import ForwardProxyServer
from claude_tap.live import LiveViewerServer
from claude_tap.trace import TraceWriter
from claude_tap.trace_store import TraceStore, get_trace_store


def _record(request_id: str, text: str, *, turn: int = 1) -> dict:
    return {
        "timestamp": "2026-11-06T10:00:00+00:00",
        "request_id": request_id,
        "turn": turn,
        "request": {"method": "POST", "path": "/v2/chat/completions", "body": {"model": "m"}},
        "response": {"status": 200, "body": {"choices": [{"message": {"role": "assistant", "content": text}}]}},
    }


def _chunk(text: str) -> bytes:
    return b"data: " + json.dumps({"id": "c1", "choices": [{"delta": {"content": text}}]}).encode() + b"\n\n"


def _reply_text(record: dict) -> str:
    body = record["response"]["body"] or {}
    return body["choices"][0]["message"]["content"]


# ── store ──────────────────────────────────────────────────────────────────


def test_inflight_upsert_bumps_revision_and_final_record_replaces_it(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "t.sqlite3")
    session_id = store.create_session(client="workbuddy", proxy_mode="forward")

    store.upsert_inflight_record(session_id, _record("req_a", "Hel"))
    store.upsert_inflight_record(session_id, _record("req_a", "Hello"))
    store.upsert_inflight_record(session_id, _record("req_b", "Other"))

    inflight = store.load_inflight_records(session_id)
    assert [(r["request_id"], r["inflight_revision"], _reply_text(r)) for r in inflight] == [
        ("req_a", 2, "Hello"),
        ("req_b", 1, "Other"),
    ]
    assert store.inflight_snapshot() == {session_id: (2, 3)}

    store.append_record(session_id, _record("req_a", "Hello world"))

    assert [r["request_id"] for r in store.load_inflight_records(session_id)] == ["req_b"]
    assert [_reply_text(r) for r in store.load_records(session_id)] == ["Hello world"]
    # In-flight rows never count as records.
    assert store.load_session_row(session_id)["record_count"] == 1


def test_inflight_rows_are_dropped_when_session_finishes(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "t.sqlite3")
    session_id = store.create_session(client="workbuddy", proxy_mode="forward")
    other_id = store.create_session(client="workbuddy", proxy_mode="forward")
    store.upsert_inflight_record(session_id, _record("req_a", "partial"))
    store.upsert_inflight_record(other_id, _record("req_b", "partial"))

    store.finalize_session(session_id, {"api_calls": 0})
    store.delete_inflight_record(other_id, "req_b")

    assert store.load_inflight_records(session_id) == []
    assert store.load_inflight_records(other_id) == []
    assert store.inflight_snapshot() == {}


def test_inflight_requires_request_id(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "t.sqlite3")
    session_id = store.create_session()
    with pytest.raises(ValueError):
        store.upsert_inflight_record(session_id, {"request": {}})


# ── forward proxy ──────────────────────────────────────────────────────────


class _ControlledUpstream:
    """HTTPS upstream whose SSE stream advances only when the test releases it."""

    def __init__(self) -> None:
        self.headers_gate = asyncio.Event()
        self.steps: dict[str, list[asyncio.Event]] = {}
        self.finished: dict[str, asyncio.Event] = {}

    def gate(self, name: str, count: int) -> list[asyncio.Event]:
        self.steps[name] = [asyncio.Event() for _ in range(count)]
        self.finished[name] = asyncio.Event()
        return self.steps[name]

    async def handler(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        name = body["messages"][0]["content"]
        await self.headers_gate.wait()
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        try:
            for index, step in enumerate(self.steps[name]):
                await step.wait()
                await resp.write(_chunk(f"{name}-{index} "))
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
        except (ConnectionError, RuntimeError):
            pass
        finally:
            self.finished[name].set()
        return resp


async def _start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, upstream: _ControlledUpstream):
    monkeypatch.setattr(forward_proxy, "INFLIGHT_UPDATE_INTERVAL_SECONDS", 0.0)
    upstream_ca = CertificateAuthority(*ensure_ca(tmp_path / "upstream-ca"))
    app = web.Application()
    app.router.add_post("/v2/chat/completions", upstream.handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0, ssl_context=upstream_ca.make_ssl_context("127.0.0.1"))
    await site.start()
    upstream_port = site._server.sockets[0].getsockname()[1]

    store = TraceStore(tmp_path / "t.sqlite3")
    session_id = store.create_session(client="workbuddy", proxy_mode="forward")
    writer = TraceWriter(session_id, store=store)

    upstream_ssl = ssl.create_default_context()
    upstream_ssl.check_hostname = False
    upstream_ssl.verify_mode = ssl.CERT_NONE
    session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=upstream_ssl), auto_decompress=False)
    original_request = session.request

    async def _to_upstream(method, url, **kwargs):
        return await original_request(method=method, url=URL(str(url)).with_host("127.0.0.1"), **kwargs)

    monkeypatch.setattr(session, "request", _to_upstream)
    ca_cert_path, ca_key_path = ensure_ca(tmp_path / "ca")
    server = ForwardProxyServer(
        host="127.0.0.1",
        port=0,
        ca=CertificateAuthority(ca_cert_path, ca_key_path),
        writer=writer,
        session=session,
        trace_methods=("POST",),
        trace_path_suffixes=("/chat/completions",),
    )
    proxy_port = await server.start()

    async def cleanup() -> None:
        await server.stop()
        await session.close()
        await runner.cleanup()

    client_ssl = ssl.create_default_context(cafile=str(ca_cert_path))
    return store, session_id, writer, upstream_port, proxy_port, client_ssl, cleanup


async def _wait_for(predicate, timeout: float = 5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.02)


def _request_json(name: str) -> dict:
    return {"model": "m", "stream": True, "messages": [{"role": "user", "content": name}]}


@pytest.mark.asyncio
async def test_forward_proxy_publishes_request_and_partial_reply_before_stream_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream = _ControlledUpstream()
    steps = upstream.gate("alpha", 2)
    store, session_id, writer, upstream_port, proxy_port, client_ssl, cleanup = await _start(
        tmp_path, monkeypatch, upstream
    )
    try:
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=client_ssl)) as client:
            response_task = asyncio.create_task(
                client.post(
                    f"https://www.workbuddy.cn:{upstream_port}/v2/chat/completions",
                    proxy=f"http://127.0.0.1:{proxy_port}",
                    json=_request_json("alpha"),
                )
            )
            # Before the upstream answers, the request is already visible.
            waiting = await _wait_for(lambda: store.load_inflight_records(session_id))
            assert waiting[0]["stream_state"] == "waiting"
            assert waiting[0]["request"]["body"]["messages"][0]["content"] == "alpha"
            assert waiting[0]["response"]["status"] is None

            upstream.headers_gate.set()
            steps[0].set()
            partial = await _wait_for(
                lambda: [r for r in store.load_inflight_records(session_id) if r.get("stream_state") == "streaming"]
            )
            assert _reply_text(partial[0]) == "alpha-0 "
            assert store.load_records(session_id) == []

            steps[1].set()
            resp = await response_task
            assert resp.status == 200
            await resp.read()
            resp.release()

        final = await _wait_for(lambda: store.load_records(session_id))
        assert len(final) == 1
        assert _reply_text(final[0]) == "alpha-0 alpha-1 "
        assert "stream_state" not in final[0]
        assert store.load_inflight_records(session_id) == []
    finally:
        writer.close()
        await cleanup()


@pytest.mark.asyncio
async def test_forward_proxy_keeps_partial_reply_when_client_cancels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream = _ControlledUpstream()
    steps = upstream.gate("beta", 3)
    upstream.headers_gate.set()
    store, session_id, writer, upstream_port, proxy_port, client_ssl, cleanup = await _start(
        tmp_path, monkeypatch, upstream
    )
    try:
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=client_ssl)) as client:
            resp = await client.post(
                f"https://www.workbuddy.cn:{upstream_port}/v2/chat/completions",
                proxy=f"http://127.0.0.1:{proxy_port}",
                json=_request_json("beta"),
            )
            steps[0].set()
            assert (await resp.content.readuntil(b"\n\n")).startswith(b"data: ")
            # User presses stop: the client drops the connection mid-stream,
            # while the provider would keep generating.
            resp.close()
        await asyncio.sleep(0.2)
        steps[1].set()
        steps[2].set()

        final = await _wait_for(lambda: store.load_records(session_id))
        assert final[0]["stream_state"] == "client_cancelled"
        assert _reply_text(final[0]).startswith("beta-0 ")
        assert "beta-2" not in _reply_text(final[0])
        assert store.load_inflight_records(session_id) == []
    finally:
        writer.close()
        await cleanup()


@pytest.mark.asyncio
async def test_forward_proxy_concurrent_streams_do_not_mix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = _ControlledUpstream()
    one = upstream.gate("one", 2)
    two = upstream.gate("two", 2)
    upstream.headers_gate.set()
    store, session_id, writer, upstream_port, proxy_port, client_ssl, cleanup = await _start(
        tmp_path, monkeypatch, upstream
    )
    try:
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=client_ssl)) as client:

            async def run(name: str) -> bytes:
                async with client.post(
                    f"https://www.workbuddy.cn:{upstream_port}/v2/chat/completions",
                    proxy=f"http://127.0.0.1:{proxy_port}",
                    json=_request_json(name),
                ) as resp:
                    return await resp.read()

            tasks = [asyncio.create_task(run("one")), asyncio.create_task(run("two"))]
            one[0].set()
            two[0].set()

            def both_streaming():
                rows = store.load_inflight_records(session_id)
                texts = {r["request"]["body"]["messages"][0]["content"]: r for r in rows if r.get("stream_state")}
                if all(texts.get(n, {}).get("stream_state") == "streaming" for n in ("one", "two")):
                    return texts
                return None

            snapshots = await _wait_for(both_streaming)
            assert _reply_text(snapshots["one"]) == "one-0 "
            assert _reply_text(snapshots["two"]) == "two-0 "

            two[1].set()
            one[1].set()
            await asyncio.gather(*tasks)

        final = await _wait_for(lambda: len(store.load_records(session_id)) == 2 and store.load_records(session_id))
        by_name = {r["request"]["body"]["messages"][0]["content"]: _reply_text(r) for r in final}
        assert by_name == {"one": "one-0 one-1 ", "two": "two-0 two-1 "}
        assert len({r["request_id"] for r in final}) == 2
        assert store.load_inflight_records(session_id) == []
    finally:
        writer.close()
        await cleanup()


# ── dashboard ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dashboard_serves_redacted_inflight_records_and_emits_events(
    trace_db, monkeypatch: pytest.MonkeyPatch
) -> None:
    from claude_tap import live

    monkeypatch.setattr(live, "_DASHBOARD_WATCH_INTERVAL_SECONDS", 0.05)
    store = get_trace_store()
    session_id = store.create_session(client="workbuddy", proxy_mode="forward")
    server = LiveViewerServer(port=0, dashboard_mode=True)
    port = await server.start()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{port}/dashboard/events") as events:
                assert (await events.content.readuntil(b"\n\n")).startswith(b"event: ready")

                record = _record("req_live", "Hel")
                record["request"]["body"]["access_token"] = "access-secret"
                record["stream_state"] = "streaming"
                store.upsert_inflight_record(session_id, record)

                event = await asyncio.wait_for(events.content.readuntil(b"\n\n"), timeout=5)
                assert event.startswith(b"event: inflight")
                assert json.loads(event.split(b"data: ", 1)[1]) == {"type": "inflight", "session_id": session_id}

            async with session.get(f"http://127.0.0.1:{port}/api/sessions/{session_id}/inflight") as resp:
                payload = await resp.json()
            async with session.get(f"http://127.0.0.1:{port}/api/sessions/missing/inflight") as resp:
                assert resp.status == 404
    finally:
        await server.stop()

    [served] = payload["records"]
    assert served["request_id"] == "req_live"
    assert served["stream_state"] == "streaming"
    assert _reply_text(served) == "Hel"
    assert "access-secret" not in json.dumps(payload)


# ── dashboard rendering ────────────────────────────────────────────────────

_DASHBOARD_HTML = Path(__file__).resolve().parents[1] / "claude_tap" / "dashboard.html"
_RENDER_FUNCTIONS = (
    "streamStateBadge",
    "toolCallsText",
    "thinkingText",
    "renderThinking",
    "renderRecord",
    "systemPromptText",
    "requestText",
    "responseTextFromBody",
    "responseTextFromRecord",
    "contentText",
    "partsText",
)


def _dashboard_function_source(html: str, name: str) -> str:
    start = html.index(f"function {name}(")
    depth = 0
    for index in range(html.index("{", html.index(")", start)), len(html)):
        if html[index] == "{":
            depth += 1
        elif html[index] == "}":
            depth -= 1
            if depth == 0:
                return html[start : index + 1]
    raise AssertionError(f"unterminated function {name}")


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for dashboard JS unit tests")
def test_dashboard_shows_thinking_while_the_reply_has_no_text_yet() -> None:
    html = _DASHBOARD_HTML.read_text(encoding="utf-8")
    functions = "\n".join(_dashboard_function_source(html, name) for name in _RENDER_FUNCTIONS)
    script = (
        "const assert = require('assert/strict');\n"
        "const esc = value => String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;');\n"
        "const t = (key, params = {}) => key === 'chars' ? `${params.count} chars` : key;\n"
        + functions
        + textwrap.dedent(
            r"""
            const request = {path: '/v2/chat/completions', body: {messages: [{role: 'user', content: 'hi'}]}};
            const thinkingOnly = {
              request_id: 'req_live', stream_state: 'streaming', request,
              response: {status: 200, body: {
                choices: [{message: {role: 'assistant', content: '', reasoning_content: 'step <one>'}}],
                content: [{type: 'thinking', thinking: 'step <one>'}],
              }},
            };
            // Message-style requests show the system prompt before the first user message,
            // like Responses-style requests that carry it inside `input`.
            const chat = {messages: [{role: 'system', content: 'SYS'}, {role: 'user', content: 'hi'}, {role: 'user', content: 'later'}]};
            assert.equal(requestText(chat), 'SYS\n\nhi');
            assert.equal(requestText({system: [{type: 'text', text: 'ANTHROPIC'}], messages: [{role: 'user', content: 'hi'}]}), 'ANTHROPIC\n\nhi');
            assert.equal(requestText({messages: [{role: 'user', content: 'hi'}]}), 'hi');
            assert.equal(requestText({input: [{role: 'developer', content: [{type: 'input_text', text: 'DEV'}]}]}), 'DEV');

            // The model can think for a minute before any answer text: show it, open, not duplicated.
            assert.equal(thinkingText(thinkingOnly.response.body), 'step <one>');
            const live = renderRecord(thinkingOnly, 'inflight-0', {inflight: true});
            assert.match(live, /<details class="thinking-block" open>/);
            assert.match(live, /thinking · 10 chars/);
            assert.match(live, /step &lt;one>/);
            assert.ok(!live.includes('response_empty'));

            // Anthropic-shape bodies only carry thinking blocks.
            assert.equal(thinkingText({content: [{type: 'thinking', thinking: 'a'}, {type: 'text', text: 'b'}]}), 'a');
            assert.equal(thinkingText({choices: [{message: {content: 'plain'}}]}), '');
            assert.equal(thinkingText(null), '');

            // Once the answer exists (and in finished records) thinking is collapsed behind its summary.
            const done = JSON.parse(JSON.stringify(thinkingOnly));
            delete done.stream_state;
            done.response.body.choices[0].message.content = 'Hello';
            const finished = renderRecord(done, 0);
            assert.match(finished, /<details class="thinking-block">/);
            assert.match(finished, /<div class="content-block">Hello<\/div>/);
            assert.ok(!finished.includes('response_empty'));
            """
        )
    )
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
