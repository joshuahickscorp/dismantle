from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from hcli.agentos import AgentOS
from hcli.backends import OpenAICompatibleBackend
from hcli.controller import Controller
from hcli.flash_next import PINNED_REVISION, REPO_ID, flash_next_profile
from hcli.models import resolve_model
from hcli.providers import CapabilityContract, ResidentProfile, RoleRouter
from hcli.runtime_iface import classify_backend, model_semantics_for
from hcli.tool_registry import ToolSpec, WORKSPACE_WRITE, default_tool_registry
from hcli.workunit import WorkUnit


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        del fmt, args

    def do_GET(self):
        body = b'{"status":"ok"}' if self.path in {"/health", "/v1/models"} else b"{}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        size = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(size))
        self.server.last_payload = payload  # type: ignore[attr-defined]
        content = "remote answer"
        if getattr(self.server, "valid_response", False):  # type: ignore[attr-defined]
            content = json.dumps({
                "kind": "answer",
                "content": "controller remote ok",
                "operations": [],
                "tests": [],
            })
        body = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _GenericResidentEngine:
    def __init__(self):
        self.calls = []

    def execute_workunit(self, unit, context):
        self.calls.append((unit.id, context.get("provider")))
        return {
            "content": f"handled by {unit.id}",
            "validation": {"ok": True, "verifier": "fixture"},
        }

    def identity(self):
        return {
            "provider": "fixture-resident",
            "model_id": "fixture-model",
            "runtime": "fixture-runtime",
        }


class AgentOSGeneralTest(unittest.TestCase):
    def test_remote_endpoint_is_a_first_class_provider(self):
        server = HTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            self.assertEqual(classify_backend(url), "remote")
            self.assertEqual(model_semantics_for(url).backend_kind, "remote")
            self.assertEqual(resolve_model(explicit=url).provider, "remote")
            with self.assertRaises(ValueError):
                OpenAICompatibleBackend("https://user:secret@example.com")
            selected = resolve_model(explicit=f"{url}?model=fixture-model&api_key=must-not-persist")
            self.assertIn("#model=fixture-model", selected.path)
            self.assertNotIn("must-not-persist", selected.path)
            backend = OpenAICompatibleBackend(url)
            backend.spawn()
            self.assertTrue(backend.ready(1))
            result = backend.complete({
                "messages": [{"role": "user", "content": "hello"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "answer", "schema": {"type": "object"}},
                },
                "grammar": "root ::= object",
                "chat_template_kwargs": {"enable_thinking": True},
            }, timeout=2)
            self.assertEqual(result.text, "remote answer")
            self.assertEqual(result.degraded, ["response_format", "grammar", "chat_template_kwargs"])
            self.assertEqual(server.last_payload["model"], "remote")  # type: ignore[attr-defined]
            self.assertNotIn("response_format", server.last_payload)  # type: ignore[attr-defined]
            self.assertNotIn("grammar", server.last_payload)  # type: ignore[attr-defined]
            self.assertNotIn("chat_template_kwargs", server.last_payload)  # type: ignore[attr-defined]
        finally:
            server.shutdown()
            server.server_close()

    def test_remote_pool_topology_keeps_endpoint_opaque(self):
        server = HTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        from hcli.runtime import RuntimePool
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            raw = f"{url}?model=fixture-model&api_key=must-not-persist"
            pool = RuntimePool(raw, requested_n=1, workspace=".", repo_root=".", topology="process")
            try:
                pool.start()
                field = pool.runtimes[0].topology["model_path"]
                self.assertEqual(field["value"], f"{url}#model=fixture-model")
                self.assertNotIn("must-not-persist", json.dumps(pool._read_ownership() or {}))
            finally:
                pool.stop()
        finally:
            server.shutdown()
            server.server_close()

    def test_controller_executes_through_a_remote_provider_without_qwen_assumptions(self):
        server = HTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                # The controller's engine contract is intentionally still
                # the same JSON result contract for every provider.
                server.valid_response = True  # type: ignore[attr-defined]
                selector = f"http://127.0.0.1:{server.server_port}?model=fixture-model&api_key=must-not-persist"
                controller = Controller(root, model=selector)
                try:
                    self.assertEqual(controller.status()["provider"], "remote")
                    self.assertIn("#model=fixture-model", controller.status()["model"])
                    result = controller.execute("say hello")
                    self.assertEqual(result["content"], "controller remote ok")
                    self.assertEqual(controller.status()["runtime"]["provider"], "remote")
                    self.assertIn("#model=fixture-model", controller.session.model)
                    self.assertNotIn("must-not-persist", controller.session.model)
                    self.assertEqual(server.last_payload["model"], "fixture-model")  # type: ignore[attr-defined]
                    session_files = list((root / ".hcli" / "sessions").glob("*.json"))
                    self.assertTrue(session_files)
                    persisted = "\n".join(path.read_text(encoding="utf-8") for path in session_files)
                    self.assertNotIn("must-not-persist", persisted)
                finally:
                    controller.shutdown()
        finally:
            server.shutdown()
            server.server_close()

    def test_typed_tools_are_discoverable_bounded_and_redacted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "note.txt").write_text("alpha\nbeta\n", encoding="utf-8")
            registry = default_tool_registry(root, repo_root=root)
            names = {item["name"] for item in registry.discover()}
            self.assertIn("fs.read", names)
            self.assertIn("huggingface.resolve", names)
            read = registry.invoke("fs.read", {"path": "note.txt"})
            self.assertTrue(read.ok)
            self.assertEqual(read.value["sha256"], read.to_dict()["value"]["sha256"])
            denied = registry.register(ToolSpec(
                "test.write", "fixture", {"type": "object"}, mutation=WORKSPACE_WRITE,
                handler=lambda _context, _args: {"changed": True},
            ))
            self.assertEqual(denied.name, "test.write")
            result = registry.invoke("test.write", {})
            self.assertFalse(result.ok)
            self.assertEqual(result.failure_class, "PERMISSION_DENIED")
            bad = registry.invoke("fs.read", {"path": "../outside.txt"})
            self.assertFalse(bad.ok)
            self.assertEqual(bad.failure_class, "PermissionError")
            secret = registry.register(ToolSpec(
                "test.redact", "redaction fixture", {"type": "object"},
                handler=lambda _context, _args: {
                    "token_count": 4,
                    "api_key": "sk-test-secret",
                    "line": "OPENAI_API_KEY=sk-test-secret",
                },
            ))
            self.assertEqual(secret.name, "test.redact")
            redacted = registry.invoke("test.redact", {}).to_dict()
            self.assertEqual(redacted["value"]["token_count"], 4)
            self.assertEqual(redacted["value"]["api_key"], "[REDACTED]")
            self.assertNotIn("sk-test-secret", json.dumps(redacted))
            private_url = registry.invoke("web.fetch", {"url": "https://example.com/?token=secret"})
            self.assertFalse(private_url.ok)
            self.assertEqual(private_url.failure_class, "PermissionError")

    def test_agentos_mission_checkpoint_and_restart_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / "marker.txt"
            marker.write_text("ok", encoding="utf-8")
            unit = WorkUnit(
                id="verify-marker",
                role="verifier",
                description="verify marker",
                resource_class="TEST",
                preferred_backend="cpu",
                verifier=f"test -f {marker}",
            )
            agent = AgentOS(root, engine=object(), repo_root=root)
            mission = agent.start_mission("verify the marker", units={unit.id: unit})
            result = agent.run()
            self.assertEqual(result["status"], "completed")
            envelope = agent.result_envelope(result)
            self.assertEqual(envelope["verdict"], "ACCEPT")
            self.assertEqual(envelope["tests"]["checks"][0]["unit_id"], "verify-marker")
            self.assertTrue((root / ".hcli" / "mission" / "state.json").is_file())
            restarted = AgentOS(root, engine=object(), repo_root=root)
            restored = restarted.recover_mission()
            self.assertEqual(restored.id, mission.id)
            self.assertEqual(restored.scheduler.units[unit.id].status, "completed")
            self.assertEqual(restarted.recovery_status()["checkpoint_id"], restored._last_checkpoint_id)
            tool = restarted.invoke_tool("fs.read", {"path": "marker.txt"})
            self.assertTrue(tool.ok)
            self.assertTrue(Path(tool.provenance["receipt_path"]).is_file())

    def test_explicit_resident_role_uses_generic_provider_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            unit = WorkUnit(
                id="resident-unit",
                role="generalist",
                description="use the explicitly selected resident",
                resource_class="TEST",
                preferred_backend="resident",
                provider="resident",
            )
            engine = _GenericResidentEngine()
            agent = AgentOS(root, engine=engine, repo_root=root)
            agent.start_mission("run through resident provider", units={unit.id: unit})
            result = agent.run()
            self.assertEqual(result["status"], "completed")
            raw = agent.mission.scheduler.units[unit.id].verification
            self.assertTrue(raw["ok"])
            self.assertEqual(agent.mission.scheduler.units[unit.id].assigned_backend, "resident")
            self.assertEqual(engine.calls, [("resident-unit", "resident")])

    def test_flash_next_is_pinned_but_not_qualified(self):
        report = flash_next_profile()
        profile = report["profile"]
        self.assertEqual(profile["model_id"], REPO_ID)
        self.assertEqual(profile["artifact"]["pinned_revision"], PINNED_REVISION)
        self.assertEqual(profile["qualification"]["status"], "IDENTITY_ONLY")
        self.assertFalse(report["download_performed"])
        self.assertFalse(report["model_lake"]["hash_verified"])
        self.assertEqual(report["model_lake"]["hash_status"], "NOT_RUN")
        self.assertEqual(len(report["organ_census"]), 7)

    def test_roles_do_not_choose_a_model_name(self):
        choice = RoleRouter().choose("science", {"specialist": object()})
        self.assertEqual(choice["provider"], "specialist")
        self.assertEqual(choice["role"], "science")

    def test_roles_honor_declared_capabilities(self):
        vision = ResidentProfile(
            profile_id="vision-profile",
            provider="multimodal",
            model_id="vision-model",
            capabilities=CapabilityContract.from_mapping({
                "features": {"vision": {"state": "supported"}}
            }),
        )
        choice = RoleRouter().choose("vision", {"multimodal": vision})
        self.assertEqual(choice["provider"], "multimodal")
        self.assertIsNone(RoleRouter().choose("vision", {"remote": object()})["provider"])
        self.assertEqual(RoleRouter().choose("verifier", {"cpu": object()})["provider"], "cpu")


if __name__ == "__main__":
    unittest.main()
