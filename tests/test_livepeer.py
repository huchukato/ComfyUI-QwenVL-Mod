import base64
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parents[1]))

_tmp = tempfile.mkdtemp()
_folder_paths = types.ModuleType("folder_paths")
_folder_paths.get_output_directory = lambda: _tmp
_folder_paths.get_save_image_path = lambda prefix, out, w, h: (_tmp, prefix.rstrip("/"), 1, "", prefix)
sys.modules["folder_paths"] = _folder_paths

import AILab_Livepeer as lp


def _envelope(result):
    return json.dumps({"jsonrpc": "2.0", "id": 1, "result": result}).encode()


class _Resp:
    def __init__(self, body):
        self._body = body if isinstance(body, bytes) else body.encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class McpClientTests(unittest.TestCase):
    def test_parses_structured_content(self):
        payload = _envelope({"structuredContent": {"url": "https://x/v.mp4"}, "content": []})
        with mock.patch("urllib.request.urlopen", return_value=_Resp(payload)):
            out = lp._mcp_call("me", {})
        self.assertEqual(out["url"], "https://x/v.mp4")

    def test_parses_sse_frames(self):
        inner = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"structuredContent": {"status": "done"}, "content": []}})
        body = f"event: message\ndata: {inner}\n\n"
        with mock.patch("urllib.request.urlopen", return_value=_Resp(body)):
            out = lp._mcp_call("get_create_media", {"job_id": "mjob_abcdef123456"})
        self.assertEqual(out["status"], "done")

    def test_raises_on_tool_error(self):
        payload = _envelope({"isError": True, "structuredContent": {"error": {"message": "bad cap"}}, "content": []})
        with mock.patch("urllib.request.urlopen", return_value=_Resp(payload)):
            with self.assertRaises(RuntimeError):
                lp._mcp_call("run_capability", {})

    def test_extract_url_nested(self):
        self.assertEqual(lp._extract_url({"result": {"url": "https://a/b.mp4"}}), "https://a/b.mp4")
        self.assertEqual(lp._extract_url({"text": "done: https://a/b.mp4."}), "https://a/b.mp4")
        self.assertIsNone(lp._extract_url({"status": "running"}))

    def test_resolve_capability(self):
        self.assertEqual(lp._resolve_capability("auto", "", True), "minimax-h3-i2v")
        self.assertEqual(lp._resolve_capability("auto", "", False), "minimax-h3-t2v")
        self.assertEqual(lp._resolve_capability("pixverse-i2v", "", True), "pixverse-i2v")
        self.assertEqual(lp._resolve_capability("auto", " my-cap ", True), "my-cap")


class RenderNodeTests(unittest.TestCase):
    def _image(self):
        import torch
        return torch.rand(1, 64, 64, 3)

    def test_tensor_to_jpeg_under_limit(self):
        b64 = lp._tensor_to_jpeg_b64(self._image())
        self.assertLessEqual(len(base64.b64decode(b64)), 2_500_000)

    def test_run_i2v_end_to_end(self):
        calls = []

        def fake_call(tool, args, api_key="", **kw):
            calls.append((tool, args))
            if tool == "upload":
                return {"url": "https://cdn/frame.jpg"}
            if tool == "run_capability":
                return {"job_id": "mjob_abcdef123456", "status": "queued"}
            if tool == "get_create_media":
                return {"status": "done", "url": "https://cdn/out.mp4"}
            return {}

        def fake_dl(req, timeout=0):
            return _Resp(b"\x00\x01mp4data")

        with mock.patch.object(lp, "_mcp_call", side_effect=fake_call), \
             mock.patch("urllib.request.urlopen", side_effect=fake_dl), \
             mock.patch("time.sleep"):
            node = lp.AILab_LivepeerRender()
            out = node.run(
                prompt="slow dolly-in, she turns to camera",
                capability="auto", custom_capability="", duration=5,
                resolution="default", aspect_ratio="auto", seed=-1,
                timeout_s=120, filename_prefix="Livepeer/",
                image=self._image(), api_key="",
            )

        tools = [c[0] for c in calls]
        self.assertEqual(tools, ["upload", "run_capability", "get_create_media"])
        rc = calls[1][1]
        self.assertEqual(rc["capability"], "minimax-h3-i2v")
        self.assertEqual(rc["source_url"], "https://cdn/frame.jpg")
        self.assertEqual(rc["inputs"]["duration"], 5)
        self.assertTrue(rc["async"])
        result = out["result"]
        self.assertEqual(result[1], "https://cdn/out.mp4")
        report = json.loads(result[2])
        self.assertEqual(report["capability"], "minimax-h3-i2v")
        self.assertEqual(report["job_id"], "mjob_abcdef123456")
        self.assertIn("images", out["ui"])

    def test_run_t2v_no_upload(self):
        calls = []

        def fake_call(tool, args, api_key="", **kw):
            calls.append((tool, args))
            if tool == "run_capability":
                return {"job_id": "mjob_000000000001", "status": "done", "url": "https://cdn/v.mp4"}
            return {}

        with mock.patch.object(lp, "_mcp_call", side_effect=fake_call), \
             mock.patch("urllib.request.urlopen", return_value=_Resp(b"mp4")), \
             mock.patch("time.sleep"):
            node = lp.AILab_LivepeerRender()
            out = node.run(
                prompt="a lantern drifting over dark water",
                capability="auto", custom_capability="", duration=8,
                resolution="768P", aspect_ratio="16:9", seed=42,
                timeout_s=120, filename_prefix="Livepeer/",
            )

        self.assertEqual([c[0] for c in calls], ["run_capability"])
        rc = calls[0][1]
        self.assertEqual(rc["capability"], "minimax-h3-t2v")
        self.assertNotIn("source_url", rc)
        self.assertEqual(rc["inputs"]["resolution"], "768P")
        self.assertEqual(rc["inputs"]["seed"], 42)
        self.assertEqual(out["result"][1], "https://cdn/v.mp4")

    def test_failed_job_raises(self):
        def fake_call(tool, args, api_key="", **kw):
            if tool == "run_capability":
                return {"job_id": "mjob_000000000002", "status": "queued"}
            if tool == "get_create_media":
                return {"status": "failed", "error": "provider timeout"}
            return {}

        with mock.patch.object(lp, "_mcp_call", side_effect=fake_call), \
             mock.patch("time.sleep"):
            node = lp.AILab_LivepeerRender()
            with self.assertRaises(RuntimeError):
                node.run(
                    prompt="x", capability="pixverse-i2v", custom_capability="",
                    duration=5, resolution="default", aspect_ratio="auto",
                    seed=-1, timeout_s=120, filename_prefix="Livepeer/",
                )

    def test_empty_prompt_rejected(self):
        with self.assertRaises(ValueError):
            lp.AILab_LivepeerRender().run(
                prompt="   ", capability="auto", custom_capability="",
                duration=5, resolution="default", aspect_ratio="auto",
                seed=-1, timeout_s=120, filename_prefix="Livepeer/",
            )


if __name__ == "__main__":
    unittest.main()
