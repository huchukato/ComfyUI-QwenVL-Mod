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
_folder_paths.get_input_directory = lambda: _tmp
_folder_paths.get_save_image_path = lambda prefix, out, w, h: (_tmp, prefix.rstrip("/"), 1, "", prefix)
sys.modules["folder_paths"] = _folder_paths

import QwenVL_Livepeer as lp
import QwenVL_LoadMedia as lm


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
            node = lp.QwenVL_LivepeerRender()
            out = node.run(
                prompt="slow dolly-in, she turns to camera",
                capability="auto", custom_capability="", duration=5,
                resolution="default", aspect_ratio="auto", seed=-1,
                timeout_s=120, filename_prefix="Livepeer/",
                image=self._image(), api_key="",
            )

        tools = [c[0] for c in calls]
        self.assertEqual(tools, ["upload", "describe_capability", "run_capability", "get_create_media"])
        rc = calls[2][1]
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
            node = lp.QwenVL_LivepeerRender()
            out = node.run(
                prompt="a lantern drifting over dark water",
                capability="auto", custom_capability="", duration=8,
                resolution="768P", aspect_ratio="16:9", seed=42,
                timeout_s=120, filename_prefix="Livepeer/",
            )

        self.assertEqual([c[0] for c in calls], ["describe_capability", "run_capability"])
        rc = calls[1][1]
        self.assertEqual(rc["capability"], "minimax-h3-t2v")
        self.assertNotIn("source_url", rc)
        self.assertEqual(rc["inputs"]["resolution"], "768P")
        self.assertEqual(rc["inputs"]["seed"], 42)
        self.assertEqual(out["result"][1], "https://cdn/v.mp4")

    def test_image_capability_drops_video_inputs(self):
        calls = []

        def fake_call(tool, args, api_key="", **kw):
            calls.append((tool, args))
            if tool == "describe_capability":
                return {"output_kind": "image", "inputs": {"prompt": {"type": "string"}}}
            if tool == "run_capability":
                return {"job_id": "j1", "status": "done", "url": "https://cdn/out.png"}
            return {}

        buf = io.BytesIO()
        from PIL import Image
        Image.new("RGB", (4, 4)).save(buf, "PNG")
        with mock.patch.object(lp, "_mcp_call", side_effect=fake_call), \
             mock.patch("urllib.request.urlopen", return_value=_Resp(buf.getvalue())), \
             mock.patch("time.sleep"):
            node = lp.QwenVL_LivepeerRender()
            out = node.run(
                prompt="anime girl", capability="auto", custom_capability="flux-schnell",
                duration=5, resolution="768P", aspect_ratio="16:9", seed=-1,
                timeout_s=120, filename_prefix="Livepeer/",
            )

        rc = next(c[1] for c in calls if c[0] == "run_capability")
        self.assertEqual(rc["capability"], "flux-schnell")
        self.assertEqual(rc["inputs"], {})
        self.assertEqual(out["result"][1], "https://cdn/out.png")

    def test_failed_job_raises(self):
        def fake_call(tool, args, api_key="", **kw):
            if tool == "run_capability":
                return {"job_id": "mjob_000000000002", "status": "queued"}
            if tool == "get_create_media":
                return {"status": "failed", "error": "provider timeout"}
            return {}

        with mock.patch.object(lp, "_mcp_call", side_effect=fake_call), \
             mock.patch("time.sleep"):
            node = lp.QwenVL_LivepeerRender()
            with self.assertRaises(RuntimeError):
                node.run(
                    prompt="x", capability="pixverse-i2v", custom_capability="",
                    duration=5, resolution="default", aspect_ratio="auto",
                    seed=-1, timeout_s=120, filename_prefix="Livepeer/",
                )

    def test_empty_prompt_rejected(self):
        with self.assertRaises(ValueError):
            lp.QwenVL_LivepeerRender().run(
                prompt="   ", capability="auto", custom_capability="",
                duration=5, resolution="default", aspect_ratio="auto",
                seed=-1, timeout_s=120, filename_prefix="Livepeer/",
            )


class LoadMediaTests(unittest.TestCase):
    def test_lists_tagged_media_files(self):
        Path(_tmp, "clip.mp4").write_bytes(b"x")
        Path(_tmp, "skip.txt").write_text("x")
        files = lm._media_files()
        self.assertIn("clip.mp4 [output]", files)
        self.assertNotIn("skip.txt", files)

    def test_resolves_tagged_name(self):
        Path(_tmp, "frame.png").write_bytes(b"x")
        path, tag = lm._resolve("frame.png [output]")
        self.assertEqual(tag, "output")
        self.assertTrue(path.endswith("frame.png"))
        self.assertEqual(lm._resolve("missing.png [output]"), (None, None))
        # Untagged names (set by the upload widget) resolve via input/output dirs
        path2, tag2 = lm._resolve("frame.png")
        self.assertEqual(tag2, "input")

    def test_loads_image_as_tensor(self):
        from PIL import Image
        Image.new("RGB", (4, 4)).save(Path(_tmp, "pic.png"))
        out = lm.QwenVL_LoadMedia().load("pic.png [output]")
        image, video, path = out["result"]
        self.assertEqual(tuple(image.shape), (1, 4, 4, 3))
        self.assertIsNone(video)
        self.assertTrue(path.endswith("pic.png"))
        self.assertEqual(out["ui"]["images"][0]["filename"], "pic.png")


if __name__ == "__main__":
    unittest.main()
