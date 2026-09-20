import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from chat_service import ChatRuntime, MINIMAX_I2VA_BINDING, THINK_CLOSE, THINK_OPEN, build_prompt, enforce_image_enhancer_routing, enforce_image_reference_bindings, list_output_images, parse_model_response, select_workflow_intent, validate_actions, validate_graph, validate_images, validate_messages


class ChatProtocolTests(unittest.TestCase):
    def test_validates_and_limits_messages(self):
        messages = [{"role": "user", "content": f"message {index}"} for index in range(25)]
        result = validate_messages(messages)
        self.assertEqual(len(result), 20)
        self.assertEqual(result[0]["content"], "message 5")

    def test_rejects_invalid_message_role_and_content(self):
        for messages in ([], [{"role": "system", "content": "x"}], [{"role": "user", "content": ""}]):
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                validate_messages(messages)

    def test_validates_graph_snapshot(self):
        graph = {"nodes": [{"id": 1, "type": "KSampler", "title": "Sampler", "mode": 0, "widgets": [{"name": "steps", "value": 20}]}]}
        result = validate_graph(graph)
        self.assertEqual(result["nodes"][0]["id"], 1)
        self.assertEqual(result["nodes"][0]["widgets"][0]["value"], 20)

    def test_rejects_invalid_graph(self):
        for graph in (None, {}, {"nodes": "bad"}, {"nodes": [{}]}):
            with self.subTest(graph=graph), self.assertRaises(ValueError):
                validate_graph(graph)

    def test_accepts_only_protocol_actions(self):
        actions = [
            {"type": "set_widget_value", "node_id": 1, "widget": "steps", "value": 25},
            {"type": "set_node_mode", "node_id": 2, "mode": "bypass"},
            {"type": "queue_workflow"},
            {"type": "delete_node", "node_id": 3},
            {"type": "set_node_mode", "node_id": 2, "mode": "delete"},
            {"type": "set_widget_value", "node_id": 1, "widget": "x", "value": {"unsafe": True}},
        ]
        self.assertEqual(len(validate_actions(actions)), 3)

    def test_parses_json_and_fenced_json(self):
        payload = {"message": "Done", "actions": [{"type": "queue_workflow"}]}
        for text in (json.dumps(payload), f"```json\n{json.dumps(payload)}\n```"):
            with self.subTest(text=text):
                result = parse_model_response(text)
                self.assertEqual(result["message"], payload["message"])
                self.assertEqual(result["actions"], payload["actions"])
                self.assertEqual(result["thinking"], "")

    def test_extracts_thinking_from_response(self):
        text = THINK_OPEN + "I should queue the workflow." + THINK_CLOSE + "\n" + json.dumps(
            {"message": "Done", "actions": [{"type": "queue_workflow"}]}
        )
        result = parse_model_response(text)
        self.assertEqual(result["thinking"], "I should queue the workflow.")
        self.assertEqual(result["message"], "Done")
        self.assertEqual(len(result["actions"]), 1)

    def test_plain_text_becomes_message_without_actions(self):
        self.assertEqual(parse_model_response("I cannot do that"), {"message": "I cannot do that", "actions": [], "choices": [], "thinking": "", "parsed": False})

    def test_prompt_contains_history_and_graph(self):
        prompt = build_prompt([{"role": "user", "content": "Set steps"}], {"nodes": [{"id": 1}]})
        self.assertIn("Set steps", prompt)
        self.assertIn('"id":1', prompt)
        self.assertIn("set_widget_value", prompt)
        self.assertIn("images", prompt)
        self.assertIn("Final generated image and video prompts MUST be in English", prompt)
        self.assertIn("MUST use the same language as the LATEST user message", prompt)
        self.assertIn("IMAGE + PRESET ENHANCER", prompt)
        self.assertIn("If queue_workflow is present, state that execution was started", prompt)

    def test_prompt_identifies_exact_promoted_passthrough_target(self):
        graph = {"nodes": [{"id": 105, "widgets": [
            {"name": "prompt", "value": ""},
            {"name": "preset_prompt", "value": "MiniMax H3 NSFW (5s)"},
            {"name": "passthrough", "value": False},
        ]}]}
        prompt = build_prompt([{"role": "user", "content": "Generate the video"}], graph)
        self.assertIn('Node 105 exposes both "prompt" and "passthrough"', prompt)
        self.assertIn('set node 105 widget "prompt"', prompt)
        self.assertIn('set node 105 widget "passthrough" to true', prompt)

    def test_prompt_uses_inner_enhancer_when_image_is_provided(self):
        graph = {"nodes": [{"id": 105, "title": "Image to Video (MiniMax H3)", "widgets": [
            {"name": "prompt", "value": ""},
            {"name": "preset_prompt", "value": "MiniMax H3 NSFW (5s)"},
            {"name": "passthrough", "value": True},
        ]}]}
        prompt = build_prompt([{"role": "user", "content": "Generate the video"}], graph, has_images=True)
        self.assertIn("Image pixels are provided", prompt)
        self.assertIn("Inspect the provided image pixels to understand how the requested action applies", prompt)
        self.assertIn('currently selects preset "MiniMax H3 NSFW (5s)"', prompt)
        self.assertIn("IGNORE any full prompt-writing guide for that preset", prompt)
        self.assertIn("the inner QwenVL node will use it to build the final prompt", prompt)
        self.assertIn('You MUST set node 105 widget "prompt" to a concise English action directive', prompt)
        self.assertIn('set node 105 widget "passthrough" to false', prompt)
        self.assertIn("inner QwenVL must analyze the image and create the final preset prompt", prompt)
        # The full MiniMax format guide must not leak into the chat prompt for an image enhancer.
        self.assertNotIn("integrated_multimodal_description:", prompt)
        self.assertNotIn("overall_soundscape:", prompt)

    def test_selects_previous_intent_after_execution_confirmation(self):
        descriptive = "Create a five-second video where she opens the dress"
        for confirmation in (
            "Generate the video", "Esegui il video", "Run it", "Genera", "Ok",
            "Sì", "Avvia", "Esegui il video con il prompt precedente", "Go ahead",
        ):
            messages = [
                {"role": "user", "content": descriptive},
                {"role": "assistant", "content": "Ready."},
                {"role": "user", "content": confirmation},
            ]
            with self.subTest(confirmation=confirmation):
                self.assertEqual(select_workflow_intent(messages), descriptive)

    def test_execution_only_detection_keeps_descriptive_requests(self):
        for content in (
            "genera un video in cui la ragazza balla",
            "crea un video di 5 secondi dall'immagine allegata",
            "la ragazza sposta la mano e si afferra il seno",
            "modifica il prompt per dire che lei sorride",
        ):
            with self.subTest(content=content):
                self.assertEqual(select_workflow_intent([{"role": "user", "content": content}]), content)

    def test_enforces_image_enhancer_prompt_and_disables_passthrough(self):
        graph = {"nodes": [{"id": 105, "title": "Image to Video (MiniMax H3)", "widgets": [
            {"name": "prompt", "value": "old prompt"},
            {"name": "preset_prompt", "value": "🎬 MiniMax H3 NSFW (5s)"},
            {"name": "passthrough", "value": True},
        ]}]}
        messages = [
            {"role": "user", "content": "La ragazza apre il vestito"},
            {"role": "assistant", "content": "Vuoi che lo esegua?"},
            {"role": "user", "content": "Esegui il video"},
        ]
        cases = [
            ([], messages[0]["content"]),
            ([{"type": "set_widget_value", "node_id": 105, "widget": "prompt", "value": "integrated_multimodal_description: [Shot 1] hallucinated scene"}], messages[0]["content"]),
            ([{"type": "set_widget_value", "node_id": 105, "widget": "prompt", "value": "Esegui il video"}], messages[0]["content"]),
        ]
        for prompt_action, expected in cases:
            with self.subTest(prompt_action=prompt_action):
                result = {
                    "message": "Done",
                    "actions": [
                        {"type": "set_widget_value", "node_id": 105, "widget": "preset_prompt", "value": "🎬 MiniMax H3 NSFW (5s)"},
                        *prompt_action,
                        {"type": "set_widget_value", "node_id": 105, "widget": "passthrough", "value": True},
                        {"type": "queue_workflow"},
                    ],
                }
                enforced = enforce_image_enhancer_routing(result, graph, messages, True)
                prompt_actions = [action for action in enforced["actions"] if action.get("widget") == "prompt"]
                self.assertEqual(len(prompt_actions), 1)
                self.assertEqual(prompt_actions[0]["value"], expected)
                self.assertFalse(next(action for action in enforced["actions"] if action.get("widget") == "passthrough")["value"])

    def test_keeps_model_directive_for_image_enhancer(self):
        graph = {"nodes": [{"id": 105, "title": "Image to Video (MiniMax H3)", "widgets": [
            {"name": "prompt", "value": "old prompt"},
            {"name": "preset_prompt", "value": "🎬 MiniMax H3 NSFW (5s)"},
            {"name": "passthrough", "value": True},
        ]}]}
        messages = [{"role": "user", "content": "la ragazza sposta la mano e si afferra il seno"}]
        directive = "The woman moves her hand and grabs her breast. Preserve the reference image exactly and change only this action."
        result = {
            "message": "Done",
            "actions": [
                {"type": "set_widget_value", "node_id": 105, "widget": "prompt", "value": directive},
                {"type": "set_widget_value", "node_id": 105, "widget": "passthrough", "value": True},
                {"type": "queue_workflow"},
            ],
        }
        enforced = enforce_image_enhancer_routing(result, graph, messages, True)
        prompt_action = next(action for action in enforced["actions"] if action.get("widget") == "prompt")
        self.assertEqual(prompt_action["value"], directive)
        self.assertIn(directive, enforced["message"])
        self.assertFalse(next(action for action in enforced["actions"] if action.get("widget") == "passthrough")["value"])

    def test_enforces_minimax_i2va_binding_for_image_passthrough(self):
        result = {
            "message": "Prompt generated.",
            "actions": [
                {"type": "set_widget_value", "node_id": 105, "widget": "prompt", "value": "integrated_multimodal_description: [Shot 1] action"},
                {"type": "set_widget_value", "node_id": 105, "widget": "passthrough", "value": True},
                {"type": "queue_workflow"},
            ],
            "choices": [],
            "thinking": "",
        }
        graph = {"nodes": [{"id": 105, "title": "Image to Video (MiniMax H3)", "widgets": [
            {"name": "prompt", "value": ""},
            {"name": "preset_prompt", "value": "🎬 MiniMax H3 NSFW (5s)"},
            {"name": "passthrough", "value": False},
        ]}]}
        enforced = enforce_image_reference_bindings(result, graph, True)
        self.assertTrue(enforced["actions"][0]["value"].startswith(MINIMAX_I2VA_BINDING))
        self.assertIn(MINIMAX_I2VA_BINDING, enforced["message"])

    def test_skips_minimax_binding_when_passthrough_set_false(self):
        """When the action set flips passthrough to false, do not prepend the I2VA binding
        even if the workflow snapshot still has passthrough=true."""
        result = {
            "message": "Prompt generated.",
            "actions": [
                {"type": "set_widget_value", "node_id": 105, "widget": "prompt", "value": "the woman touches her nipple"},
                {"type": "set_widget_value", "node_id": 105, "widget": "passthrough", "value": False},
                {"type": "queue_workflow"},
            ],
            "choices": [],
            "thinking": "",
        }
        graph = {"nodes": [{"id": 105, "title": "Image to Video (MiniMax H3)", "widgets": [
            {"name": "prompt", "value": ""},
            {"name": "preset_prompt", "value": "🎬 MiniMax H3 NSFW (5s)"},
            {"name": "passthrough", "value": True},
        ]}]}
        enforced = enforce_image_reference_bindings(result, graph, True)
        self.assertNotIn(MINIMAX_I2VA_BINDING, enforced["actions"][0]["value"])
        self.assertNotIn(MINIMAX_I2VA_BINDING, enforced["message"])

    def test_validates_images(self):
        import base64
        valid = base64.b64encode(b"fake-image-data").decode("ascii")
        self.assertEqual(len(validate_images([valid, "not-valid", 123])), 1)
        self.assertEqual(len(validate_images([valid, valid, valid, valid])), 3)

    def test_video_frames_use_higher_limit(self):
        import base64
        frame = base64.b64encode(b"frame").decode("ascii")
        self.assertEqual(len(validate_images([frame] * 5, 4)), 4)
        self.assertEqual(len(validate_images([frame] * 5)), 3)

    def test_prompt_mentions_video_frames(self):
        prompt = build_prompt([{"role": "user", "content": "make it faster"}], {"nodes": []}, has_video=True)
        self.assertIn("VIDEO INPUT", prompt)
        prompt_no_video = build_prompt([{"role": "user", "content": "hi"}], {"nodes": []})
        self.assertNotIn("VIDEO INPUT", prompt_no_video)

    def _jpeg_b64(self):
        import base64
        import io
        from PIL import Image
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), (128, 64, 32)).save(buffer, "JPEG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def test_hf_chat_forwards_video_frames(self):
        class FakeQuantization:
            Q8 = types.SimpleNamespace(value="q8")

        class FakeBase:
            def load_model(self, *args):
                pass

            def generate(self, *args, **kwargs):
                self.kwargs = kwargs
                self.frame_count = args[3]
                return '{"message":"ok","actions":[]}'

        previous = sys.modules.get("AILab_QwenVL")
        sys.modules["AILab_QwenVL"] = types.SimpleNamespace(
            HF_ALL_MODELS={"test-model": {}},
            Quantization=FakeQuantization,
            QwenVLBase=FakeBase,
        )
        try:
            runtime = ChatRuntime()
            frames = [self._jpeg_b64() for _ in range(4)]
            result = runtime.chat("hf", "test-model", [{"role": "user", "content": "make it faster"}], {"nodes": []}, {}, video=frames)
            self.assertEqual(result["message"], "ok")
            instance = runtime._instances["hf"]
            self.assertEqual(len(instance.kwargs["video"]), 4)
            self.assertEqual(instance.frame_count, 4)
        finally:
            if previous is None:
                sys.modules.pop("AILab_QwenVL", None)
            else:
                sys.modules["AILab_QwenVL"] = previous

    def test_gguf_chat_reencodes_video_frames_as_images(self):
        class FakeBase:
            def _load_model(self, *args):
                pass

            def _invoke(self, system, prompt, images_b64, *args, **kwargs):
                self.images_b64 = images_b64
                return '{"message":"ok","actions":[]}'

        previous = sys.modules.get("AILab_QwenVL_GGUF")
        sys.modules["AILab_QwenVL_GGUF"] = types.SimpleNamespace(
            GGUF_VL_CATALOG={"models": {"test-gguf": {}}},
            QwenVLGGUFBase=FakeBase,
        )
        try:
            runtime = ChatRuntime()
            frames = [self._jpeg_b64() for _ in range(4)]
            result = runtime.chat("gguf", "test-gguf", [{"role": "user", "content": "make it faster"}], {"nodes": []}, {}, video=frames)
            self.assertEqual(result["message"], "ok")
            instance = runtime._instances["gguf"]
            self.assertEqual(len(instance.images_b64), 4)
            # Frames must round-trip as valid base64 strings (not raw bytes)
            import base64
            for item in instance.images_b64:
                self.assertIsInstance(item, str)
                base64.b64decode(item, validate=True)
        finally:
            if previous is None:
                sys.modules.pop("AILab_QwenVL_GGUF", None)
            else:
                sys.modules["AILab_QwenVL_GGUF"] = previous

    def test_lists_nested_output_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "root.png").write_bytes(b"png")
            (root / "PMP" / "2026-09-16").mkdir(parents=True)
            (root / "PMP" / "2026-09-16" / "nested.webp").write_bytes(b"webp")
            (root / "ignored.mp4").write_bytes(b"video")
            (root / "ignored.txt").write_text("text")
            assets = list_output_images(root)
        self.assertEqual(set(assets), {"root.png [output]", "PMP/2026-09-16/nested.webp [output]", "ignored.mp4 [output]"})

    def test_hf_runtime_reuses_and_unloads_model(self):
        class FakeQuantization:
            Q8 = types.SimpleNamespace(value="q8")

        class FakeBase:
            def __init__(self):
                self.loaded = 0
                self.cleared = False

            def load_model(self, *args):
                self.loaded += 1

            def generate(self, *args, **kwargs):
                return '{"message":"ok","actions":[]}'

            def clear(self):
                self.cleared = True

        previous = sys.modules.get("AILab_QwenVL")
        sys.modules["AILab_QwenVL"] = types.SimpleNamespace(
            HF_ALL_MODELS={"test-model": {}},
            Quantization=FakeQuantization,
            QwenVLBase=FakeBase,
        )
        try:
            runtime = ChatRuntime()
            result = runtime.chat("hf", "test-model", [{"role": "user", "content": "hello"}], {"nodes": []}, {})
            self.assertEqual(result["message"], "ok")
            instance = runtime._instances["hf"]
            self.assertEqual(instance.loaded, 1)
            self.assertEqual(runtime.unload("hf"), ["hf"])
            self.assertTrue(instance.cleared)
        finally:
            if previous is None:
                sys.modules.pop("AILab_QwenVL", None)
            else:
                sys.modules["AILab_QwenVL"] = previous


if __name__ == "__main__":
    unittest.main()
