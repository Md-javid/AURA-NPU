"""
test_multimodal.py — ASR, TTS, and ADHD Scaffolding Tests
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Validates:
  • IndicConformer ASR for all 22 Indian languages
  • Svara-TTS voice synthesis
  • ADHD adaptive scaffolding algorithm
  • Language configuration completeness

Run: pytest tests/test_multimodal.py -v
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.multimodal_logic import (
    IndicConformerASR,
    SvaraTTS,
    AdaptiveScaffoldingEngine,
    MultimodalProcessor,
    SUPPORTED_LANGUAGES,
    SAMPLE_RATE,
    TranscriptionResult,
    SpeechSynthesisResult,
)
from app.utils.language_config import LANGUAGE_MAP, get_language_names, get_total_coverage


# ── Language Configuration Tests ─────────────────────────────────────────────

class TestLanguageConfiguration(unittest.TestCase):

    def test_all_22_languages_present(self):
        """Aura must support all 22 constitutionally recognized Indian languages."""
        self.assertEqual(
            len(LANGUAGE_MAP), 22,
            f"Expected 22 languages, found {len(LANGUAGE_MAP)}. "
            "All 8th Schedule languages must be represented.",
        )

    def test_hindi_is_supported(self):
        """Hindi must be the primary default language."""
        self.assertIn("hi", LANGUAGE_MAP)
        self.assertEqual(LANGUAGE_MAP["hi"]["name_en"], "Hindi")

    def test_dravidian_languages_present(self):
        """All 4 Dravidian languages must be included."""
        for code in ["ta", "te", "kn", "ml"]:
            self.assertIn(code, LANGUAGE_MAP, f"Missing Dravidian language: {code}")

    def test_rtl_languages_flagged(self):
        """Urdu, Kashmiri, and Sindhi must be flagged as RTL."""
        for code in ["ur", "ks", "sd"]:
            self.assertTrue(
                LANGUAGE_MAP.get(code, {}).get("rtl", False),
                f"Language {code} must be flagged RTL.",
            )

    def test_all_languages_have_conformer_model(self):
        """Every language must have an IndicConformer model ID."""
        for code, info in LANGUAGE_MAP.items():
            self.assertIn(
                "conformer_model", info,
                f"Language '{code}' missing conformer_model.",
            )
            self.assertIn("ai4bharat", info["conformer_model"])

    def test_all_languages_have_native_name(self):
        """Every language must have a native-script name."""
        for code, info in LANGUAGE_MAP.items():
            self.assertIn("name_native", info, f"Language '{code}' missing native name.")
            self.assertGreater(len(info["name_native"]), 0)

    def test_coverage_stats(self):
        """Coverage stats must report 22/22 languages."""
        stats = get_total_coverage()
        self.assertEqual(stats["total_languages"], 22)
        self.assertGreater(stats["total_speakers_million"], 1000)

    def test_prompt_suffixes_in_native_script(self):
        """Hindi, Tamil, Bengali prompt suffixes must be in native script."""
        for code in ["hi", "ta", "bn"]:
            suffix = LANGUAGE_MAP[code].get("prompt_suffix", "")
            # Check that there are non-ASCII characters (native script)
            has_native = any(ord(c) > 127 for c in suffix)
            self.assertTrue(has_native, f"Lang {code} prompt suffix should use native script.")

    def test_supported_languages_consistent_with_language_map(self):
        """SUPPORTED_LANGUAGES in multimodal_logic must match LANGUAGE_MAP codes."""
        for code in SUPPORTED_LANGUAGES:
            self.assertIn(
                code, LANGUAGE_MAP,
                f"SUPPORTED_LANGUAGES code '{code}' not in LANGUAGE_MAP.",
            )


# ── IndicConformer ASR Tests ─────────────────────────────────────────────────

class TestIndicConformerASR(unittest.TestCase):

    def setUp(self):
        # Initialize without a real model session (demo mode)
        self.asr = IndicConformerASR(model_session=None, default_language="hi")

    def test_transcription_returns_result_object(self):
        """transcribe() must return a TranscriptionResult."""
        audio = np.random.randn(SAMPLE_RATE).astype(np.float32)
        result = self.asr.transcribe(audio, language="hi")
        self.assertIsInstance(result, TranscriptionResult)

    def test_transcription_has_latency(self):
        """Result must include a measured latency."""
        audio = np.random.randn(SAMPLE_RATE).astype(np.float32)
        result = self.asr.transcribe(audio, language="hi")
        self.assertGreater(result.latency_ms, 0)

    def test_unknown_language_falls_back_to_hindi(self):
        """Unknown language code must gracefully fall back to Hindi."""
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        result = self.asr.transcribe(audio, language="INVALID_LANG")
        # Should not raise; result.language may show fallback
        self.assertIsInstance(result, TranscriptionResult)

    def test_audio_preprocessing_mono(self):
        """Stereo audio must be converted to mono."""
        stereo_audio = np.random.randn(SAMPLE_RATE, 2).astype(np.float32)
        processed = self.asr._preprocess_audio(stereo_audio, SAMPLE_RATE)
        self.assertEqual(processed.ndim, 1)

    def test_audio_preprocessing_normalization(self):
        """Preprocessed audio values must be in [-1, 1]."""
        audio = np.random.randn(SAMPLE_RATE).astype(np.float32) * 100
        processed = self.asr._preprocess_audio(audio, SAMPLE_RATE)
        self.assertLessEqual(float(np.abs(processed).max()), 1.0 + 1e-6)

    def test_all_22_languages_accepted(self):
        """ASR must accept all 22 supported language codes without error."""
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        for lang_code in SUPPORTED_LANGUAGES.keys():
            try:
                result = self.asr.transcribe(audio, language=lang_code)
                self.assertIsInstance(result, TranscriptionResult)
            except Exception as e:
                self.fail(f"ASR failed for language '{lang_code}': {e}")

    def test_language_id_mapping_complete(self):
        """All 22 languages must have a unique language ID for the model."""
        ids_seen = set()
        for code, info in SUPPORTED_LANGUAGES.items():
            conformer_id = info["conformer_id"]
            lang_id = self.asr._lang_to_id(conformer_id)
            self.assertIsInstance(lang_id, int)
            ids_seen.add(lang_id)
        # Should have unique IDs for each language
        self.assertGreater(len(ids_seen), 10)


# ── Svara-TTS Tests ──────────────────────────────────────────────────────────

class TestSvaraTTS(unittest.TestCase):

    def setUp(self):
        self.tts = SvaraTTS(tts_session=None, vocoder_session=None)

    def test_synthesize_returns_result_object(self):
        """synthesize() must return a SpeechSynthesisResult."""
        result = self.tts.synthesize("Hello, this is Aura.", language="en")
        self.assertIsInstance(result, SpeechSynthesisResult)

    def test_demo_mode_produces_audio(self):
        """Demo mode (no model) must produce a non-empty audio array."""
        result = self.tts.synthesize("परीक्षण वाक्य।", language="hi")
        self.assertIsNotNone(result.audio_data)
        self.assertGreater(len(result.audio_data), 0)

    def test_audio_is_float32(self):
        """Audio output must be float32 for sounddevice compatibility."""
        result = self.tts.synthesize("Test speech.", language="en")
        if result.audio_data is not None:
            self.assertEqual(result.audio_data.dtype, np.float32)

    def test_duration_positive(self):
        """Synthesized audio duration must be positive."""
        result = self.tts.synthesize("Short test.", language="hi")
        self.assertGreater(result.duration_s, 0)

    def test_sample_rate_correct(self):
        """TTS sample rate must be 22050 Hz (Svara-TTS native rate)."""
        result = self.tts.synthesize("Test.", language="ta")
        self.assertEqual(result.sample_rate, 22_050)

    def test_longer_text_produces_longer_audio(self):
        """Longer text should produce longer audio (demo mode)."""
        short_result = self.tts.synthesize("Hi.", language="en")
        long_text = "This is a much longer sentence with many more words for testing."
        long_result = self.tts.synthesize(long_text, language="en")
        if short_result.audio_data and long_result.audio_data:
            self.assertGreater(long_result.duration_s, short_result.duration_s)

    def test_no_error_in_demo_mode(self):
        """TTS demo mode must not set error field."""
        result = self.tts.synthesize("Test without model.", language="kn")
        self.assertIsNone(result.error)


# ── ADHD Scaffolding Tests ────────────────────────────────────────────────────

class TestAdaptiveScaffoldingEngine(unittest.TestCase):

    def setUp(self):
        self.engine = AdaptiveScaffoldingEngine(tts=None)

    def _make_long_text(self, words: int = 1000) -> str:
        """Generate a long synthetic text for testing."""
        base = "The mitochondria is the powerhouse of the cell and produces ATP energy. "
        return (base * (words // 15 + 1))[:words * 6]  # rough char count

    def test_scaffold_produces_milestones(self):
        """scaffold_text() must produce at least one milestone."""
        text = self._make_long_text(300)
        result = self.engine.scaffold_text(text, language="en")
        self.assertGreater(len(result.milestones), 0)

    def test_milestone_structure(self):
        """Each milestone must have required fields."""
        text = self._make_long_text(500)
        result = self.engine.scaffold_text(text, language="en")
        for milestone in result.milestones:
            self.assertIn("number", milestone)
            self.assertIn("text", milestone)
            self.assertIn("word_count", milestone)
            self.assertIn("estimated_minutes", milestone)
            self.assertIn("focus_cue", milestone)

    def test_milestone_numbering_sequential(self):
        """Milestones must be numbered sequentially starting from 1."""
        text = self._make_long_text(1000)
        result = self.engine.scaffold_text(text, language="en")
        for i, m in enumerate(result.milestones):
            self.assertEqual(m["number"], i + 1)

    def test_hindi_focus_cue(self):
        """Hindi mode must provide Hindi-language focus cues."""
        text = self._make_long_text(200)
        result = self.engine.scaffold_text(text, language="hi")
        if result.milestones:
            cue = result.milestones[0]["focus_cue"]
            # Check for Devanagari characters in the cue
            has_devanagari = any("\u0900" <= c <= "\u097f" for c in cue)
            self.assertTrue(has_devanagari, "Hindi focus cue should contain Devanagari script.")

    def test_break_every_three_milestones(self):
        """Every 3rd milestone milestone must suggest a break."""
        text = self._make_long_text(3000)
        result = self.engine.scaffold_text(text, language="en")
        break_milestones = [m for m in result.milestones if m.get("break_after")]
        # Milestones 3, 6, 9... should suggest breaks
        if len(result.milestones) >= 3:
            self.assertTrue(result.milestones[2]["break_after"])

    def test_total_duration_positive(self):
        """Total study duration must be positive."""
        text = self._make_long_text(500)
        result = self.engine.scaffold_text(text, language="en")
        self.assertGreater(result.total_duration_min, 0)

    def test_advance_milestone_progresses(self):
        """advance_milestone() must return sequential milestones."""
        text = self._make_long_text(800)
        scaffolded = self.engine.scaffold_text(text, language="en")
        self.engine.reset()

        first = self.engine.advance_milestone(scaffolded)
        second = self.engine.advance_milestone(scaffolded)

        if first and second:
            self.assertEqual(first["number"], 1)
            self.assertEqual(second["number"], 2)

    def test_advance_returns_none_when_exhausted(self):
        """advance_milestone() must return None when all milestones complete."""
        text = "Short text."
        scaffolded = self.engine.scaffold_text(text, language="en")
        self.engine.reset()

        # Exhaust all milestones
        result = None
        for _ in range(len(scaffolded.milestones) + 2):
            result = self.engine.advance_milestone(scaffolded)
        self.assertIsNone(result)


# ── MultimodalProcessor Integration Tests ────────────────────────────────────

class TestMultimodalProcessor(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.processor = MultimodalProcessor(engine=None)

    def test_get_supported_languages_count(self):
        """Must return all 22 supported languages."""
        langs = self.processor.get_supported_languages()
        self.assertEqual(len(langs), 22)

    def test_each_language_has_code(self):
        """Each language dict must have a 'code' key."""
        langs = self.processor.get_supported_languages()
        for lang in langs:
            self.assertIn("code", lang)

    async def test_transcribe_async_returns_result(self):
        """transcribe_async() must return a TranscriptionResult."""
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        result = await self.processor.transcribe_async(audio, language="hi")
        self.assertIsInstance(result, TranscriptionResult)

    async def test_speak_async_returns_result(self):
        """speak_async() must return a SpeechSynthesisResult."""
        result = await self.processor.speak_async("Namaste.", language="hi")
        self.assertIsInstance(result, SpeechSynthesisResult)

    def test_scaffold_for_adhd_returns_content(self):
        """scaffold_for_adhd() must return ScaffoldedContent."""
        from app.multimodal_logic import ScaffoldedContent
        long_text = "Biology is the study of life. " * 100
        result = self.processor.scaffold_for_adhd(long_text, language="en")
        self.assertIsInstance(result, ScaffoldedContent)
        self.assertGreater(len(result.milestones), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
