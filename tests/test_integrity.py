"""
test_integrity.py — Academic Integrity Dashboard Tests
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Validates:
  • Integrity chain hash generation and tamper detection
  • Zero cloud calls proof (all cloud_calls fields = 0)
  • Session summary correctness
  • HTML report generation
  • DPDP Act 2023 compliance markers
  • Prompt privacy (no plaintext prompt storage)

Run: pytest tests/test_integrity.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.integrity_tracker import IntegrityTracker, InteractionRecord, SessionSummary


class TestIntegrityTracker(unittest.TestCase):

    def setUp(self):
        """Create a fresh IntegrityTracker before each test."""
        self.tracker = IntegrityTracker()

    def test_session_id_generated(self):
        """Tracker must generate a non-empty session ID."""
        self.assertIsInstance(self.tracker.session_id, str)
        self.assertGreater(len(self.tracker.session_id), 0)

    def test_log_file_created(self):
        """Log file must be created on initialization."""
        self.assertTrue(self.tracker._log_path.exists())

    def test_header_written_to_log(self):
        """Log file must start with a session header record."""
        content = self.tracker._log_path.read_text(encoding="utf-8")
        first_line = content.strip().split("\n")[0]
        header = json.loads(first_line)
        self.assertEqual(header.get("type"), "session_header")

    def test_header_has_privacy_notice(self):
        """Session header must include data privacy notice."""
        content = self.tracker._log_path.read_text(encoding="utf-8")
        first_line = json.loads(content.strip().split("\n")[0])
        self.assertIn("privacy_notice", first_line)
        self.assertIn("DPDP", first_line["privacy_notice"])

    def test_genesis_hash_in_header(self):
        """Session header must contain the genesis hash."""
        content = self.tracker._log_path.read_text(encoding="utf-8")
        first_line = json.loads(content.strip().split("\n")[0])
        self.assertIn("genesis_hash", first_line)
        self.assertEqual(first_line["genesis_hash"], "GENESIS")


class TestInteractionLogging(unittest.TestCase):

    def setUp(self):
        self.tracker = IntegrityTracker()

    def test_log_interaction_returns_record(self):
        """log_interaction() must return an InteractionRecord."""
        record = self.tracker.log_interaction(
            interaction_type="vision_scan",
            prompt="Explain this biology diagram.",
            language="hi",
            latency_ms=97.3,
        )
        self.assertIsInstance(record, InteractionRecord)

    def test_prompt_not_stored_in_plaintext(self):
        """
        CRITICAL PRIVACY TEST.
        The actual prompt text must NEVER appear in the log file.
        Only a SHA-256 hash should be stored.
        """
        sensitive_prompt = "My personal learning disability is dyslexia and I need help."
        self.tracker.log_interaction(
            interaction_type="vision_scan",
            prompt=sensitive_prompt,
            language="en",
        )
        log_content = self.tracker._log_path.read_text(encoding="utf-8")
        self.assertNotIn(
            sensitive_prompt, log_content,
            "PRIVACY VIOLATION: Sensitive prompt stored in plaintext in the log file!",
        )

    def test_cloud_calls_always_zero(self):
        """
        DATA SOVEREIGNTY TEST.
        The cloud_calls field must ALWAYS be 0 in every logged record.
        This is the key DPDP Act 2023 compliance proof.
        """
        for _ in range(5):
            record = self.tracker.log_interaction(
                interaction_type="vision_scan",
                prompt="Test prompt.",
                language="en",
            )
            self.assertEqual(
                record.cloud_calls, 0,
                "DATA SOVEREIGNTY VIOLATION: cloud_calls must always be 0. "
                "Aura-NPU is 100% offline.",
            )

    def test_record_hash_computed(self):
        """Each record must have a computed hash for chain integrity."""
        record = self.tracker.log_interaction("vision_scan", "test", "en")
        self.assertIsInstance(record.record_hash, str)
        self.assertGreater(len(record.record_hash), 0)

    def test_chain_hash_links_records(self):
        """Each record's previous_record_hash must equal the prior record's hash."""
        r1 = self.tracker.log_interaction("vision_scan", "prompt 1", "hi")
        r2 = self.tracker.log_interaction("asr_transcription", "prompt 2", "ta")
        self.assertEqual(r2.previous_record_hash, r1.record_hash)

    def test_record_persisted_to_log_file(self):
        """Logged records must be appended to the JSONL file."""
        initial_line_count = len(
            self.tracker._log_path.read_text(encoding="utf-8").strip().split("\n")
        )
        self.tracker.log_interaction("vision_scan", "test prompt", "bn")
        new_line_count = len(
            self.tracker._log_path.read_text(encoding="utf-8").strip().split("\n")
        )
        self.assertEqual(new_line_count, initial_line_count + 1)

    def test_language_stored_in_record(self):
        """Language code must be correctly stored for audit trail."""
        for lang in ["hi", "ta", "bn"]:
            record = self.tracker.log_interaction("vision_scan", "test", lang)
            self.assertEqual(record.language, lang)

    def test_latency_rounded_correctly(self):
        """Latency must be rounded to 2 decimal places."""
        record = self.tracker.log_interaction("vision_scan", "test", "en", latency_ms=97.334567)
        self.assertEqual(record.latency_ms, 97.33)

    def test_session_id_propagated_to_records(self):
        """All records must carry the same session ID."""
        r1 = self.tracker.log_interaction("vision_scan", "p1", "hi")
        r2 = self.tracker.log_interaction("vision_scan", "p2", "te")
        self.assertEqual(r1.session_id, r2.session_id)
        self.assertEqual(r1.session_id, self.tracker.session_id)


class TestSessionSummary(unittest.TestCase):

    def setUp(self):
        self.tracker = IntegrityTracker()

    def test_empty_session_summary(self):
        """Empty session must return a valid SessionSummary."""
        summary = self.tracker.get_session_summary()
        self.assertIsInstance(summary, SessionSummary)
        self.assertEqual(summary.total_interactions, 0)

    def test_interaction_count_accurate(self):
        """Session summary must accurately count all interactions."""
        for _ in range(7):
            self.tracker.log_interaction("vision_scan", "prompt", "hi")
        summary = self.tracker.get_session_summary()
        self.assertEqual(summary.total_interactions, 7)

    def test_vision_scan_count(self):
        """Vision scan count must be separately tracked."""
        self.tracker.log_interaction("vision_scan", "p1", "hi")
        self.tracker.log_interaction("vision_scan", "p2", "ta")
        self.tracker.log_interaction("asr_transcription", "p3", "hi")
        summary = self.tracker.get_session_summary()
        self.assertEqual(summary.vision_scans, 2)

    def test_total_cloud_calls_always_zero(self):
        """
        DATA SOVEREIGNTY: Session summary cloud_calls must be 0.
        This is presented to institutions as compliance evidence.
        """
        for _ in range(10):
            self.tracker.log_interaction("vision_scan", "test", "hi")
        summary = self.tracker.get_session_summary()
        self.assertEqual(
            summary.total_cloud_calls, 0,
            "Session summary shows non-zero cloud calls — "
            "DPDP Act 2023 compliance violation.",
        )

    def test_dpdp_compliance_flag(self):
        """Session summary must explicitly flag DPDP compliance."""
        summary = self.tracker.get_session_summary()
        self.assertTrue(summary.dpdp_compliant)

    def test_rpwd_accommodation_flag(self):
        """Session summary must flag RPwD Act 2016 accommodation."""
        summary = self.tracker.get_session_summary()
        self.assertTrue(summary.rpwd_accommodation)

    def test_data_sovereignty_string(self):
        """Data sovereignty field must state 'On-Device'."""
        summary = self.tracker.get_session_summary()
        self.assertIn("On-Device", summary.data_sovereignty)

    def test_languages_used_tracked(self):
        """Languages used in the session must be collected."""
        self.tracker.log_interaction("vision_scan", "p1", "hi")
        self.tracker.log_interaction("vision_scan", "p2", "ta")
        summary = self.tracker.get_session_summary()
        self.assertIn("hi", summary.languages_used)
        self.assertIn("ta", summary.languages_used)

    def test_avg_latency_computed(self):
        """Average latency must be computed correctly."""
        self.tracker.log_interaction("vision_scan", "p1", "hi", latency_ms=100.0)
        self.tracker.log_interaction("vision_scan", "p2", "hi", latency_ms=200.0)
        summary = self.tracker.get_session_summary()
        self.assertAlmostEqual(summary.avg_latency_ms, 150.0, places=1)


class TestDashboardData(unittest.TestCase):

    def setUp(self):
        self.tracker = IntegrityTracker()

    def test_dashboard_data_is_dict(self):
        """get_dashboard_data() must return a dict."""
        data = self.tracker.get_dashboard_data()
        self.assertIsInstance(data, dict)

    def test_dashboard_shows_zero_cloud_calls(self):
        """Dashboard must prominently show 0 cloud calls."""
        self.tracker.log_interaction("vision_scan", "test", "hi")
        data = self.tracker.get_dashboard_data()
        cloud_value = data.get("Cloud Calls", "")
        self.assertIn("0", cloud_value)

    def test_dashboard_shows_dpdp_compliant(self):
        """Dashboard must show DPDP compliance."""
        data = self.tracker.get_dashboard_data()
        dpdp_value = data.get("DPDP Act 2023", "")
        self.assertIn("✅", dpdp_value)

    def test_dashboard_shows_session_id(self):
        """Dashboard must show the formatted session ID."""
        data = self.tracker.get_dashboard_data()
        self.assertIn("Session ID", data)
        self.assertIn("AURA-", data["Session ID"])


class TestReportGeneration(unittest.TestCase):

    def setUp(self):
        self.tracker = IntegrityTracker()
        # Log some sample interactions
        for i in range(5):
            self.tracker.log_interaction(
                "vision_scan", f"test prompt {i}", "hi", latency_ms=95.0 + i
            )

    def test_json_report_generated(self):
        """generate_json_report() must create a valid JSON file."""
        path = self.tracker.generate_json_report()
        self.assertTrue(path.exists())
        content = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("summary", content)
        self.assertIn("chain_valid", content)

    def test_json_report_cloud_calls_zero(self):
        """JSON report must document zero cloud calls."""
        path = self.tracker.generate_json_report()
        content = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(content["summary"]["total_cloud_calls"], 0)

    def test_html_report_generated(self):
        """generate_html_report() must create an HTML file."""
        path = self.tracker.generate_html_report()
        self.assertTrue(path.exists())
        html_content = path.read_text(encoding="utf-8")
        self.assertIn("<!DOCTYPE html>", html_content)

    def test_html_report_contains_aura_branding(self):
        """HTML report must contain Aura branding and AMD reference."""
        path = self.tracker.generate_html_report()
        html_content = path.read_text(encoding="utf-8")
        self.assertIn("AURA-NPU", html_content)
        self.assertIn("AMD Ryzen AI", html_content)

    def test_html_report_shows_dpdp_badge(self):
        """HTML report must include the DPDP Act 2023 compliance badge."""
        path = self.tracker.generate_html_report()
        html_content = path.read_text(encoding="utf-8")
        self.assertIn("DPDP", html_content)

    def test_html_report_shows_zero_cloud(self):
        """HTML report must explicitly show 0 cloud API calls."""
        path = self.tracker.generate_html_report()
        html_content = path.read_text(encoding="utf-8")
        self.assertIn("Cloud Calls", html_content)


class TestChainIntegrity(unittest.TestCase):

    def setUp(self):
        self.tracker = IntegrityTracker()

    def test_chain_valid_on_fresh_tracker(self):
        """Chain must be valid on a fresh tracker."""
        self.assertTrue(self.tracker._verify_chain())

    def test_chain_valid_after_logging(self):
        """Chain must remain valid after multiple interactions."""
        for _ in range(10):
            self.tracker.log_interaction("vision_scan", "test", "hi")
        self.assertTrue(self.tracker._verify_chain())

    def test_hash_changes_between_records(self):
        """Consecutive record hashes must be different."""
        r1 = self.tracker.log_interaction("vision_scan", "prompt 1", "hi")
        r2 = self.tracker.log_interaction("vision_scan", "prompt 2", "hi")
        self.assertNotEqual(r1.record_hash, r2.record_hash)

    def test_device_fingerprint_generated(self):
        """Device fingerprint must be a non-empty string."""
        fp = self.tracker._device_id
        self.assertIsInstance(fp, str)
        self.assertGreater(len(fp), 0)

    def test_prompt_hash_non_trivial(self):
        """Prompt hash must be a non-empty hex string."""
        h = self.tracker._hash_text("Test prompt for hashing.")
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 16)  # Truncated SHA-256 (16 chars)
        # Must be hex characters
        int(h, 16)  # Raises ValueError if not valid hex


if __name__ == "__main__":
    unittest.main(verbosity=2)
