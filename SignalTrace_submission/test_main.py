import csv
import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import main


ROOT = Path(__file__).resolve().parent


def utc(time_text: str) -> datetime:
    return datetime.strptime(time_text, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )


def payment(
    payment_id: str,
    time_text: str,
    status: str,
    amount: int = 1000,
    provider_ref: str = "",
    order_id: str = "ORD-100",
) -> main.Payment:
    return main.Payment(
        payment_id, order_id, utc(time_text), status, amount, provider_ref
    )


class SignalTraceTests(unittest.TestCase):
    def test_dotenv_loads_safely_without_overriding_exported_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "GEMINI_API_KEY=file-key\n"
                "export GROQ_API_KEY='groq-key'\n"
                "GROQ_MODEL=openai/gpt-oss-20b\n",
                encoding="utf-8",
            )
            with mock.patch.dict(
                main.os.environ, {"GEMINI_API_KEY": "shell-key"}, clear=True
            ):
                self.assertTrue(main.load_env_file(path))
                self.assertEqual(main.os.environ["GEMINI_API_KEY"], "shell-key")
                self.assertEqual(main.os.environ["GROQ_API_KEY"], "groq-key")
                self.assertEqual(main.os.environ["GROQ_MODEL"], "openai/gpt-oss-20b")

    def test_auto_provider_prefers_verified_groq_when_both_keys_exist(self) -> None:
        with mock.patch.dict(
            main.os.environ,
            {"GEMINI_API_KEY": "gemini-key", "GROQ_API_KEY": "groq-key"},
            clear=True,
        ):
            extractor = main.LiveLLMExtractor("auto", None, 1)
        self.assertEqual(extractor.provider, "groq")

    def test_timestamp_parser_requires_exact_utc_and_real_datetime(self) -> None:
        self.assertIsNotNone(main.parse_timestamp("2026-10-25T10:00:00Z"))
        for invalid in (
            "2026-10-25T10:00:00+00:00",
            "2026-10-25 10:00:00Z",
            "2026-02-30T10:00:00Z",
            "2026-10-25T10:00:00.000Z",
        ):
            with self.subTest(timestamp=invalid):
                self.assertIsNone(main.parse_timestamp(invalid))

    def test_smoke_result_matches_supplied_expected_result(self) -> None:
        with mock.patch.object(
            main.request,
            "urlopen",
            side_effect=AssertionError("offline mode attempted a network request"),
        ):
            result = main.build_result(
                ROOT / "data/smoke/payments.csv",
                ROOT / "data/smoke/shipments.jsonl",
                ROOT / "data/smoke/complaints.jsonl",
                ROOT / "data/smoke/extractions.json",
            )
        expected = json.loads((ROOT / "data/smoke/expected.json").read_text())
        self.assertEqual(result, expected)

    def test_live_extraction_failures_are_isolated_with_safe_fallbacks(self) -> None:
        class FailingExtractor:
            def extract(self, complaint: main.Complaint):
                raise main.LLMError("test provider failure")

        with mock.patch("sys.stderr"):
            result = main.build_result(
                ROOT / "data/sample/payments.csv",
                ROOT / "data/sample/shipments.jsonl",
                ROOT / "data/sample/complaints.jsonl",
                None,
                FailingExtractor(),
            )

        self.assertEqual(result["stats"]["llm_errors"], 15)
        self.assertTrue(
            all(
                case["order_id"] is None
                and case["issue"] == "OTHER"
                and case["verdict"] == "NOT_APPLICABLE"
                and case["evidence_ids"] == []
                for case in result["cases"]
            )
        )

    def test_invalid_record_does_not_reserve_id_and_blank_rows_do_not_count(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payments.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(main.PAYMENT_FIELDS)
                writer.writerow(["P1", "ORD-100", "bad", "SUCCESS", "10", "ref"])
                writer.writerow([])
                writer.writerow(
                    ["P1", "ORD-100", "2026-10-25T10:00:00Z", "SUCCESS", "10", "a"]
                )
                writer.writerow(
                    ["P1", "ORD-100", "2026-10-25T10:01:00Z", "SUCCESS", "10", "b"]
                )
            records, stats = main.load_payments(path)

        self.assertEqual([record.payment_id for record in records], ["P1"])
        self.assertEqual(
            main.asdict(stats),
            {
                "total": 3,
                "accepted": 1,
                "invalid": 1,
                "duplicates": 1,
            },
        )

    def test_payment_loader_sorts_by_timestamp_then_payment_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payments.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(main.PAYMENT_FIELDS)
                writer.writerow(
                    ["P3", "ORD-100", "2026-10-25T10:01:00Z", "FAILED", "10", ""]
                )
                writer.writerow(
                    ["P2", "ORD-100", "2026-10-25T10:00:00Z", "FAILED", "10", ""]
                )
                writer.writerow(
                    ["P1", "ORD-100", "2026-10-25T10:00:00Z", "FAILED", "10", ""]
                )
            records, _ = main.load_payments(path)

        self.assertEqual([record.payment_id for record in records], ["P1", "P2", "P3"])

    def test_jsonl_sources_count_invalid_duplicates_and_blank_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shipments_path = root / "shipments.jsonl"
            shipments_path.write_text(
                "\n".join(
                    [
                        "",
                        json.dumps(
                            {
                                "event_id": "E1",
                                "order_id": "ORD-100",
                                "timestamp": "2026-10-25T10:00:00Z",
                                "event_type": "SHIPMENT_CREATED",
                            }
                        ),
                        json.dumps(
                            {
                                "event_id": "E1",
                                "order_id": "ORD-100",
                                "timestamp": "2026-10-25T10:01:00Z",
                                "event_type": "SUPPORT_NOTE",
                            }
                        ),
                        "not json",
                        json.dumps(
                            {
                                "event_id": "E2",
                                "order_id": "ORD-100",
                                "timestamp": "bad",
                                "event_type": "SHIPMENT_CREATED",
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            complaints_path = root / "complaints.jsonl"
            complaints_path.write_text(
                "\n".join(
                    [
                        json.dumps({"ticket_id": "T1", "message": "Hello"}),
                        json.dumps({"ticket_id": "T1", "message": "Duplicate"}),
                        json.dumps({"ticket_id": "T2", "message": ""}),
                        "{broken",
                    ]
                ),
                encoding="utf-8",
            )
            shipments, shipment_stats = main.load_shipments(shipments_path)
            complaints, complaint_stats = main.load_complaints(complaints_path)

        self.assertEqual([shipment.event_id for shipment in shipments], ["E1"])
        self.assertEqual(
            main.asdict(shipment_stats),
            {"total": 4, "accepted": 1, "invalid": 2, "duplicates": 1},
        )
        self.assertEqual([complaint.ticket_id for complaint in complaints], ["T1"])
        self.assertEqual(
            main.asdict(complaint_stats),
            {"total": 4, "accepted": 1, "invalid": 2, "duplicates": 1},
        )

    def test_extraction_requires_one_distinct_explicit_matching_order(self) -> None:
        self.assertEqual(
            main.validate_extraction(
                "Compare ORD-100 with ORD-101",
                {"order_id": "ORD-100", "issue": "DOUBLE_CHARGE"},
            ),
            (None, "DOUBLE_CHARGE"),
        )
        self.assertEqual(
            main.validate_extraction(
                "ORD-100 appears twice: ORD-100",
                {"order_id": "ORD-100", "issue": "MADE_UP"},
            ),
            ("ORD-100", "OTHER"),
        )
        self.assertEqual(
            main.validate_extraction(
                "Check ORD-100", {"order_id": "ORD-999", "issue": "DOUBLE_CHARGE"}
            ),
            (None, "DOUBLE_CHARGE"),
        )

    def test_order_id_pattern_rejects_short_long_and_embedded_ids(self) -> None:
        for message, proposed in (
            ("Check ORD-12", "ORD-12"),
            ("Check ORD-1234567", "ORD-1234567"),
            ("Check XORD-123", "ORD-123"),
        ):
            with self.subTest(message=message):
                self.assertEqual(
                    main.validate_extraction(
                        message, {"order_id": proposed, "issue": "DOUBLE_CHARGE"}
                    ),
                    (None, "DOUBLE_CHARGE"),
                )

    def test_double_charge_uses_earliest_second_then_earliest_first(self) -> None:
        payments = [
            payment("P1", "2026-10-25T10:00:00Z", "SUCCESS", 1000, "same"),
            payment("P2", "2026-10-25T10:01:00Z", "SUCCESS", 1000, "same"),
            payment("P3", "2026-10-25T10:02:00Z", "SUCCESS", 1000, "other"),
            payment("P4", "2026-10-25T10:03:00Z", "SUCCESS", 1000, "third"),
        ]
        self.assertEqual(main.find_double_charge(payments, "ORD-100"), ["P1", "P3"])

    def test_double_charge_requires_same_amount_and_distinct_provider_refs(
        self,
    ) -> None:
        payments = [
            payment("P1", "2026-10-25T10:00:00Z", "SUCCESS", 1000, "same"),
            payment("P2", "2026-10-25T10:01:00Z", "SUCCESS", 1000, "same"),
            payment("P3", "2026-10-25T10:02:00Z", "SUCCESS", 2000, "different"),
        ]
        self.assertEqual(main.find_double_charge(payments, "ORD-100"), [])

    def test_failure_window_is_inclusive_and_ignores_nonfailed_attempts(self) -> None:
        qualifies = [
            payment("P1", "2026-10-25T10:00:00Z", "FAILED"),
            payment("P2", "2026-10-25T10:01:00Z", "SUCCESS", provider_ref="ok"),
            payment("P3", "2026-10-25T10:05:00Z", "FAILED"),
            payment("P4", "2026-10-25T10:10:00Z", "FAILED"),
        ]
        misses = [
            payment("P1", "2026-10-25T10:00:00Z", "FAILED"),
            payment("P2", "2026-10-25T10:05:00Z", "FAILED"),
            payment("P3", "2026-10-25T10:10:01Z", "FAILED"),
        ]
        self.assertEqual(
            main.find_payment_failures(qualifies, "ORD-100"), ["P1", "P3", "P4"]
        )
        self.assertEqual(main.find_payment_failures(misses, "ORD-100"), [])

    def test_equal_timestamp_payment_is_not_earlier_than_shipment(self) -> None:
        payments = [payment("P1", "2026-10-25T10:00:00Z", "SUCCESS", provider_ref="ok")]
        shipments = [
            main.Shipment(
                "E1", "ORD-100", utc("2026-10-25T10:00:00Z"), "SHIPMENT_CREATED"
            )
        ]
        self.assertEqual(
            main.find_shipped_unpaid(payments, shipments, "ORD-100"), ["E1"]
        )

    def test_verdict_precedence_and_known_order_definition(self) -> None:
        shipments = [
            main.Shipment("E1", "ORD-100", utc("2026-10-25T10:00:00Z"), "SUPPORT_NOTE")
        ]
        self.assertEqual(
            main.investigate([], shipments, None, "OTHER"), ("NOT_APPLICABLE", [])
        )
        self.assertEqual(
            main.investigate([], shipments, None, "DOUBLE_CHARGE"),
            ("INSUFFICIENT_DATA", []),
        )
        self.assertEqual(
            main.investigate([], shipments, "ORD-100", "DOUBLE_CHARGE"),
            ("NOT_VERIFIED", []),
        )

    def test_sqlite_persistence_appends_auditable_runs(self) -> None:
        result = main.build_result(
            ROOT / "data/smoke/payments.csv",
            ROOT / "data/smoke/shipments.jsonl",
            ROOT / "data/smoke/complaints.jsonl",
            ROOT / "data/smoke/extractions.json",
        )
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "history" / "signaltrace.db"
            keyword_paths = {
                "payments_path": ROOT / "data/smoke/payments.csv",
                "shipments_path": ROOT / "data/smoke/shipments.jsonl",
                "complaints_path": ROOT / "data/smoke/complaints.jsonl",
                "output_path": ROOT / "smoke_result.json",
            }
            first_id = main.persist_result(
                database,
                result,
                mode="offline",
                provider=None,
                model=None,
                **keyword_paths,
            )
            second_id = main.persist_result(
                database,
                result,
                mode="offline",
                provider=None,
                model=None,
                **keyword_paths,
            )

            with sqlite3.connect(database) as connection:
                run_count = connection.execute("SELECT COUNT(*) FROM runs").fetchone()[
                    0
                ]
                case_count = connection.execute(
                    "SELECT COUNT(*) FROM cases"
                ).fetchone()[0]
                stats_count = connection.execute(
                    "SELECT COUNT(*) FROM source_stats"
                ).fetchone()[0]
                stored_json = connection.execute(
                    "SELECT result_json FROM runs WHERE run_id = ?", (first_id,)
                ).fetchone()[0]

        self.assertEqual((first_id, second_id), (1, 2))
        self.assertEqual((run_count, case_count, stats_count), (2, 8, 6))
        self.assertEqual(json.loads(stored_json), result)


if __name__ == "__main__":
    unittest.main()
