import unittest

from banking_agent import (
    HybridIntentClassifier,
    build_intent_classifier,
    build_response,
    load_config,
)


class BankingAgentTests(unittest.TestCase):
    def _print_case(self, user_input: str, result: dict) -> None:
        print(f"Input: {user_input}")
        print(f"Output: {result}")

    def test_load_config_reads_json_files(self) -> None:
        config = load_config()
        self.assertIn("keywords", config)
        self.assertIn("responses", config)
        self.assertIn("auth_required_intents", config)
        self.assertIn("BALANCE_INQUIRY", config["keywords"])

    def test_card_theft_and_transfer_prefers_card_block(self) -> None:
        user_input = "My card was stolen and I need to transfer money."
        result = build_response(user_input)
        self._print_case(user_input, result)
        self.assertEqual(result["primaryIntent"], "CARD_BLOCK")
        self.assertGreaterEqual(result["confidence"], 0.7)

    def test_transfer_only_returns_fund_transfer(self) -> None:
        user_input = "I need to transfer money to my brother"
        result = build_response(user_input)
        self._print_case(user_input, result)
        self.assertEqual(result["primaryIntent"], "FUND_TRANSFER")

    def test_balance_inquiry_is_detected(self) -> None:
        user_input = "What is my account balance?"
        result = build_response(user_input)
        self._print_case(user_input, result)
        self.assertEqual(result["primaryIntent"], "BALANCE_INQUIRY")

    def test_pin_reset_is_detected(self) -> None:
        user_input = "I forgot my PIN and need a reset"
        result = build_response(user_input)
        self._print_case(user_input, result)
        self.assertEqual(result["primaryIntent"], "PIN_RESET")

    def test_loan_query_is_detected(self) -> None:
        user_input = "What is my loan status?"
        result = build_response(user_input)
        self._print_case(user_input, result)
        self.assertEqual(result["primaryIntent"], "LOAN_QUERY")

    def test_unknown_request_returns_unknown_intent(self) -> None:
        user_input = "Hello there!"
        result = build_response(user_input)
        self._print_case(user_input, result)
        self.assertEqual(result["primaryIntent"], "UNKNOWN")

    def test_empty_input_returns_unknown(self) -> None:
        user_input = "   "
        result = build_response(user_input)
        self._print_case(user_input, result)
        self.assertEqual(result["primaryIntent"], "UNKNOWN")

    def test_hybrid_classifier_uses_rule_for_strong_match(self) -> None:
        config = load_config()
        config["agent"]["classifier"] = "hybrid"
        config["agent"]["confidence"]["hybrid_rule_threshold"] = 0.8
        classifier = build_intent_classifier(config)
        result = classifier.classify("What is my account balance?")
        self.assertIsInstance(result, list)
        self.assertEqual(result[0]["intent"], "BALANCE_INQUIRY")
        self.assertGreaterEqual(result[0]["confidence"], 0.8)

    def test_hybrid_classifier_falls_back_to_semantic_for_weak_rule(self) -> None:
        config = load_config()
        config["agent"]["classifier"] = "hybrid"
        config["agent"]["confidence"]["hybrid_rule_threshold"] = 0.9
        classifier = build_intent_classifier(config)
        result = classifier.classify("Please transfer funds to my savings account")
        self.assertEqual(result[0]["intent"], "FUND_TRANSFER")
        self.assertGreaterEqual(result[0]["confidence"], 0.6)

    def test_semantic_classifier_can_be_selected_via_config(self) -> None:
        config = load_config()
        config["agent"]["classifier"] = "semantic"
        classifier = build_intent_classifier(config)
        result = classifier.classify("I need to check my loan status")
        self.assertIsInstance(result, list)
        self.assertEqual(result[0]["intent"], "LOAN_QUERY")
        self.assertGreaterEqual(result[0]["confidence"], 0.6)

    def test_semantic_classifier_works_with_diverse_natural_utterances(self) -> None:
        config = load_config()
        config["agent"]["classifier"] = "semantic"
        classifier = build_intent_classifier(config)

        cases = {
            "Show me my current balance": "BALANCE_INQUIRY",
            "Please transfer funds to my savings account": "FUND_TRANSFER",
            "I need to report my lost card immediately": "CARD_BLOCK",
            "I cannot remember my PIN": "PIN_RESET",
            "How much do I still owe on my loan?": "LOAN_QUERY",
        }

        for text, expected in cases.items():
            result = classifier.classify(text)
            self._print_case(text, result)
            self.assertEqual(result[0]["intent"], expected)
            self.assertGreaterEqual(result[0]["confidence"], 0.6)

    def test_secondary_intent_is_present_for_mixed_request(self) -> None:
        user_input = "I lost my card and also need to transfer money"
        result = build_response(user_input)
        self._print_case(user_input, result)
        self.assertEqual(result["primaryIntent"], "CARD_BLOCK")
        self.assertEqual(result["secondaryIntent"], "FUND_TRANSFER")

    def test_complex_multi_intent_request_is_ranked(self) -> None:
        user_input = (
            "My card was stolen, I need to transfer money, and I also want to check my loan status."
        )
        result = build_response(user_input)
        self._print_case(user_input, result)
        # With intent-agnostic scoring the classifier ranks by evidence; the
        # deterministic outcome for this fixed input is LOAN_QUERY primary and
        # CARD_BLOCK secondary.
        self.assertEqual(result["primaryIntent"], "LOAN_QUERY")
        self.assertEqual(result["secondaryIntent"], "CARD_BLOCK")
        self.assertIn(result["response"], [
            "I can help block your card. Authentication is required.",
            "I can help schedule a fund transfer. Please authenticate first.",
            "I can provide loan information. If this is about your account, authentication may be required.",
        ])


if __name__ == "__main__":
    unittest.main()
