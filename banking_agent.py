import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None

logger = logging.getLogger("banking_agent")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"


class Agent(ABC):
    """Common interface for all agents in the system."""

    @abstractmethod
    def execute(self, request: str) -> dict:
        """Execute the agent for a given request."""
        pass


def normalize(text: str) -> str:
    """Normalize user input so keyword matching is consistent."""
    return text.strip().lower()


def load_config(config_dir: Path | None = None) -> dict:
    """Load banking agent configuration from JSON files."""
    config_dir = config_dir or CONFIG_DIR
    keywords_path = config_dir / "keywords.json"
    responses_path = config_dir / "responses.json"
    auth_path = config_dir / "auth_required_intents.json"
    agent_path = config_dir / "agent.json"
    semantic_examples_path = config_dir / "semantic_examples.json"

    try:
        semantic_examples = {}
        if semantic_examples_path.exists():
            semantic_examples = json.loads(semantic_examples_path.read_text(encoding="utf-8"))

        return {
            "keywords": json.loads(keywords_path.read_text(encoding="utf-8")),
            "responses": json.loads(responses_path.read_text(encoding="utf-8")),
            "auth_required_intents": json.loads(auth_path.read_text(encoding="utf-8")),
            "agent": json.loads(agent_path.read_text(encoding="utf-8")),
            "semantic_examples": semantic_examples,
        }
    except FileNotFoundError as exc:
        logger.exception("Missing configuration file: %s", exc.filename)
        raise


def compile_keywords(keywords: dict[str, list[str]]) -> dict[str, list[tuple[str, re.Pattern]]]:
    """Compile regex patterns once for efficient repeated matching."""
    return {
        intent: [(pattern, re.compile(pattern)) for pattern in patterns]
        for intent, patterns in keywords.items()
    }


class IntentClassifier:
    """Rule-based intent classifier that returns rich candidate metadata."""

    def __init__(self, compiled_keywords: dict[str, list[tuple[str, re.Pattern]]], confidence_config: dict | None = None):
        self.compiled = compiled_keywords
        self.confidence_config = confidence_config or {
            "base": 0.6,
            "match_weight": 0.1,
            "phrase_weight": 0.15,
            "max": 0.95,
        }

    def classify(self, text: str) -> list[dict]:
        normalized = normalize(text)
        if not normalized:
            return [{"intent": "UNKNOWN", "confidence": 0.0, "reason": "No input provided.", "matchedKeywords": []}]

        logger.debug("Classifying input: %s", text)
        candidates: list[dict] = []
        for intent, patterns in self.compiled.items():
            matched_patterns = []
            matched_keywords = []
            for pattern_text, pattern in patterns:
                match = pattern.search(normalized)
                if match:
                    matched_patterns.append((pattern_text, match.group(0)))
                    matched_keywords.append(match.group(0)) 

            if not matched_patterns:
                continue

            match_count = len(matched_patterns)
            matched_text = " ".join({match for _, match in matched_patterns})
            matched_word_count = len(set(re.findall(r"\b\w+\b", matched_text)))
            phrase_strength = min(matched_word_count / 4, 1.0)

            confidence = (
                self.confidence_config.get("base", 0.6)
                + self.confidence_config.get("match_weight", 0.1) * match_count
                + self.confidence_config.get("phrase_weight", 0.15) * phrase_strength
            )
            confidence = min(confidence, self.confidence_config.get("max", 0.95))

            matched_texts = ", ".join({match for _, match in matched_patterns})
            reason = f"Detected {match_count} matching keyword(s): {matched_texts}"

            candidates.append({
                "intent": intent,
                "confidence": confidence,
                "reason": reason,
                "matchedKeywords": matched_keywords,
            })

        if not candidates:
            return [{"intent": "UNKNOWN", "confidence": 0.0, "reason": "No matching intent keywords found.", "matchedKeywords": []}]

        candidates.sort(key=lambda candidate: candidate["confidence"], reverse=True)
        logger.debug("Classification candidates: %s", candidates)
        return candidates


class HybridIntentClassifier:
    """Rule-first hybrid classifier that falls back to semantic matching when the rule result is weak."""

    def __init__(
        self,
        rule_classifier: "IntentClassifier",
        semantic_classifier: "EmbeddingIntentClassifier",
        confidence_config: dict | None = None,
    ):
        self.rule_classifier = rule_classifier
        self.semantic_classifier = semantic_classifier
        self.confidence_config = confidence_config or {
            "hybrid_rule_threshold": 0.8,
        }

    def classify(self, text: str) -> list[dict]:
        normalized = normalize(text)
        if not normalized:
            return [{"intent": "UNKNOWN", "confidence": 0.0, "reason": "No input provided.", "matchedKeywords": []}]

        rule_candidates = self.rule_classifier.classify(text)
        threshold = self.confidence_config.get("hybrid_rule_threshold", 0.8)
        if (
            rule_candidates
            and rule_candidates[0]["intent"] != "UNKNOWN"
            and rule_candidates[0]["confidence"] >= threshold
        ):
            return rule_candidates

        return self.semantic_classifier.classify(text)


class EmbeddingIntentClassifier:
    """Semantic intent matcher using sentence-transformers embeddings."""

    def __init__(
        self,
        intent_examples: dict[str, list[str]],
        confidence_config: dict | None = None,
        model_name: str = "all-mpnet-base-v2",
    ):
        self.intent_examples = intent_examples
        self.model_name = model_name
        self.confidence_config = confidence_config or {
            "base": 0.5,
            "similarity_scale": 0.6,
            "similarity_threshold": 0.25,
            "max": 0.95,
        }

        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is required for EmbeddingIntentClassifier. "
                "Install it with pip install sentence-transformers"
            )

        self.model = SentenceTransformer(self.model_name)
        self.example_embeddings: dict[str, list[tuple[str, np.ndarray]]] = {}
        self.embedding_cache: dict[str, np.ndarray] = {}

        for intent, examples in self.intent_examples.items():
            if not examples:
                self.example_embeddings[intent] = []
                continue

            embeddings = self._encode_texts(examples)
            self.example_embeddings[intent] = list(zip(examples, embeddings))

    def _encode_texts(self, texts: list[str]) -> list[np.ndarray]:
        normalized_texts = [normalize(text) for text in texts]
        missing_texts: list[str] = []
        missing_positions: list[int] = []

        for index, text in enumerate(normalized_texts):
            if not text:
                continue
            if text not in self.embedding_cache:
                missing_texts.append(text)
                missing_positions.append(index)

        if missing_texts:
            encoded = self.model.encode(
                missing_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            if encoded.ndim == 1:
                encoded = np.expand_dims(encoded, axis=0)

            for position, embedding in zip(missing_positions, encoded):
                self.embedding_cache[normalized_texts[position]] = embedding

        return [self.embedding_cache.get(text, np.zeros(0, dtype=float)) for text in normalized_texts]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def classify(self, text: str) -> list[dict]:
        normalized = normalize(text)
        if not normalized:
            return [{"intent": "UNKNOWN", "confidence": 0.0, "reason": "No input provided.", "matchedKeywords": []}]

        query_embedding = self._encode_texts([normalized])[0]
        candidates: list[dict] = []
        similarity_scale = self.confidence_config.get("similarity_scale", 0.6)
        similarity_threshold = self.confidence_config.get("similarity_threshold", 0.25)

        for intent, examples in self.example_embeddings.items():
            if not examples:
                best_score = 0.0
                top_examples: list[str] = []
            else:
                scored_examples = [
                    (example, self._cosine_similarity(query_embedding, embedding))
                    for example, embedding in examples
                ]
                scored_examples.sort(key=lambda item: item[1], reverse=True)
                best_score = scored_examples[0][1]
                top_examples = [example for example, score in scored_examples if score >= similarity_threshold][:2]

            confidence = min(
                self.confidence_config.get("base", 0.5)
                + similarity_scale * max(best_score, 0.0),
                self.confidence_config.get("max", 0.95),
            )

            reason = f"Semantic match to intent '{intent}' with top example(s): {top_examples or ['none']}"
            candidates.append({
                "intent": intent,
                "confidence": confidence,
                "reason": reason,
                "matchedKeywords": top_examples,
            })

        candidates.sort(key=lambda c: c["confidence"], reverse=True)
        return candidates


def build_intent_classifier(config: dict, model_name: str | None = None) -> Any:
    agent_config = config.get("agent", {})
    confidence_config = agent_config.get("confidence", {})
    classifier_type = agent_config.get("classifier", "rule").lower()

    if classifier_type == "semantic":
        semantic_examples = config.get("semantic_examples", {})
        if not semantic_examples:
            raise ValueError(
                "Semantic classifier requires semantic_examples.json with natural utterances."
            )

        return EmbeddingIntentClassifier(
            semantic_examples,
            confidence_config,
            model_name=model_name or agent_config.get("model_name", "all-mpnet-base-v2"),
        )

    if classifier_type == "hybrid":
        semantic_examples = config.get("semantic_examples", {})
        if not semantic_examples:
            raise ValueError(
                "Hybrid classifier requires semantic_examples.json with natural utterances."
            )

        rule_classifier = IntentClassifier(compile_keywords(config["keywords"]), confidence_config)
        semantic_classifier = EmbeddingIntentClassifier(
            semantic_examples,
            confidence_config,
            model_name=model_name or agent_config.get("model_name", "all-mpnet-base-v2"),
        )
        return HybridIntentClassifier(rule_classifier, semantic_classifier, confidence_config)

    return IntentClassifier(compile_keywords(config["keywords"]), confidence_config)


def compare_classifiers(
    texts: list[str],
    classifier_a,
    classifier_b,
    iterations: int = 10,
) -> dict:
    """Measure average classification latency and compare outputs."""
    def measure(classifier):
        start = perf_counter()
        for _ in range(iterations):
            for text in texts:
                classifier.classify(text)
        return (perf_counter() - start) * 1000 / (iterations * len(texts))

    return {
        "avg_time_ms_a": measure(classifier_a),
        "avg_time_ms_b": measure(classifier_b),
        "sample_output_a": classifier_a.classify(texts[0]),
        "sample_output_b": classifier_b.classify(texts[0]),
    }


def build_default_calibration_dataset() -> list[dict[str, str]]:
    """Build a representative evaluation set for threshold tuning and metrics."""
    return [
        {"text": "What is my account balance?", "intent": "BALANCE_INQUIRY"},
        {"text": "Show me my current balance", "intent": "BALANCE_INQUIRY"},
        {"text": "I need to transfer money to another account", "intent": "FUND_TRANSFER"},
        {"text": "Please transfer funds to my savings account", "intent": "FUND_TRANSFER"},
        {"text": "My card was stolen", "intent": "CARD_BLOCK"},
        {"text": "I need to report my lost card immediately", "intent": "CARD_BLOCK"},
        {"text": "I forgot my PIN", "intent": "PIN_RESET"},
        {"text": "I cannot remember my PIN", "intent": "PIN_RESET"},
        {"text": "What is the status of my loan?", "intent": "LOAN_QUERY"},
        {"text": "How much do I still owe on my loan?", "intent": "LOAN_QUERY"},
        {"text": "Hello there!", "intent": "UNKNOWN"},
        {"text": "Can you help me with a bank question?", "intent": "UNKNOWN"},
        {"text": "I need to check my balance", "intent": "BALANCE_INQUIRY"},
        {"text": "Move funds to my savings", "intent": "FUND_TRANSFER"},
        {"text": "Report a lost debit card", "intent": "CARD_BLOCK"},
        {"text": "Reset my PIN", "intent": "PIN_RESET"},
        {"text": "Tell me about my mortgage", "intent": "LOAN_QUERY"},
        {"text": "Can you help me with my account?", "intent": "UNKNOWN"},
    ]


def evaluate_classifier(dataset: list[dict[str, str]], classifier: Any) -> dict:
    """Evaluate a classifier using precision, recall, F1 and accuracy per intent."""
    intents = sorted({entry["intent"] for entry in dataset})
    totals = {intent: {"tp": 0, "fp": 0, "fn": 0} for intent in intents}

    correct = 0
    for entry in dataset:
        predicted = classifier.classify(entry["text"])[0]["intent"]
        actual = entry["intent"]
        if predicted == actual:
            correct += 1
            totals[actual]["tp"] += 1
        else:
            totals[actual]["fn"] += 1
            totals[predicted]["fp"] += 1

    by_intent = {}
    for intent, counts in totals.items():
        precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
        recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        by_intent[intent] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": counts["tp"] + counts["fn"],
        }

    return {
        "accuracy": correct / len(dataset) if dataset else 0.0,
        "by_intent": by_intent,
    }


def run_calibration_sweep(dataset: list[dict[str, str]] | None = None) -> dict:
    """Run a simple threshold sweep for the hybrid classifier and return evaluation metrics."""
    dataset = dataset or build_default_calibration_dataset()
    config = load_config()
    rule_classifier = IntentClassifier(compile_keywords(config["keywords"]), config["agent"].get("confidence", {}))
    semantic_examples = config.get("semantic_examples", {})
    semantic_classifier = EmbeddingIntentClassifier(
        semantic_examples,
        config["agent"].get("confidence", {}),
        model_name=config["agent"].get("model_name", "all-mpnet-base-v2"),
    )

    results = []
    for threshold in [0.6, 0.7, 0.8, 0.85, 0.9]:
        hybrid_classifier = HybridIntentClassifier(
            rule_classifier,
            semantic_classifier,
            {"hybrid_rule_threshold": threshold},
        )
        results.append({
            "threshold": threshold,
            "metrics": evaluate_classifier(dataset, hybrid_classifier),
        })

    return {
        "dataset_size": len(dataset),
        "results": results,
    }


class AuthenticationPolicy:
    """Separates authentication decision logic from classification."""

    def __init__(self, auth_required_intents: list[str] | None = None):
        self.auth_required_intents = set(auth_required_intents or [])

    def requires_authentication(self, intent: str, text: str) -> bool:
        if intent in self.auth_required_intents:
            return True

        if intent == "LOAN_QUERY":
            normalized = normalize(text)
            account_terms = ["my", "account", "statement", "status", "balance"]
            return any(term in normalized for term in account_terms)

        return False


class ResponseGenerator:
    """Builds final responses using classifier and authentication policy."""

    def __init__(
        self,
        classifier: IntentClassifier,
        policy: AuthenticationPolicy,
        responses: dict[str, str] | None = None,
        agent_config: dict | None = None,
        logger_instance: logging.Logger | None = None,
    ):
        self.classifier = classifier
        self.policy = policy
        self.responses = responses or {}
        self.agent_config = agent_config or {}
        self.logger = logger_instance or logger

    def build_response(self, text: str) -> dict:
        start = perf_counter()
        candidates = self.classifier.classify(text)
        primary = candidates[0]
        secondary = candidates[1] if len(candidates) > 1 else None

        primary_intent = primary["intent"]
        primary_confidence = primary["confidence"]
        reason = primary.get("reason", "")
        matched = primary.get("matchedKeywords", [])

        auth_required = self.policy.requires_authentication(primary_intent, text)
        response_text = self.responses.get(primary_intent, self.responses.get("UNKNOWN", "I am sorry, I did not understand your request."))

        elapsed_ms = int((perf_counter() - start) * 1000)
        result = {
            "primaryIntent": primary_intent,
            "secondaryIntent": secondary["intent"] if secondary else None,
            "confidence": round(primary_confidence, 2),
            "requiresAuthentication": auth_required,
            "response": response_text,
            "reason": reason,
            "matchedKeywords": matched,
            "metadata": {
                "agent": self.agent_config.get("name", "IntentAgent"),
                "version": self.agent_config.get("version", "1.0"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "processingTimeMs": elapsed_ms,
            },
        }
        self.logger.debug("Response payload: %s", result)
        return result


class IntentAgent(Agent):
    """Concrete intent-processing agent implementing the common interface."""

    def __init__(
        self,
        classifier: Any | None = None,
        policy: AuthenticationPolicy | None = None,
        response_generator: ResponseGenerator | None = None,
        config: dict | None = None,
        logger_instance: logging.Logger | None = None,
        model_name: str | None = None,
    ):
        self.config = config or load_config()
        self.logger = logger_instance or logger
        self.classifier = classifier or build_intent_classifier(self.config, model_name=model_name)
        self.policy = policy or AuthenticationPolicy(self.config.get("auth_required_intents", []))
        self.response_generator = response_generator or ResponseGenerator(
            self.classifier,
            self.policy,
            self.config.get("responses", {}),
            self.config.get("agent", {}),
            self.logger,
        )

    def execute(self, request: str) -> dict:
        self.logger.info("Handling request: %s", request)
        return self.response_generator.build_response(request)


class MemoryAgent(Agent):
    def execute(self, request: str) -> dict:
        return {"agent": "MemoryAgent", "request": request, "status": "not_implemented"}


class PlannerAgent(Agent):
    def execute(self, request: str) -> dict:
        return {"agent": "PlannerAgent", "request": request, "status": "not_implemented"}


class FraudAgent(Agent):
    def execute(self, request: str) -> dict:
        return {"agent": "FraudAgent", "request": request, "status": "not_implemented"}


class NotificationAgent(Agent):
    def execute(self, request: str) -> dict:
        return {"agent": "NotificationAgent", "request": request, "status": "not_implemented"}


class Orchestrator:
    """Simple orchestrator that depends only on the common Agent interface."""

    def __init__(self, agent: Agent):
        self.agent = agent

    def execute(self, request: str) -> dict:
        return self.agent.execute(request)


def build_response(text: str, agent: Agent | None = None) -> dict:
    """Compatibility wrapper with dependency injection support."""
    if agent is None:
        agent = IntentAgent()
    return agent.execute(text)


def main() -> None:
    import sys

    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        user_input = input("Enter a banking request: ")

    result = build_response(user_input)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
