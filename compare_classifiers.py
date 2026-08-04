import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from banking_agent import (
    load_config,
    compile_keywords,
    build_intent_classifier,
    IntentClassifier,
    compare_classifiers,
)

OUT = Path("compare_report.json")

config = load_config()

# Rule-based classifier
rule = IntentClassifier(compile_keywords(config["keywords"]), config["agent"].get("confidence", {}))

# Semantic classifier (use semantic examples from config)
semantic_config = {**config, "agent": {**config["agent"], "classifier": "semantic"}}
semantic = build_intent_classifier(semantic_config, model_name=config["agent"].get("model_name"))

texts = [
    "My card was stolen and I need to transfer money.",
    "What is my account balance?",
    "I forgot my PIN and need a reset",
    "What is my loan status?",
    "Hello there!",
    "Please transfer funds to my savings account",
    "I need to report my lost card immediately",
]

# Per-text detailed outputs
per_text = []
for t in texts:
    per_text.append({
        "text": t,
        "rule": rule.classify(t),
        "semantic": semantic.classify(t),
    })

# Aggregate timing comparison
compare = compare_classifiers(texts, rule, semantic, iterations=10)

report = {"per_text": per_text, "compare": compare}

OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"Wrote {OUT}")
print(json.dumps(report, indent=2))
