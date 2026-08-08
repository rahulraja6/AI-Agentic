import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from banking_agent import build_response, load_config

app = Flask(__name__)

# Pre-load config and operations metadata
CONFIG = load_config()

OPERATIONS = [
    {
        "id": "BALANCE_INQUIRY",
        "title": "Account Balance Inquiry",
        "icon": "bi-wallet2",
        "badge": "Auth Required",
        "description": "Check current available balance and account totals.",
        "examples": [
            "What is my account balance?",
            "Show me my available funds",
            "Check my account total"
        ]
    },
    {
        "id": "FUND_TRANSFER",
        "title": "Fund Transfer",
        "icon": "bi-arrow-right-left",
        "badge": "Auth Required",
        "description": "Transfer money or move funds between accounts.",
        "examples": [
            "I need to transfer money to another account",
            "Please transfer funds to my savings account",
            "Wire money to my checking account"
        ]
    },
    {
        "id": "CARD_BLOCK",
        "title": "Card Management & Blocking",
        "icon": "bi-credit-card-2-front-fill",
        "badge": "Auth Required",
        "description": "Report lost or stolen debit/credit cards and freeze access.",
        "examples": [
            "My card was stolen and I need to block it",
            "I need to report my lost card immediately",
            "Freeze my debit card"
        ]
    },
    {
        "id": "PIN_RESET",
        "title": "PIN Reset",
        "icon": "bi-key-fill",
        "badge": "Auth Required",
        "description": "Reset or change forgotten security PINs.",
        "examples": [
            "I forgot my PIN and need a reset",
            "I cannot remember my PIN",
            "Change my account PIN"
        ]
    },
    {
        "id": "LOAN_QUERY",
        "title": "Loan & Mortgage Query",
        "icon": "bi-cash-coin",
        "badge": "Info / Auth",
        "description": "Inquire about loan status, mortgages, and interest rates.",
        "examples": [
            "What is the status of my loan?",
            "How much do I still owe on my loan?",
            "Tell me about current mortgage rates"
        ]
    }
]

@app.route("/")
def index():
    return render_template("index.html", operations=OPERATIONS)

@app.route("/api/operations", methods=["GET"])
def get_operations():
    return jsonify({"operations": OPERATIONS})

@app.route("/api/process", methods=["POST"])
def process_request():
    data = request.get_json() or {}
    user_input = data.get("request", "").strip()
    if not user_input:
        return jsonify({"error": "Request text cannot be empty."}), 400
    
    result = build_response(user_input)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
