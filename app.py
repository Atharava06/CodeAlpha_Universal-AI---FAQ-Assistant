"""
Flask Application for FAQ Assistant & Universal AI Chatbot.

Provides RESTful API endpoints, hybrid retrieval (TF-IDF Cosine Similarity
for verified local FAQs + Google Gemini AI for all-domain reasoning),
and serves the modern chat web interface.
"""

import logging
import os
from typing import Any, Dict

from flask import Flask, jsonify, render_template, request

try:
    from dotenv import load_dotenv
    load_dotenv()
    if not os.environ.get("GOOGLE_API_KEY"):
        load_dotenv(".env.example")
except ImportError:
    pass

from chatbot.faq_engine import FAQEngine
from chatbot.gemini_service import GeminiService
from chatbot.preprocessing import ensure_nltk_resources

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("FAQAssistant")

# Initialize Flask app
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # Max 2MB payload

# Configuration
FAQ_DATA_PATH = os.environ.get("FAQ_DATA_PATH", "faqs.json")
DEFAULT_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.25"))
MAX_QUERY_LENGTH = int(os.environ.get("MAX_QUERY_LENGTH", "1000"))

# Initialize Services
logger.info("Initializing NLTK resources, FAQ Engine, and Google Gemini Service...")
ensure_nltk_resources()

try:
    faq_engine = FAQEngine(
        faq_file_path=FAQ_DATA_PATH,
        similarity_threshold=DEFAULT_THRESHOLD
    )
    logger.info(f"Loaded {len(faq_engine.faqs)} FAQs with threshold {DEFAULT_THRESHOLD}")
except Exception as e:
    logger.error(f"Failed to load FAQ engine: {e}")
    faq_engine = None

gemini_service = GeminiService()
if gemini_service.is_configured():
    logger.info("Google Gemini API is configured and active via environment.")
else:
    logger.info("Google Gemini API key not detected in environment. Client-provided keys will be accepted.")


# ---------------------------------------------------------
# Web Interface Route
# ---------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """Serve the main chatbot web application interface."""
    return render_template("index.html")


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Health check endpoint returning application status,
    loaded FAQ count, Gemini configuration state, and engine settings.
    """
    is_ready = faq_engine is not None and len(faq_engine.faqs) > 0
    return jsonify({
        "status": "healthy" if is_ready else "degraded",
        "service": "FAQ & Universal AI Assistant API",
        "version": "1.1.0",
        "faqs_loaded": len(faq_engine.faqs) if faq_engine else 0,
        "similarity_threshold": faq_engine.similarity_threshold if faq_engine else DEFAULT_THRESHOLD,
        "gemini_configured": gemini_service.is_configured()
    }), 200 if is_ready else 503


@app.route("/api/verify-key", methods=["POST"])
def verify_key():
    """
    Test and validate a user-supplied Google API key.
    Accepts JSON: { "api_key": "AIza..." }
    """
    if not request.is_json:
        return jsonify({"valid": False, "message": "Expected JSON payload"}), 400

    data = request.get_json(silent=True) or {}
    test_key = data.get("api_key", "").strip()

    if not test_key:
        return jsonify({"valid": False, "message": "API key cannot be empty"}), 400

    is_valid, msg = gemini_service.verify_key(test_key)
    return jsonify({
        "valid": is_valid,
        "message": msg
    }), 200 if is_valid else 400


@app.route("/api/faqs", methods=["GET"])
def get_faqs():
    """
    Return available FAQ categories, suggested prompt questions,
    and a summary list of FAQs for frontend exploration.
    """
    if faq_engine is None:
        return jsonify({
            "error": "FAQ engine is unavailable."
        }), 503

    category_filter = request.args.get("category", None)
    all_categories = faq_engine.get_categories()
    suggested = faq_engine.get_suggested_questions(limit=8)
    summary_list = faq_engine.get_faqs_summary()

    if category_filter:
        summary_list = [f for f in summary_list if f.get("category", "").lower() == category_filter.lower()]

    return jsonify({
        "total": len(faq_engine.faqs),
        "categories": all_categories,
        "suggested_questions": suggested,
        "faqs": summary_list,
        "gemini_active": gemini_service.is_configured()
    }), 200


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Universal Chat endpoint.
    Handles questions across all domains using a hybrid architecture:
    1. Checks local FAQs for institutional matching.
    2. Uses Google Gemini AI for questions across all other domains
       (coding, science, general knowledge, arts, everyday questions).

    Accepts JSON:
    {
        "message": "string",
        "threshold": float (optional),
        "api_key": string (optional, Google Gemini key),
        "mode": "hybrid" | "gemini" | "faq" (optional, default "hybrid")
    }
    """
    if not request.is_json:
        return jsonify({
            "error": "Invalid request content-type. Expected application/json."
        }), 400

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({
            "error": "Malformed JSON payload in request."
        }), 400

    if "message" not in data:
        return jsonify({
            "error": "Missing required field 'message' in request body."
        }), 400

    raw_message = data.get("message")
    if not isinstance(raw_message, str):
        return jsonify({
            "error": "Field 'message' must be a string."
        }), 400

    user_query = raw_message.strip()
    if not user_query:
        return jsonify({
            "error": "Message cannot be empty or contain only whitespace."
        }), 400

    if len(user_query) > MAX_QUERY_LENGTH:
        return jsonify({
            "error": f"Message exceeds maximum allowed length of {MAX_QUERY_LENGTH} characters."
        }), 400

    # Extract options
    custom_threshold = data.get("threshold")
    active_threshold = faq_engine.similarity_threshold if faq_engine else DEFAULT_THRESHOLD
    if custom_threshold is not None:
        try:
            val = float(custom_threshold)
            if 0.0 <= val <= 1.0:
                active_threshold = val
        except (ValueError, TypeError):
            pass

    client_api_key = data.get("api_key")
    mode = (data.get("mode") or "hybrid").lower().strip()
    has_gemini = bool((client_api_key and len(client_api_key.strip()) > 10) or gemini_service.is_configured())

    # Mode 1: Force Gemini Only
    if mode == "gemini":
        if not has_gemini:
            return jsonify({
                "answer": (
                    "Google Gemini mode is selected, but no Google API key is configured. "
                    "Please enter your Google API key in Settings (⚙️) or set the GOOGLE_API_KEY environment variable."
                ),
                "matched_question": None,
                "category": "Settings Required",
                "confidence": 0.0,
                "matched": False,
                "source": "System"
            }), 200

        gemini_res = gemini_service.generate_answer(user_query, api_key_override=client_api_key)
        if gemini_res["success"]:
            return jsonify({
                "answer": gemini_res["answer"],
                "matched_question": None,
                "category": "Universal AI",
                "confidence": 1.0,
                "matched": True,
                "source": "Google Gemini AI"
            }), 200
        else:
            return jsonify({
                "answer": f"Unable to generate response: {gemini_res.get('error')}",
                "matched_question": None,
                "category": "API Error",
                "confidence": 0.0,
                "matched": False,
                "source": "Google Gemini AI"
            }), 200

    # Mode 2 & 3: Hybrid (Default) or FAQ Only
    faq_match = None
    if faq_engine:
        original_threshold = faq_engine.similarity_threshold
        faq_engine.similarity_threshold = active_threshold
        try:
            faq_match = faq_engine.find_best_match(user_query)
        finally:
            faq_engine.similarity_threshold = original_threshold

    # If FAQ match is strong (>= 0.40 or mode is faq-only)
    is_strong_faq = faq_match and faq_match["matched"] and faq_match["confidence"] >= max(active_threshold, 0.40)
    if is_strong_faq or mode == "faq":
        if faq_match and faq_match["matched"]:
            return jsonify({
                "answer": faq_match["answer"],
                "matched_question": faq_match["matched_question"],
                "category": faq_match["category"],
                "confidence": faq_match["confidence"],
                "matched": True,
                "source": "Verified FAQ"
            }), 200
        elif mode == "faq":
            # FAQ only mode fallback
            return jsonify({
                "answer": faq_match["answer"] if faq_match else "No answer found.",
                "matched_question": None,
                "category": None,
                "confidence": 0.0,
                "matched": False,
                "source": "Verified FAQ"
            }), 200

    # If question is outside FAQ or low confidence, escalate to Google Gemini for universal domain answering!
    if has_gemini:
        gemini_res = gemini_service.generate_answer(user_query, api_key_override=client_api_key)
        if gemini_res["success"]:
            return jsonify({
                "answer": gemini_res["answer"],
                "matched_question": None,
                "category": "Universal AI",
                "confidence": 1.0,
                "matched": True,
                "source": "Google Gemini AI"
            }), 200
        else:
            logger.warning(f"Gemini generation fallback error: {gemini_res.get('error')}")

    # If moderate FAQ match exists (between active_threshold and 0.40) and Gemini didn't answer
    if faq_match and faq_match["matched"]:
        return jsonify({
            "answer": faq_match["answer"],
            "matched_question": faq_match["matched_question"],
            "category": faq_match["category"],
            "confidence": faq_match["confidence"],
            "matched": True,
            "source": "Verified FAQ"
        }), 200

    # Universal Fallback with guidance to enable Google API Key
    fallback_message = (
        "I couldn't find a direct match in the university FAQ database. "
        "To enable universal answers across **every domain** (programming, science, literature, history, math, and general knowledge), "
        "please enter your **Google Gemini API Key** in **Settings (⚙️)** or configure `GOOGLE_API_KEY`."
    )
    return jsonify({
        "answer": fallback_message,
        "matched_question": None,
        "category": "Universal Domain",
        "confidence": 0.0,
        "matched": False,
        "source": "Fallback Assistant",
        "needs_google_key": not has_gemini
    }), 200


# ---------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------
@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": "Bad request", "details": str(error)}), 400


@app.errorhandler(404)
def not_found(error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Endpoint not found"}), 404
    return render_template("index.html"), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed on this endpoint"}), 405


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "Request payload too large (max 2MB)"}), 413


@app.errorhandler(500)
def internal_server_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "An internal server error occurred"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "True").lower() in ("true", "1", "yes")
    print(f"Starting FAQ & Universal AI Assistant on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
