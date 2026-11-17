from flask import Flask, jsonify, request
from pydantic import ValidationError

from .service import Ask, Assistant


def create_app(assistant=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
    core = assistant if assistant is not None else Assistant()

    @app.get("/healthz")
    def health():
        return jsonify(status="ok", mode=core.settings.mode,
                       products=len(core.retrieval.products),
                       semantic_index_ready=bool(core.retrieval.vector))

    @app.post("/ask")
    def ask():
        try:
            body = Ask.model_validate(request.get_json(silent=True))
        except ValidationError as exc:
            return jsonify(error="invalid_request", details=exc.errors(include_input=False,
                                                                       include_context=False,
                                                                       include_url=False)), 400
        reply = core.ask(body)
        return jsonify(reply.model_dump()), 503 if reply.route == "service" else 200

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="request_too_large"), 413

    return app
