# src/main.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from src.routes.preprocessing_enhanced import preprocessing_bp

def create_app():
    app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'cleansight-secret')
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

    allowed = os.environ.get(
        "ALLOWED_ORIGINS",
        "https://viniciuskanh.github.io,http://localhost:3000,http://localhost:5173"
    ).split(",")
    CORS(app, resources={r"/api/*": {"origins": allowed}})

    app.register_blueprint(preprocessing_bp, url_prefix="/api")

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"}), 200

    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve(path):
        sf = app.static_folder
        if sf and path and os.path.exists(os.path.join(sf, path)):
            return send_from_directory(sf, path)
        if sf and os.path.exists(os.path.join(sf, 'index.html')):
            return send_from_directory(sf, 'index.html')
        return "index.html not found", 404

    @app.after_request
    def headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["X-XSS-Protection"] = "1; mode=block"
        return resp

    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
