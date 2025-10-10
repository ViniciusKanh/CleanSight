# src/main.py
import os
import sys

# Garante que "src" esteja no PYTHONPATH (não altere)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS
from src.routes.preprocessing_enhanced import preprocessing_bp

def create_app():
    app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'cleansight-secret')
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

    # CORS: somente os domínios permitidos
    allowed_origins = os.environ.get(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:5173,https://viniciuskanh.github.io"
    ).split(",")

    CORS(app, resources={r"/api/*": {"origins": allowed_origins}}, supports_credentials=False)

    # Blueprints
    app.register_blueprint(preprocessing_bp, url_prefix='/api')

    # Healthcheck
    @app.get("/health")
    def health():
        return jsonify({"status": "ok"}), 200

    # Raiz/estáticos (opcional para desenvolvimento local)
    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve(path):
        static_folder_path = app.static_folder
        if static_folder_path is None:
            return "Static folder not configured", 404

        if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
            return send_from_directory(static_folder_path, path)
        else:
            index_path = os.path.join(static_folder_path, 'index.html')
            if os.path.exists(index_path):
                return send_from_directory(static_folder_path, 'index.html')
            else:
                return "index.html not found", 404

    # Cabeçalhos de segurança básicos
    @app.after_request
    def set_headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["X-XSS-Protection"] = "1; mode=block"
        return resp

    return app


app = create_app()

if __name__ == '__main__':
    # Para rodar localmente
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
