from flask import Flask, jsonify, send_from_directory, request, abort
from flask_cors import CORS
from routes.data import data_bp
from werkzeug.utils import secure_filename
import mimetypes

app = Flask(__name__)
# During local development allow all origins so dev frontends on any localhost port can load images
# In production you should lock this down to specific origins.
# Use the correct resources mapping for flask_cors: pattern -> options dict
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

# Register the blueprint
app.register_blueprint(data_bp)

@app.route("/")
def root():
    return jsonify({"message": "Welcome to Flask!"})

@app.route("/hello")
def hello():
    return jsonify({"message": "Hello from Flask!"})


import os

# Use an absolute path to the output directory next to this file
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


# Route kept for backward-compatibility with frontend code that requests via query param
@app.route('/output-image')
def serve_output_image():
    filename = request.args.get('filename')
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400

    # Prevent directory traversal and unsafe names
    safe_name = secure_filename(filename)
    if safe_name != filename:
        # If sanitization changed the name, reject to avoid surprises
        return jsonify({'error': 'Invalid filename provided'}), 400

    file_path = os.path.join(OUTPUT_DIR, safe_name)
    if not os.path.isfile(file_path):
        return jsonify({'error': 'File not found'}), 404

    # Let Flask and Werkzeug set the appropriate mimetype
    return send_from_directory(OUTPUT_DIR, safe_name)


# Also expose a cleaner direct route that maps to /output/<filename>
@app.route('/output/<path:filename>')
def serve_output_direct(filename: str):
    # secure the filename to avoid traversal
    safe_name = secure_filename(filename)
    if safe_name != filename:
        abort(400)
    file_path = os.path.join(OUTPUT_DIR, safe_name)
    if not os.path.isfile(file_path):
        abort(404)

    # Try to guess mimetype; fallback to octet-stream
    mimetype, _ = mimetypes.guess_type(file_path)
    return send_from_directory(OUTPUT_DIR, safe_name, mimetype=mimetype)

if __name__ == "__main__":
    # Prefer port 5500 for the dev server to avoid common macOS service conflicts.
    # Allow overriding with the PORT env var if desired.
    import sys
    port = int(os.environ.get('PORT') or 5500)
    host = os.environ.get('HOST') or '127.0.0.1'

    try:
        print(f"Starting Flask on {host}:{port}")
        app.run(debug=True, port=port, host=host)
    except OSError as e:
        # If chosen port is unavailable, try a nearby port
        fallback = port + 1
        print(f"Port {port} unavailable ({e}), falling back to {fallback}", file=sys.stderr)
        app.run(debug=True, port=fallback, host=host)
