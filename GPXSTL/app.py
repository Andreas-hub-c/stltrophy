import os
import io
from flask import Flask, request, send_file, send_from_directory, jsonify

# We importeren jouw geoptimaliseerde generator logica (sla je eerdere script op als generator.py)
from generator import generate_terrain_and_route, GeneratorConfig

app = Flask(__name__)

# --- ROUTES VOOR DE HTML PAGINA'S ---
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/preview.html')
def preview():
    return send_from_directory('.', 'preview.html')

@app.route('/checkout.html')
def checkout():
    return send_from_directory('.', 'checkout.html')

# --- API ROUTE VOOR DE 3MF GENERATIE ---
@app.route('/api/generate', methods=['POST'])
def api_generate():
    # 1. Controleer of er een GPX bestand is meegestuurd
    if 'gpxFile' not in request.files:
        return jsonify({"error": "Geen GPX bestand geüpload"}), 400
    
    file = request.files['gpxFile']
    if file.filename == '':
        return jsonify({"error": "Leeg bestand"}), 400

    # 2. Lees het bestand uit het RAM-geheugen (geen trage opslag op schijf)
    gpx_data = file.read().decode('utf-8')

    # 3. Haal de slider-waarden op uit de POST-request
    try:
        config = GeneratorConfig(
            grid_resolutie=int(request.form.get('resolutie', 150)),
            marge_graden=float(request.form.get('marge', 0.03)),
            route_breedte=float(request.form.get('breedte', 2.0)),
            route_dikte=float(request.form.get('dikte', 1.0)),
            z_schaal=float(request.form.get('zschaal', 2.0)),
            model_grootte=float(request.form.get('grootte', 100.0)),
            boord_marge=float(request.form.get('boordmarge', 10.0)),
            boord_dikte=float(request.form.get('boorddikte', 5.0))
        )
    except ValueError as e:
        return jsonify({"error": "Fout in parameter waarden."}), 400

    # 4. Genereer het model
    try:
        # Functie wordt aangeroepen zonder output_filename, zodat hij bytes retouneert
        result_bytes = generate_terrain_and_route(gpx_data, config)
        
        if not result_bytes:
            return jsonify({"error": "Genereren mislukt (bijv. geen coördinaten in GPX)"}), 500

        # 5. Stuur het direct als een downloadbaar bestand terug naar de browser
        return send_file(
            io.BytesIO(result_bytes),
            mimetype='application/vnd.ms-3mfdocument',
            as_attachment=True,
            download_name='jouw_route_trofee.3mf'
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Nodig voor lokaal testen
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
