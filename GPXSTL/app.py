import os
import io
from flask import Flask, request, send_file, render_template, jsonify

# We importeren jouw geoptimaliseerde generator logica uit generator.py
from generator import generate_terrain_and_route, GeneratorConfig

app = Flask(__name__)

# =====================================================================
# ROUTES VOOR DE HTML PAGINA'S (UIT DE 'TEMPLATES' MAP)
# =====================================================================

# Vangt zowel de kale link (/) als (/index.html) af
@app.route('/')
@app.route('/index.html')
def index():
    # Flask kijkt automatisch in de map genaamd 'templates'
    return render_template('index.html')

@app.route('/preview.html')
def preview():
    return render_template('preview.html')

@app.route('/checkout.html')
def checkout():
    return render_template('checkout.html')


# =====================================================================
# API ROUTE VOOR DE 3MF GENERATIE
# =====================================================================
@app.route('/api/generate', methods=['POST'])
def api_generate():
    # 1. Controleer of er daadwerkelijk een bestand is meegestuurd vanuit het HTML formulier
    if 'gpxFile' not in request.files:
        return jsonify({"error": "Geen GPX bestand geüpload"}), 400
    
    file = request.files['gpxFile']
    if file.filename == '':
        return jsonify({"error": "Leeg bestand"}), 400

    # 2. Lees het bestand rechtstreeks uit het werkgeheugen (RAM) voor maximale snelheid
    try:
        gpx_data = file.read().decode('utf-8')
    except Exception as e:
        return jsonify({"error": f"Fout bij lezen van GPX bestand: {str(e)}"}), 400

    # 3. Haal alle slider-waarden veilig op uit de POST-request en stop ze in de Config
    try:
        config = GeneratorConfig(
            grid_resolutie=int(request.form.get('resolutie', 150)),
            marge_graden=float(request.form.get('marge', 0.03)),
            route_breedte=float(request.form.get('breedte', 2.0)),
            route_dikte=float(request.form.get('dikte', 1.0)),
            z_schaal=float(request.form.get('zschaal', 2.0)),
            model_grootte=float(request.form.get('grootte', 100.0)),
            boord_marge=float(request.form.get('boordmarge', 10.0)),
            boord_dikte=float(request.form.get('boorddikte', 5.0)),
            water_diepte=float(request.form.get('water_diepte', 0.6)),             # <-- Toegevoegd
            min_water_grootte=int(request.form.get('min_water_grootte', 150))    # <-- Toegevoegd
        )
    except ValueError as e:
        # Als iemand vreemde tekst invult in plaats van getallen, vangen we dat netjes op
        return jsonify({"error": "Ongeldige parameter waarden. Er werden getallen verwacht."}), 400

    # 4. Roep de zware generatie-logica aan
    try:
        # Functie wordt aangeroepen zonder output_filename, zodat hij puur bytes (de 3MF raw data) retouneert
        result_bytes = generate_terrain_and_route(gpx_data, config)
        
        if not result_bytes:
            return jsonify({"error": "Genereren mislukt. Bevat dit GPX-bestand wel route-coördinaten?"}), 500

        # 5. Stuur de gegenereerde 3MF-file direct als download terug naar de browser van de bezoeker
        return send_file(
            io.BytesIO(result_bytes),
            mimetype='application/vnd.ms-3mfdocument',
            as_attachment=True,
            download_name='jouw_route_trofee.3mf'
        )
    except Exception as e:
        # Vangt fouten in de generator code (generator.py) op en toont ze netjes
        return jsonify({"error": f"Interne serverfout tijdens generatie: {str(e)}"}), 500


# =====================================================================
# SERVER STARTUP LOKAAL (Render gebruikt gunicorn, maar dit helpt bij testen op je eigen pc)
# =====================================================================
if __name__ == '__main__':
    # Pak de Render PORT environment variabele, of val terug op poort 5000 voor lokale tests
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
