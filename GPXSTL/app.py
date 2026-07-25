import os
from flask import Flask, render_template, request, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import gpx_to_trophy

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# We slaan de bestandsnaam tijdelijk op in het geheugen voor deze sessie
current_session = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_preview', methods=['POST'])
def generate_preview():
    if 'gpx_file' not in request.files:
        return "Geen GPX bestand geselecteerd", 400
    
    file = request.files['gpx_file']
    if file.filename == '':
        return "Geen bestand gekozen", 400
    
    if file:
        filename = secure_filename(file.filename)
        gpx_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(gpx_path)
        
        # Sla parameters op in sessie
        current_session['gpx_path'] = gpx_path
        current_session['breedte'] = float(request.form.get('breedte', 2.0))
        current_session['dikte'] = float(request.form.get('dikte', 1.0))
        current_session['z_schaal'] = float(request.form.get('z_schaal', 2.0))
        current_session['grootte'] = float(request.form.get('grootte', 100.0))
        current_session['bodem'] = float(request.form.get('bodem', 5.0))
        current_session['marge'] = float(request.form.get('marge', 0.01))
        
        # PREVIEW: Gebruik een lage resolutie (bijv. 50) zodat het supersnel laadt
        preview_filename = "preview_model.stl"
        preview_path = os.path.join(OUTPUT_FOLDER, preview_filename)
        current_session['preview_path'] = preview_path
        
        try:
            gpx_to_trophy.generate_terrain_and_route(
                gpx_filename=gpx_path,
                stl_filename=preview_path,
                grid_resolutie=50,  # Laag voor snelheid!
                marge_graden=current_session['marge'],
                route_breedte=current_session['breedte'],
                route_dikte=current_session['dikte'],
                z_schaal=current_session['z_schaal'],
                model_grootte=current_session['grootte'],
                bodem_dikte=current_session['bodem']
            )
        except Exception as e:
            return f"Fout bij genereren preview: {str(e)}", 500
        
        return redirect(url_for('show_preview'))

@app.route('/preview')
def show_preview():
    return render_template('preview.html')

@app.route('/download-file/<filename>')
def download_file(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), as_attachment=True)

@app.route('/checkout')
def checkout():
    # Hier komt straks de koppeling met Stripe of Mollie
    return render_template('checkout.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)