import gpxpy
import numpy as np
from stl import mesh
import math
import srtm
import argparse
from scipy.ndimage import gaussian_filter, gaussian_filter1d, uniform_filter, map_coordinates
from scipy.interpolate import splprep, splev

def fix_lake_holes(grid_array):
    """
    Vult gaten (NaN waarden of onrealistische nullen bij meren) op 
    door ze te interpoleren op basis van de omliggende bergen/oevers.
    """
    mask = (grid_array == 0) | (np.isnan(grid_array))
    
    if np.any(mask):
        filled = grid_array.copy()
        local_mean = uniform_filter(filled, size=5, mode='reflect')
        filled[mask] = local_mean[mask]
        return filled
    
    return grid_array

def generate_terrain_and_route(gpx_filename, stl_filename, grid_resolutie, marge_graden, route_breedte, route_dikte, z_schaal, model_grootte, bodem_dikte):
    print(f"Inlezen van {gpx_filename} en berekenen van het 3D-model (resolutie: {grid_resolutie})...")

    # 1. GPX Inlezen
    try:
        with open(gpx_filename, 'r') as gpx_file:
            gpx = gpxpy.parse(gpx_file)
    except FileNotFoundError:
        print(f"Fout: Kan GPX bestand '{gpx_filename}' niet vinden.")
        return

    route_coords = []
    for track in gpx.tracks:
        for segment in track.segments:
            for p in segment.points:
                route_coords.append([p.longitude, p.latitude])
    
    route_coords = np.array(route_coords)

    # --- ROUTE SMOOTHEN ---
    unieke_coords = [route_coords[0]]
    for p in route_coords[1:]:
        if not np.allclose(p, unieke_coords[-1]):
            unieke_coords.append(p)
    unieke_coords = np.array(unieke_coords)

    smoothing_mate = 0.0  
    detail_factor = 4     

    tck, u = splprep([unieke_coords[:, 0], unieke_coords[:, 1]], s=smoothing_mate)
    u_nieuw = np.linspace(0, 1, len(unieke_coords) * detail_factor)
    smooth_lon, smooth_lat = splev(u_nieuw, tck)
    
    route_coords = np.column_stack((smooth_lon, smooth_lat))
    N_route = len(route_coords)

    # 2. Bounding Box bepalen
    min_lon, max_lon = route_coords[:, 0].min() - marge_graden, route_coords[:, 0].max() + marge_graden
    min_lat, max_lat = route_coords[:, 1].min() - marge_graden, route_coords[:, 1].max() + marge_graden

    # 3. Terrein Grid genereren en Hoogtes ophalen
    elevation_data = srtm.get_data()
    lons = np.linspace(min_lon, max_lon, grid_resolutie)
    lats = np.linspace(min_lat, max_lat, grid_resolutie)
    grid_z = np.zeros((grid_resolutie, grid_resolutie))
    
    for i in range(grid_resolutie):
        for j in range(grid_resolutie):
            hoogte = elevation_data.get_elevation(lats[i], lons[j])
            grid_z[i, j] = hoogte if hoogte is not None else 0.0

    # --- MEREN FIX: Gaten en oneffenheden opvullen ---
    grid_z = fix_lake_holes(grid_z)

    grid_z_smoothed = gaussian_filter(grid_z, sigma=0.8)

    # 4. Omzetten naar Meters en schalen
    mean_lat = np.radians((min_lat + max_lat) / 2)
    x_meters = (lons - min_lon) * 111139.0 * math.cos(mean_lat)
    y_meters = (lats - min_lat) * 111139.0
    
    max_range = max(x_meters[-1], y_meters[-1])
    
    x_geschaald = (x_meters / max_range) * model_grootte
    y_geschaald = (y_meters / max_range) * model_grootte
    z_min_echt = grid_z_smoothed.min()
    z_geschaald = ((grid_z_smoothed - z_min_echt) / max_range) * model_grootte * z_schaal

    # 5. TERREIN MESH BOUWEN (Compleet massief blok)
    faces = []
    vertices = []
    
    for i in range(grid_resolutie):
        for j in range(grid_resolutie):
            vertices.append([x_geschaald[j], y_geschaald[i], z_geschaald[i, j] + bodem_dikte])

    offset = grid_resolutie * grid_resolutie
    for i in range(grid_resolutie):
        for j in range(grid_resolutie):
            vertices.append([x_geschaald[j], y_geschaald[i], 0.0])

    for i in range(grid_resolutie - 1):
        for j in range(grid_resolutie - 1):
            p1 = i * grid_resolutie + j
            p2 = p1 + 1
            p3 = (i + 1) * grid_resolutie + j
            p4 = p3 + 1
            faces.extend([[p1, p2, p3], [p2, p4, p3]])

    for i in range(grid_resolutie - 1):
        for j in range(grid_resolutie - 1):
            p1 = offset + i * grid_resolutie + j
            p2 = p1 + 1
            p3 = offset + (i + 1) * grid_resolutie + j
            p4 = p3 + 1
            faces.extend([[p1, p3, p2], [p2, p3, p4]])

    for j in range(grid_resolutie - 1):
        t1, t2 = j, j + 1
        b1, b2 = offset + j, offset + j + 1
        faces.extend([[t1, b1, t2], [t2, b1, b2]])
        
        idx_top = (grid_resolutie - 1) * grid_resolutie + j
        t1, t2 = idx_top, idx_top + 1
        b1, b2 = offset + idx_top, offset + idx_top + 1
        faces.extend([[t1, t2, b1], [t2, b2, b1]])

    for i in range(grid_resolutie - 1):
        idx_left = i * grid_resolutie
        t1, t2 = idx_left, idx_left + grid_resolutie
        b1, b2 = offset + idx_left, offset + idx_left + grid_resolutie
        faces.extend([[t1, t2, b1], [t2, b2, b1]])
        
        idx_right = i * grid_resolutie + (grid_resolutie - 1)
        t1, t2 = idx_right, idx_right + grid_resolutie
        b1, b2 = offset + idx_right, offset + idx_right + grid_resolutie
        faces.extend([[t1, b1, t2], [t2, b1, b2]])

    vertices = np.array(vertices)
    terrain_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            terrain_mesh.vectors[i][j] = vertices[f[j], :]

    # 6. ROUTE PROJECTEREN OP HET GESATELEERDE & GESMOOTHED TERREIN
    route_3d = []
    for lon, lat in route_coords:
        # Bepaal exacte coördinaten in raster-indices (sub-pixel precisie voor vloeiende interpolatie)
        xf = max(0.0, min(1.0, (lon - min_lon) / (max_lon - min_lon)))
        yf = max(0.0, min(1.0, (lat - min_lat) / (max_lat - min_lat)))
        
        gx_idx = xf * (grid_resolutie - 1)
        gy_idx = yf * (grid_resolutie - 1)
        
        # Haal de exacte gesoomete terreinhoogte op via map_coordinates (sub-pixel accurate bilinear/spline lookup)
        z_terrain_val = map_coordinates(z_geschaald, [[gy_idx], [gx_idx]], order=1, mode='nearest')[0] + bodem_dikte
        
        rx_m = (lon - min_lon) * 111139.0 * math.cos(mean_lat)
        ry_m = (lat - min_lat) * 111139.0
        rx_mm = (rx_m / max_range) * model_grootte
        ry_mm = (ry_m / max_range) * model_grootte
        
        route_3d.append([rx_mm, ry_mm, z_terrain_val])
    
    route_3d = np.array(route_3d)
    # Lichte smoothing van de routehoogte zodat deze mooi meebeweegt
    route_3d[:, 2] = gaussian_filter1d(route_3d[:, 2], sigma=2.0)

    # 7. ROUTE MESH GENEREREN (Tot aan de bodem)
    route_verts = np.zeros((N_route * 4, 3))
    
    for i in range(N_route):
        if i < N_route - 1:
            dx = route_3d[i+1, 0] - route_3d[i, 0]
            dy = route_3d[i+1, 1] - route_3d[i, 1]
        else:
            dx = route_3d[i, 0] - route_3d[i-1, 0]
            dy = route_3d[i, 1] - route_3d[i-1, 1]
            
        length = math.hypot(dx, dy)
        if length == 0: length = 1
        
        offset_x = -dy / length * (route_breedte / 2)
        offset_y = dx / length * (route_breedte / 2)
        
        z_top = route_3d[i, 2] + route_dikte
        z_bodem = 0.0  
        
        route_verts[i] = [route_3d[i, 0] + offset_x, route_3d[i, 1] + offset_y, z_top]
        route_verts[i + N_route] = [route_3d[i, 0] - offset_x, route_3d[i, 1] - offset_y, z_top]
        route_verts[i + 2*N_route] = [route_3d[i, 0] + offset_x, route_3d[i, 1] + offset_y, z_bodem] 
        route_verts[i + 3*N_route] = [route_3d[i, 0] - offset_x, route_3d[i, 1] - offset_y, z_bodem] 

    route_faces = []
    for i in range(N_route - 1):
        route_faces.extend([
            [i, i + 1, i + N_route], [i + 1, i + N_route + 1, i + N_route],                     
            [i + 2*N_route, i + 3*N_route, i + 2*N_route + 1], [i + 2*N_route + 1, i + 3*N_route, i + 3*N_route + 1], 
            [i, i + 2*N_route, i + 1], [i + 1, i + 2*N_route, i + 2*N_route + 1],                
            [i + N_route, i + N_route + 1, i + 3*N_route], [i + N_route + 1, i + 3*N_route + 1, i + 3*N_route]   
        ])

    route_faces.extend([
        [0, N_route, 2*N_route], [N_route, 3*N_route, 2*N_route],
        [N_route-1, 2*N_route-1, 3*N_route-1], [2*N_route-1, 4*N_route-1, 3*N_route-1]
    ])

    route_mesh = mesh.Mesh(np.zeros(len(route_faces), dtype=mesh.Mesh.dtype))
    for i, f in enumerate(route_faces):
        for j in range(3):
            route_mesh.vectors[i][j] = route_verts[f[j], :]

    # 8. ALLES SAMENVOEGEN
    combined_mesh = mesh.Mesh(np.concatenate([terrain_mesh.data, route_mesh.data]))
    combined_mesh.save(stl_filename)
    print(f"Succes! Opgeslagen als '{stl_filename}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genereer een 3D-printbare bergtrofee van een GPX-bestand.")
    
    parser.add_argument("gpx_bestand", help="Pad naar het invoer GPX-bestand (bijv. mijn_rit.gpx)")
    parser.add_argument("-o", "--output", default="complete_3d_trofee.stl", help="Naam van het STL uitvoerbestand")
    parser.add_argument("--resolutie", type=int, default=150, help="Detail van het landschap (standaard: 150)")
    parser.add_argument("--breedte", type=float, default=2.0, help="Breedte van de route in mm (standaard: 2.0)")
    parser.add_argument("--dikte", type=float, default=1.0, help="Hoogte van de route boven de berg in mm (standaard: 1.0)")
    parser.add_argument("--z_schaal", type=float, default=2.0, help="Hoogte-overdrijving van de bergen (standaard: 2.0)")
    parser.add_argument("--grootte", type=float, default=100.0, help="Totale lengte van het model in mm (standaard: 100.0)")
    parser.add_argument("--bodem", type=float, default=5.0, help="Dikte van de massieve bodemplaat (standaard: 5.0)")
    parser.add_argument("--marge", type=float, default=0.01, help="Extra ruimte rondom de route in graden (standaard: 0.01)")

    args = parser.parse_args()

    generate_terrain_and_route(
        gpx_filename=args.gpx_bestand,
        stl_filename=args.output,
        grid_resolutie=args.resolutie,
        marge_graden=args.marge,
        route_breedte=args.breedte,
        route_dikte=args.dikte,
        z_schaal=args.z_schaal,
        model_grootte=args.grootte,
        bodem_dikte=args.bodem
    )