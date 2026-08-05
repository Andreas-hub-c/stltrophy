import argparse
import math
import requests
import gpxpy
import srtm
import numpy as np
import trimesh
import gc  # Geheugenbeheer
from dataclasses import dataclass
from typing import Optional, Union
from trimesh.creation import box
from scipy.interpolate import splev, splprep
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates, uniform_filter, label
from matplotlib.path import Path  # Voor het opvullen van meren

# =====================================================================
# CONFIGURATIE VOOR WEB & API
# =====================================================================
@dataclass
class GeneratorConfig:
    grid_resolutie: int = 150       # Detail van het landschap
    marge_graden: float = 0.03      # Marge rondom de GPX route
    route_breedte: float = 2.0      # Breedte van de route in mm
    route_dikte: float = 1.0        # Hoogte van de route in mm
    z_schaal: float = 2.0           # Hoogte-overdrijving bergen
    model_grootte: float = 100.0    # Fysieke grootte (X/Y) in mm
    bodem_dikte: float = 5.0        # Dikte van de bodem in mm
    boord_marge: float = 10.0       # Zwarte sokkel uitsteek (mm)
    boord_dikte: float = 5.0        # Dikte zwarte sokkel (mm)
    water_diepte: float = 0.6       # Diepte van rivieren/meren in mm
    min_water_grootte: int = 150    # Filter voor overbodige kleine riviertjes/ruis


def fetch_osm_waterways(min_lat, min_lon, max_lat, max_lon):
    """Haalt gericht grote rivieren en meren op via OpenStreetMap API's (geen kleine sloten)."""
    print("Ophalen van gerichte OSM waterdata (meren & rivieren)...")
    overpass_urls = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter"
    ]
    
    # Gerichte query: alleen echte rivieren en stilstaand water (geen kleine beekjes/sloten)
    overpass_query = f"""
    [out:json][timeout:30];
    (
      way["waterway"="river"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["natural"="water"]({min_lat},{min_lon},{max_lat},{max_lon});
      relation["natural"="water"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["water"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["landuse"="reservoir"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out geom;
    """
    headers = {'User-Agent': 'GPX3MF-WebGenerator/9.0'}

    for url in overpass_urls:
        try:
            response = requests.post(url, data={'data': overpass_query}, headers=headers, timeout=40)
            if response.status_code == 200:
                data = response.json()
                segments = [[pt['lon'], pt['lat']] for el in data.get('elements', []) 
                            for geom in [el.get('geometry', el.get('coords', []))] if geom and len(geom) > 1 
                            for pt in geom]
                
                result = []
                idx = 0
                for el in data.get('elements', []):
                    geom_len = len(el.get('geometry', el.get('coords', [])))
                    if geom_len > 1:
                        result.append(np.array(segments[idx:idx+geom_len]))
                        idx += geom_len
                print(f"Succes: {len(result)} waterobjecten gevonden.")
                
                del data
                del segments
                gc.collect()

                return result
        except Exception:
            continue
    print("Waarschuwing: Kon geen waterdata ophalen.")
    return []

def fill_all_nans(grid_array):
    mask = np.isnan(grid_array) | (grid_array <= -500.0)
    if np.any(mask):
        filled = grid_array.copy()
        for _ in range(3):
            local_mean = uniform_filter(filled, size=5, mode='reflect')
            filled[mask] = local_mean[mask]
        if np.any(np.isnan(filled)):
            valid_median = np.nanmedian(filled)
            filled = np.nan_to_num(filled, nan=valid_median if not np.isnan(valid_median) else 100.0)
        return filled
    return grid_array

def filter_small_water_noise(water_mask, min_size=150):
    labeled_array, num_features = label(water_mask)
    if num_features == 0: return water_mask
    cleaned_mask = np.zeros_like(water_mask, dtype=bool)
    for i in range(1, num_features + 1):
        component = (labeled_array == i)
        if np.sum(component) >= min_size:
            cleaned_mask |= component
    return cleaned_mask

def draw_line_on_grid(ii0, ji0, ii1, ji1, grid_shape):
    points = []
    n = max(abs(ii1 - ii0), abs(ji1 - ji0))
    if n == 0:
        return [(np.clip(ii0, 0, grid_shape[0] - 1), np.clip(ji0, 0, grid_shape[1] - 1))]
    for step in range(n + 1):
        t = step / float(n)
        r = max(0, min(int(round((1.0 - t) * ii0 + t * ii1)), grid_shape[0] - 1))
        c = max(0, min(int(round((1.0 - t) * ji0 + t * ji1)), grid_shape[1] - 1))
        points.append((r, c))
    return points

# =====================================================================
# HOOFDFUNCTIE
# =====================================================================
def generate_terrain_and_route(
    gpx_input: Union[str, bytes],  
    config: GeneratorConfig,
    output_filename: Optional[str] = None
) -> Optional[bytes]:
    print("Start generatie proces...")

    try:
        if isinstance(gpx_input, str) and (gpx_input.endswith('.gpx') or gpx_input.endswith('.xml')):
            with open(gpx_input, "r") as f:
                gpx = gpxpy.parse(f)
        else:
            gpx = gpxpy.parse(gpx_input)
    except Exception as e:
        print(f"Fout bij parsen GPX: {e}")
        return None

    route_coords = np.array([[p.longitude, p.latitude] for track in gpx.tracks for segment in track.segments for p in segment.points])
    if len(route_coords) == 0:
        print("Geen coördinaten gevonden in GPX.")
        return None

    unieke_coords = route_coords[np.insert(np.any(np.diff(route_coords, axis=0), axis=1), 0, True)]
    tck, u = splprep([unieke_coords[:, 0], unieke_coords[:, 1]], s=0.0)
    smooth_lon, smooth_lat = splev(np.linspace(0, 1, len(unieke_coords) * 4), tck)
    route_coords = np.column_stack((smooth_lon, smooth_lat))
    N_route = len(route_coords)

    min_lon, max_lon = route_coords[:, 0].min() - config.marge_graden, route_coords[:, 0].max() + config.marge_graden
    min_lat, max_lat = route_coords[:, 1].min() - config.marge_graden, route_coords[:, 1].max() + config.marge_graden

    elevation_data = srtm.get_data()
    lons = np.linspace(min_lon, max_lon, config.grid_resolutie)
    lats = np.linspace(min_lat, max_lat, config.grid_resolutie)

    grid_z_echt = np.array([[elevation_data.get_elevation(lat, lon) or np.nan for lon in lons] for lat in lats])
    
    # 1. Detecteer zee op basis van lege SRTM data (NaN)
    initial_sea = np.isnan(grid_z_echt) | (grid_z_echt <= -500.0)
    sea_mask = filter_small_water_noise(initial_sea, min_size=config.min_water_grootte)

    # 2. Vul landgaten op
    grid_z_echt = fill_all_nans(grid_z_echt)
    
    water_mask = sea_mask.copy()
    river_mask = np.zeros_like(water_mask, dtype=bool)
    lake_mask = np.zeros_like(water_mask, dtype=bool)

    osm_segments = fetch_osm_waterways(min_lat, min_lon, max_lat, max_lon)
    if osm_segments:
        lon_step = (max_lon - min_lon) / (config.grid_resolutie - 1)
        lat_step = (max_lat - min_lat) / (config.grid_resolutie - 1)
        
        grid_c, grid_r = np.meshgrid(np.arange(config.grid_resolutie), np.arange(config.grid_resolutie))
        grid_points = np.column_stack((grid_c.ravel(), grid_r.ravel()))

        for seg in osm_segments:
            is_closed = len(seg) > 3 and np.allclose(seg[0], seg[-1], atol=1e-4)

            if is_closed:
                polygon_pixels = []
                for pt in seg:
                    c = (pt[0] - min_lon) / lon_step
                    r = (pt[1] - min_lat) / lat_step
                    polygon_pixels.append([c, r])
                try:
                    path = Path(polygon_pixels)
                    mask = path.contains_points(grid_points).reshape((config.grid_resolutie, config.grid_resolutie))
                    lake_mask[mask] = True
                except Exception as e:
                    print(f"Fout bij invullen meer polygoon: {e}")
            else:
                for k in range(len(seg) - 1):
                    ji0, ii0 = int(round((seg[k][0] - min_lon) / lon_step)), int(round((seg[k][1] - min_lat) / lat_step))
                    ji1, ii1 = int(round((seg[k+1][0] - min_lon) / lon_step)), int(round((seg[k+1][1] - min_lat) / lat_step))
                    for r, c in draw_line_on_grid(ii0, ji0, ii1, ji1, (config.grid_resolutie, config.grid_resolutie)):
                        river_mask[r, c] = True
        
        del osm_segments
        gc.collect()

    # MEREN KRIJGEN EEN ZEER LAGE DRIJFVERVALLING (Zodat ze nagenoeg altijd aanstaan en niet weggefilterd worden)
    lake_mask = filter_small_water_noise(lake_mask, min_size=10) 
    # RIVIEREN GEBRUIKEN DE GEBRUIKERSFILTER (min_water_grootte)
    if config.min_water_grootte > 0:
        river_mask = filter_small_water_noise(river_mask, min_size=config.min_water_grootte)

    water_mask = sea_mask | lake_mask | river_mask

    grid_z_smoothed = gaussian_filter(grid_z_echt, sigma=1.0)

    mean_lat = np.radians((min_lat + max_lat) / 2)
    max_range = max((lons[-1] - min_lon) * 111139.0 * math.cos(mean_lat), (lats[-1] - min_lat) * 111139.0)
    x_geschaald = ((lons - min_lon) * 111139.0 * math.cos(mean_lat) / max_range) * config.model_grootte
    y_geschaald = ((lats - min_lat) * 111139.0 / max_range) * config.model_grootte
    
    z_geschaald = np.maximum(0.5, ((grid_z_smoothed - grid_z_smoothed.min()) / max_range) * config.model_grootte * config.z_schaal)
    flat_sea_z = z_geschaald[water_mask & sea_mask].min() if np.any(water_mask & sea_mask) else 0.5

    xx, yy = np.meshgrid(x_geschaald, y_geschaald)
    z_off = config.boord_dikte 
    
    eff_water_depth = max(config.water_diepte, 0.4) if (np.any(river_mask) or np.any(lake_mask)) else config.water_diepte

    Z_terrein_top = z_geschaald.copy() + config.bodem_dikte + z_off
    Z_terrein_top[river_mask] = z_geschaald[river_mask] + config.bodem_dikte - eff_water_depth + z_off
    Z_terrein_top[sea_mask] = config.bodem_dikte + z_off

    # 100% VLAKKE HORIZONTALE WATERPIEGEL VOOR ELK MEER APART
    labeled_lakes, num_lakes = label(lake_mask)
    for l_idx in range(1, num_lakes + 1):
        l_comp = (labeled_lakes == l_idx)
        if not np.any(l_comp):
            continue
        lake_surface_z = np.min(z_geschaald[l_comp])
        Z_terrein_top[l_comp] = lake_surface_z + config.bodem_dikte - eff_water_depth + z_off

    Z_terrein_bot = np.full_like(Z_terrein_top, z_off)
    
    Z_water_bot = Z_terrein_top.copy()
    Z_water_top = Z_water_bot.copy()
    
    Z_water_top[sea_mask] = flat_sea_z + config.bodem_dikte + z_off
    Z_water_top[river_mask] = z_geschaald[river_mask] + config.bodem_dikte + z_off

    for l_idx in range(1, num_lakes + 1):
        l_comp = (labeled_lakes == l_idx)
        if not np.any(l_comp):
            continue
        lake_surface_z = np.min(z_geschaald[l_comp])
        Z_water_top[l_comp] = lake_surface_z + config.bodem_dikte + z_off

    Z_water_top = np.maximum(Z_water_top, Z_water_bot)

    r = config.grid_resolutie
    offset = r * r

    t_verts = np.vstack((np.stack((xx, yy, Z_terrein_top), axis=-1).reshape(-1, 3), np.stack((xx, yy, Z_terrein_bot), axis=-1).reshape(-1, 3)))
    w_verts = np.vstack((np.stack((xx, yy, Z_water_top), axis=-1).reshape(-1, 3), np.stack((xx, yy, Z_water_bot), axis=-1).reshape(-1, 3)))
    
    i_idx, j_idx = np.meshgrid(np.arange(r - 1), np.arange(r - 1), indexing="ij")
    p1 = i_idx * r + j_idx; p2 = p1 + 1; p3 = (i_idx + 1) * r + j_idx; p4 = p3 + 1
    b1, b2, b3, b4 = p1 + offset, p2 + offset, p3 + offset, p4 + offset
    
    t_f_top = np.vstack((np.column_stack((p1.ravel(), p2.ravel(), p3.ravel())), np.column_stack((p2.ravel(), p4.ravel(), p3.ravel()))))
    t_f_bot = np.vstack((np.column_stack((b1.ravel(), b3.ravel(), b2.ravel())), np.column_stack((b2.ravel(), b3.ravel(), b4.ravel()))))

    w_thick_mask = (Z_water_top - Z_water_bot) > 0.001
    w_active = w_thick_mask[:-1, :-1] | w_thick_mask[1:, :-1] | w_thick_mask[:-1, 1:] | w_thick_mask[1:, 1:]

    t_f_sides, w_f_top, w_f_bot, w_f_sides = [], [], [], []
    for i in range(r - 1):
        for j in range(r - 1):
            cur_p1, cur_p2, cur_p3, cur_p4 = i*r+j, i*r+j+1, (i+1)*r+j, (i+1)*r+j+1
            cur_b1, cur_b2, cur_b3, cur_b4 = cur_p1+offset, cur_p2+offset, cur_p3+offset, cur_p4+offset
            
            if i == 0: t_f_sides.extend([[cur_p1, cur_b1, cur_p2], [cur_p2, cur_b1, cur_b2]])
            if i == r - 2: t_f_sides.extend([[cur_p3, cur_p4, cur_b3], [cur_p4, cur_b4, cur_b3]])
            if j == 0: t_f_sides.extend([[cur_p1, cur_p3, cur_b3], [cur_p1, cur_b3, cur_b1]])
            if j == r - 2: t_f_sides.extend([[cur_p2, cur_b4, cur_p4], [cur_p2, cur_b2, cur_b4]])
            
            if w_active[i, j]:
                w_f_top.extend([[cur_p1, cur_p2, cur_p3], [cur_p2, cur_p4, cur_p3]])
                w_f_bot.extend([[cur_b1, cur_b3, cur_b2], [cur_b2, cur_b3, cur_b4]])
                if i == 0 or not w_active[i-1, j]: w_f_sides.extend([[cur_p1, cur_b1, cur_p2], [cur_p2, cur_b1, cur_b2]])
                if i == r - 2 or not w_active[i+1, j]: w_f_sides.extend([[cur_p3, cur_p4, cur_b3], [cur_p4, cur_b4, cur_b3]])
                if j == 0 or not w_active[i, j-1]: w_f_sides.extend([[cur_p1, cur_p3, cur_b3], [cur_p1, cur_b3, cur_b1]])
                if j == r - 2 or not w_active[i, j+1]: w_f_sides.extend([[cur_p2, cur_b4, cur_p4], [cur_p2, cur_b2, cur_b4]])

    terrain_mesh = trimesh.Trimesh(vertices=t_verts, faces=np.vstack((t_f_top, t_f_bot, np.array(t_f_sides))), process=True)
    trimesh.repair.fix_normals(terrain_mesh)

    water_mesh = trimesh.Trimesh()
    if w_f_top:
        water_mesh = trimesh.Trimesh(vertices=w_verts, faces=np.vstack((w_f_top, w_f_bot, w_f_sides)), process=True)
        water_mesh.remove_unreferenced_vertices()
        trimesh.repair.fix_normals(water_mesh)

    gx_idx = np.clip((route_coords[:, 0] - min_lon) / (max_lon - min_lon), 0.0, 1.0) * (r - 1)
    gy_idx = np.clip((route_coords[:, 1] - min_lat) / (max_lat - min_lat), 0.0, 1.0) * (r - 1)

    z_val = map_coordinates(np.maximum(Z_terrein_top, Z_water_top), np.vstack((gy_idx, gx_idx)), order=1, mode="nearest")
    rx_mm = ((route_coords[:, 0] - min_lon) * 111139.0 * math.cos(mean_lat) / max_range) * config.model_grootte
    ry_mm = ((route_coords[:, 1] - min_lat) * 111139.0 / max_range) * config.model_grootte

    route_3d = np.column_stack((rx_mm, ry_mm, gaussian_filter1d(z_val, sigma=1.0)))
    dx, dy = np.gradient(route_3d[:, 0]), np.gradient(route_3d[:, 1])
    lengths = np.where(np.hypot(dx, dy) == 0, 1.0, np.hypot(dx, dy))

    offset_x = -dy / lengths * (config.route_breedte / 2)
    offset_y = dx / lengths * (config.route_breedte / 2)
    
    route_verts = np.vstack((
        np.column_stack((route_3d[:, 0] + offset_x, route_3d[:, 1] + offset_y, route_3d[:, 2] + config.route_dikte)),
        np.column_stack((route_3d[:, 0] - offset_x, route_3d[:, 1] - offset_y, route_3d[:, 2] + config.route_dikte)),
        np.column_stack((route_3d[:, 0] + offset_x, route_3d[:, 1] + offset_y, route_3d[:, 2])),
        np.column_stack((route_3d[:, 0] - offset_x, route_3d[:, 1] - offset_y, route_3d[:, 2]))
    ))

    i_arr = np.arange(N_route - 1)
    r_faces = np.vstack([
        np.column_stack((i_arr, i_arr + 1, i_arr + N_route)),
        np.column_stack((i_arr + 1, i_arr + N_route + 1, i_arr + N_route)),
        np.column_stack((i_arr + 2 * N_route, i_arr + 3 * N_route, i_arr + 2 * N_route + 1)),
        np.column_stack((i_arr + 2 * N_route + 1, i_arr + 3 * N_route, i_arr + 3 * N_route + 1)),
        np.column_stack((i_arr, i_arr + 2 * N_route, i_arr + 1)),
        np.column_stack((i_arr + 1, i_arr + 2 * N_route, i_arr + 2 * N_route + 1)),
        np.column_stack((i_arr + N_route, i_arr + N_route + 1, i_arr + 3 * N_route)),
        np.column_stack((i_arr + N_route + 1, i_arr + 3 * N_route + 1, i_arr + 3 * N_route)),
    ])

    route_trimesh = trimesh.Trimesh(vertices=route_verts, faces=r_faces)
    route_trimesh.merge_vertices()
    trimesh.repair.fix_normals(route_trimesh)

    bx_min, bx_max = x_geschaald.min() - config.boord_marge, x_geschaald.max() + config.boord_marge
    by_min, by_max = y_geschaald.min() - config.boord_marge, y_geschaald.max() + config.boord_marge
    
    transform_matrix = np.eye(4)
    transform_matrix[0, 3] = (bx_max + bx_min) / 2.0
    transform_matrix[1, 3] = (by_max + by_min) / 2.0
    transform_matrix[2, 3] = config.boord_dikte / 2.0
    boord_mesh = box(extents=[bx_max - bx_min, by_max - by_min, config.boord_dikte], transform=transform_matrix)

    boord_mesh.visual.face_colors = [30, 30, 30, 255]      
    terrain_mesh.visual.face_colors = [120, 200, 120, 255] 
    route_trimesh.visual.face_colors = [255, 0, 0, 255]    
    if not water_mesh.is_empty: water_mesh.visual.face_colors = [50, 100, 255, 255] 

    scene = trimesh.Scene({'1_Zwarte_Sokkel': boord_mesh, '2_Terrein': terrain_mesh, '3_Water': water_mesh, '4_Route': route_trimesh})
    
    if not output_filename:
        return scene.export(file_type='3mf')
        
    if not output_filename.endswith('.3mf'):
        output_filename = output_filename.rsplit('.', 1)[0] + '.3mf'
    scene.export(output_filename, file_type='3mf')
    print(f"Succes! Opgeslagen als '{output_filename}'.")
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Web-ready Generator voor 3D GPX modellen.")
    parser.add_argument("gpx_bestand", help="Pad naar GPX-bestand")
    parser.add_argument("-o", "--output", default="trofee.3mf", help="Uitvoerbestand")
    
    default_config = GeneratorConfig()
    parser.add_argument("--resolutie", type=int, default=default_config.grid_resolutie)
    parser.add_argument("--marge", type=float, default=default_config.marge_graden)
    parser.add_argument("--breedte", type=float, default=default_config.route_breedte)
    parser.add_argument("--dikte", type=float, default=default_config.route_dikte)
    parser.add_argument("--z_schaal", type=float, default=default_config.z_schaal)
    parser.add_argument("--grootte", type=float, default=default_config.model_grootte)
    parser.add_argument("--bodem", type=float, default=default_config.bodem_dikte)
    parser.add_argument("--boord_marge", type=float, default=default_config.boord_marge)
    parser.add_argument("--boord_dikte", type=float, default=default_config.boord_dikte)
    parser.add_argument("--water_diepte", type=float, default=default_config.water_diepte)
    parser.add_argument("--min_water_grootte", type=int, default=default_config.min_water_grootte)

    args = parser.parse_args()

    config = GeneratorConfig(
        grid_resolutie=args.resolutie,
        marge_graden=args.marge,
        route_breedte=args.breedte,
        route_dikte=args.dikte,
        z_schaal=args.z_schaal,
        model_grootte=args.grootte,
        bodem_dikte=args.bodem,
        boord_marge=args.boord_marge,
        boord_dikte=args.boord_dikte,
        water_diepte=args.water_diepte,
        min_water_grootte=args.min_water_grootte
    )

    generate_terrain_and_route(args.gpx_bestand, config, args.output)
