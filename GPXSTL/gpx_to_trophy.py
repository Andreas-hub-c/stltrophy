import argparse
import math
import gpxpy
import srtm
import numpy as np
from scipy.interpolate import splev, splprep
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates, uniform_filter
from stl import mesh


def fix_lake_holes(grid_array):
    """Vult gaten (NaN waarden of onrealistische nullen bij meren) op

    door ze te interpoleren op basis van de omliggende bergen/oevers.
    """
    mask = (grid_array == 0) | (np.isnan(grid_array))

    if np.any(mask):
      filled = grid_array.copy()
      local_mean = uniform_filter(filled, size=5, mode='reflect')
      filled[mask] = local_mean[mask]
      return filled

    return grid_array


def generate_terrain_and_route(
    gpx_filename,
    stl_filename,
    grid_resolutie,
    marge_graden,
    route_breedte,
    route_dikte,
    z_schaal,
    model_grootte,
    bodem_dikte,
):
  print(
      f"Inlezen van {gpx_filename} en berekenen van het 3D-model (resolutie:"
      f" {grid_resolutie})..."
  )

  # 1. GPX Inlezen
  try:
    with open(gpx_filename, "r") as gpx_file:
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
  min_lon, max_lon = (
      route_coords[:, 0].min() - marge_graden,
      route_coords[:, 0].max() + marge_graden,
  )
  min_lat, max_lat = (
      route_coords[:, 1].min() - marge_graden,
      route_coords[:, 1].max() + marge_graden,
  )

  # 3. Terrein Grid genereren en Hoogtes ophalen (GEVECTORISEERD ipv trage lussen)
  elevation_data = srtm.get_data()
  lons = np.linspace(min_lon, max_lon, grid_resolutie)
  lats = np.linspace(min_lat, max_lat, grid_resolutie)

  # Vectorized elevation lookup using outer loops via list comprehension (sneller) of np.fromfunction
  grid_z = np.empty((grid_resolutie, grid_resolutie))
  for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
      h = elevation_data.get_elevation(lat, lon)
      grid_z[i, j] = h if h is not None else 0.0

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
  z_geschaald = (
      (grid_z_smoothed - z_min_echt) / max_range
  ) * model_grootte * z_schaal

  # 5. TERREIN MESH BOUWEN (Geoptimaliseerd met NumPy rasters)
  xx, yy = np.meshgrid(x_geschaald, y_geschaald)
  zz_top = z_geschaald + bodem_dikte
  zz_bodem = np.zeros_like(zz_top)

  # Maak vertices in één keer aan
  top_verts = np.stack(
      (xx.ravel(), yy.ravel(), zz_top.ravel()), axis=-1
  )  # Index 0 tot N
  bottom_verts = np.stack(
      (xx.ravel(), yy.ravel(), zz_bodem.ravel()), axis=-1
  )  # Index N tot 2N
  vertices = np.vstack((top_verts, bottom_verts))

  # Genereer indices voor de rasterstructuur in NumPy
  r = grid_resolutie
  i_idx, j_idx = np.meshgrid(
      np.arange(r - 1), np.arange(r - 1), indexing="ij"
  )
  p1 = i_idx * r + j_idx
  p2 = p1 + 1
  p3 = (i_idx + 1) * r + j_idx
  p4 = p3 + 1

  # Top vlakken
  f_top1 = np.stack((p1.ravel(), p2.ravel(), p3.ravel()), axis=-1)
  f_top2 = np.stack((p2.ravel(), p4.ravel(), p3.ravel()), axis=-1)

  # Bodem vlakken (omgekeerde winding)
  offset = r * r
  f_bot1 = f_top1 + offset
  f_bot2 = f_top2 + offset
  # Wissel p3 en p2 om voor correcte normale richtingen aan de onderkant
  f_bot1[:, [1, 2]] = f_bot1[:, [2, 1]]
  f_bot2[:, [1, 2]] = f_bot2[:, [2, 1]]

  # Randen (zijkanten van het blok)
  j_vals = np.arange(r - 1)
  t1_top = j_vals
  t2_top = j_vals + 1
  b1_bot = offset + j_vals
  b2_bot = offset + j_vals + 1
  side_front = np.stack(
      (t1_top, b1_bot, t2_top, t2_top, b1_bot, b2_bot), axis=-1
  ).reshape(-1, 3)

  idx_top_row = (r - 1) * r + j_vals
  t1_back = idx_top_row
  t2_back = idx_top_row + 1
  b1_back = offset + idx_top_row
  b2_back = offset + idx_top_row + 1
  side_back = np.stack(
      (t1_back, t2_back, b1_back, t2_back, b2_back, b1_back), axis=-1
  ).reshape(-1, 3)

  i_vals = np.arange(r - 1)
  idx_left = i_vals * r
  t1_l = idx_left
  t2_l = idx_left + r
  b1_l = offset + idx_left
  b2_l = offset + idx_left + r
  side_left = np.stack(
      (t1_l, t2_l, b1_l, t2_l, b2_l, b1_l), axis=-1
  ).reshape(-1, 3)

  idx_right = i_vals * r + (r - 1)
  t1_r = idx_right
  t2_r = idx_right + r
  b1_r = offset + idx_right
  b2_r = offset + idx_right + r
  side_right = np.stack(
      (t1_r, b1_r, t2_r, t2_r, b1_r, b2_r), axis=-1
  ).reshape(-1, 3)

  faces = np.vstack((
      f_top1,
      f_top2,
      f_bot1,
      f_bot2,
      side_front,
      side_back,
      side_left,
      side_right,
  ))

  terrain_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
  terrain_mesh.vectors[:] = vertices[faces]

  # 6. ROUTE PROJECTEREN OP HET TERREIN
  xf = (route_coords[:, 0] - min_lon) / (max_lon - min_lon)
  yf = (route_coords[:, 1] - min_lat) / (max_lat - min_lat)
  gx_idx = np.clip(xf, 0.0, 1.0) * (grid_resolutie - 1)
  gy_idx = np.clip(yf, 0.0, 1.0) * (grid_resolutie - 1)

  z_terrain_val = (
      map_coordinates(
          z_geschaald, np.vstack((gy_idx, gx_idx)), order=1, mode="nearest"
      )
      + bodem_dikte
  )

  rx_m = (route_coords[:, 0] - min_lon) * 111139.0 * math.cos(mean_lat)
  ry_m = (route_coords[:, 1] - min_lat) * 111139.0
  rx_mm = (rx_m / max_range) * model_grootte
  ry_mm = (ry_m / max_range) * model_grootte

  route_3d = np.column_stack(
      (rx_mm, ry_mm, z_terrain_val + route_dikte)
  )  # Direct top-offset
  route_3d[:, 2] = gaussian_filter1d(route_3d[:, 2], sigma=2.0)

  # 7. ROUTE MESH GENEREREN
  dx = np.gradient(route_3d[:, 0])
  dy = np.gradient(route_3d[:, 1])
  lengths = np.hypot(dx, dy)
  lengths[lengths == 0] = 1.0

  offset_x = -dy / lengths * (route_breedte / 2)
  offset_y = dx / lengths * (route_breedte / 2)

  z_top_arr = route_3d[:, 2]
  z_bodem_arr = np.zeros(N_route)

  route_verts = np.vstack((
      np.column_stack(
          (route_3d[:, 0] + offset_x, route_3d[:, 1] + offset_y, z_top_arr)
      ),
      np.column_stack(
          (route_3d[:, 0] - offset_x, route_3d[:, 1] - offset_y, z_top_arr)
      ),
      np.column_stack(
          (route_3d[:, 0] + offset_x, route_3d[:, 1] + offset_y, z_bodem_arr)
      ),
      np.column_stack(
          (route_3d[:, 0] - offset_x, route_3d[:, 1] - offset_y, z_bodem_arr)
      ),
  ))

  i_arr = np.arange(N_route - 1)
  r_faces = []
  for i in i_arr:
    r_faces.extend([
        [i, i + 1, i + N_route],
        [i + 1, i + N_route + 1, i + N_route],
        [
            i + 2 * N_route,
            i + 3 * N_route,
            i + 2 * N_route + 1,
        ],
        [
            i + 2 * N_route + 1,
            i + 3 * N_route,
            i + 3 * N_route + 1,
        ],
        [i, i + 2 * N_route, i + 1],
        [i + 1, i + 2 * N_route, i + 2 * N_route + 1],
        [i + N_route, i + N_route + 1, i + 3 * N_route],
        [i + N_route + 1, i + 3 * N_route + 1, i + 3 * N_route],
    ])

  r_faces.extend([
      [0, N_route, 2 * N_route],
      [N_route, 3 * N_route, 2 * N_route],
      [N_route - 1, 2 * N_route - 1, 3 * N_route - 1],
      [2 * N_route - 1, 4 * N_route - 1, 3 * N_route - 1],
  ])

  route_faces = np.array(r_faces)
  route_mesh = mesh.Mesh(np.zeros(len(route_faces), dtype=mesh.Mesh.dtype))
  route_mesh.vectors[:] = route_verts[route_faces]

  # 8. ALLES SAMENVOEGEN
  combined_mesh = mesh.Mesh(
      np.concatenate([terrain_mesh.data, route_mesh.data])
  )
  combined_mesh.save(stl_filename)
  print(f"Succes! Opgeslagen als '{stl_filename}'.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description=(
          "Genereer een 3D-printbare bergtrofee van een GPX-bestand."
      )
  )

  parser.add_argument(
      "gpx_bestand",
      help="Pad naar het invoer GPX-bestand (bijv. mijn_rit.gpx)",
  )
  parser.add_argument(
      "-o",
      "--output",
      default="complete_3d_trofee.stl",
      help="Naam van het STL uitvoerbestand",
  )
  parser.add_argument(
      "--resolutie",
      type=int,
      default=150,
      help="Detail van het landschap (standaard: 150)",
  )
  parser.add_argument(
      "--breedte",
      type=float,
      default=2.0,
      help="Breedte van de route in mm (standaard: 2.0)",
  )
  parser.add_argument(
      "--dikte",
      type=float,
      default=1.0,
      help="Hoogte van de route boven de berg in mm (standaard: 1.0)",
  )
  parser.add_argument(
      "--z_schaal",
      type=float,
      default=2.0,
      help="Hoogte-overdrijving van de bergen (standaard: 2.0)",
  )
  parser.add_argument(
      "--grootte",
      type=float,
      default=100.0,
      help="Totale lengte van het model in mm (standaard: 100.0)",
  )
  parser.add_argument(
      "--bodem",
      type=float,
      default=5.0,
      help="Dikte van de massieve bodemplaat (standaard: 5.0)",
  )
  parser.add_argument(
      "--marge",
      type=float,
      default=0.01,
      help="Extra ruimte rondom de route in graden (standaard: 0.01)",
  )

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
      bodem_dikte=args.bodem,
  )
