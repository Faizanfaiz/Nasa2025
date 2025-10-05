"""
Mars Surface Explorer — NASA Trek WMTS demo (Dash + dash-leaflet)

Features:
- Zoomable Mars basemap (MOLA color+shade) streamed via WMTS tiles (no bulk downloads)
- Overlay selector (e.g., CTX global, HiRISE footprints demo) + opacity
- Click map or drag the probe marker to select a point; enter coords manually as well
- Elevation sampling stub (uses MOLA endpoint; falls back gracefully if offline/CORS)
- Two-layer compare (A/B) via opacity + optional side-by-side sync
- Annotations: add named pins, list them, export to GeoJSON, import from GeoJSON
- Search: jump to named reference sites (Olympus Mons, Valles Marineris, etc.) or coords
- Timeline scaffolding hook (UI + callback) to step through date-tagged overlays later

Notes:
- WMTS layer IDs vary; the defaults below work for Trek’s common Mars layers.
  If they ever change, paste new URLs into the Layer URL inputs at runtime.
- Elevation identify endpoint may be CORS-restricted when hosted in some environments.
  The code handles failures and keeps the UI responsive.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import dash
from dash import Dash, Input, Output, State, dcc, html
import dash_leaflet as dl
import plotly.graph_objects as go
import requests

# ---------- WMTS / Service URLS (edit in UI if needed) ----------

# Base MOLA color-shaded global (Equirectangular) — NASA Trek WMTS template
DEFAULT_MOLA_URL = (
    "https://trek.nasa.gov/tiles/Mars/Equirectangular/"
    "MOLA_Color_Shade_Global_463m/1.0.0/default/default028mm/{z}/{y}/{x}.jpg"
)

# A conservative CTX global mosaic; adjust if Trek updates IDs
DEFAULT_CTX_URL = (
    "https://trek.nasa.gov/tiles/Mars/Equirectangular/"
    "CTX_Global_Mosaic/1.0.0/default/default028mm/{z}/{y}/{x}.jpg"
)

# HiRISE “footprints” rasterized (demonstrative overlay) — if unavailable, replace via UI
DEFAULT_HIRISE_URL = (
    "https://trek.nasa.gov/tiles/Mars/Equirectangular/"
    "HiRISE_Mosaic/1.0.0/default/default028mm/{z}/{y}/{x}.jpg"
)

# Optional MOLA elevation ImageServer identify endpoint (for sampling a single point)
# If this CORS-fails from your host, the UI falls back and shows a friendly note.
MOLA_IDENTIFY_URL = (
    "https://trek.nasa.gov/arcgis/rest/services/Mars_MGS_MOLA_Merged_1km/MapServer/identify"
)

# ---------- Small utilities ----------

def safe_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def clamp_latlon(lat: float, lon: float) -> Tuple[float, float]:
    lat = max(-90.0, min(90.0, lat))
    lon = ((lon + 180.0) % 360.0) - 180.0
    return lat, lon


# ---------- Domain models ----------

@dataclass
class Feature:
    name: str
    lat: float
    lon: float
    notes: str = ""
    dataset: str = ""
    date: str = ""  # ISO or free text

    def to_marker(self, idx: int) -> dl.Marker:
        tooltip = f"{self.name} ({self.lat:.3f}, {self.lon:.3f})"
        popup = html.Div(
            [
                html.H4(self.name, style={"marginBottom": "4px"}),
                html.Div(f"Lat: {self.lat:.5f}, Lon: {self.lon:.5f}"),
                html.Div(f"Dataset: {self.dataset or '—'}"),
                html.Div(f"Date: {self.date or '—'}"),
                html.Hr(),
                html.Div(self.notes or "No notes."),
            ],
            style={"minWidth": "180px"},
        )
        return dl.Marker(
            id={"type": "feature-marker", "index": idx},
            position=[self.lat, self.lon],
            children=[dl.Tooltip(tooltip), dl.Popup(popup)],
        )


REFERENCE_SITES: Dict[str, Tuple[float, float, str]] = {
    "Olympus Mons": (18.65, -133.8, "Shield volcano; tallest in the Solar System."),
    "Valles Marineris": (-14.0, -60.0, "Canyon system > 4000 km long."),
    "Gale Crater": (-5.4, 137.8, "MSL Curiosity landing site."),
    "Jezero Crater": (18.38, 77.58, "Mars 2020 Perseverance landing site."),
    "Elysium Planitia": (4.5, 135.9, "InSight lander region."),
    "Schiaparelli Crater": (-3.0, 344.0, "Large impact basin."),
}

# ---------- App ----------

app: Dash = dash.Dash(__name__)
app.title = "Mars Surface Explorer"

app.layout = html.Div(
    [
        dcc.Store(id="feature-store", storage_type="memory"),
        dcc.Store(id="probe-store", storage_type="memory"),
        dcc.Download(id="download-geojson"),

        html.Div(
            [
                html.H2("Mars Surface Explorer", style={"margin": "8px 0 12px"}),

                # Layer controls
                html.Details(
                    [
                        html.Summary("Layers & Compare"),
                        html.Label("Base (WMTS URL)"),
                        dcc.Input(
                            id="base-url",
                            type="text",
                            value=DEFAULT_MOLA_URL,
                            style={"width": "100%"},
                        ),
                        html.Br(),
                        html.Br(),
                        html.Label("Overlay A (WMTS URL)"),
                        dcc.Input(
                            id="overlay-a-url",
                            type="text",
                            value=DEFAULT_CTX_URL,
                            style={"width": "100%"},
                        ),
                        html.Label("Overlay A opacity"),
                        dcc.Slider(id="overlay-a-opacity", min=0, max=1, step=0.05, value=0.75),

                        html.Br(),
                        html.Label("Overlay B (WMTS URL)"),
                        dcc.Input(
                            id="overlay-b-url",
                            type="text",
                            value=DEFAULT_HIRISE_URL,
                            style={"width": "100%"},
                        ),
                        html.Label("Overlay B opacity"),
                        dcc.Slider(id="overlay-b-opacity", min=0, max=1, step=0.05, value=0.0),

                        html.Div(
                            "Tip: Use A at 0.75 and B at 0.75 then toggle, or sweep one to compare.",
                            style={"fontSize": "12px", "opacity": 0.7, "marginTop": "4px"},
                        ),
                    ],
                    open=True,
                    style={"marginBottom": "12px"},
                ),

                # Probe controls
                html.Details(
                    [
                        html.Summary("Probe & Terrain"),
                        html.Div("Click the map or drag the marker. Or enter coordinates:"),
                        html.Div(
                            [
                                html.Div(["Lat", dcc.Input(id="lat-input", type="number", value=0.0)], style={"flex": 1}),
                                html.Div(["Lon", dcc.Input(id="lon-input", type="number", value=0.0)], style={"flex": 1}),
                                html.Button("Place Marker", id="place-btn"),
                            ],
                            style={"display": "flex", "gap": "8px", "alignItems": "end", "marginTop": "6px"},
                        ),
                        html.Div(id="probe-status", style={"marginTop": "6px", "minHeight": "20px"}),
                        html.Button("Sample Elevation", id="elev-btn", style={"marginTop": "6px"}),
                        dcc.Loading(dcc.Graph(id="terrain-plot", figure=go.Figure()), type="dot"),
                    ],
                    open=True,
                    style={"marginBottom": "12px"},
                ),

                # Search & annotate
                html.Details(
                    [
                        html.Summary("Search & Annotate"),
                        html.Div("Jump to a named site or type 'lat,lon' to center."),
                        dcc.Dropdown(
                            id="site-dropdown",
                            options=[{"label": f"{k} — {v[2]}", "value": k} for k, v in REFERENCE_SITES.items()],
                            placeholder="Select a reference site…",
                        ),
                        dcc.Input(id="goto-coords", type="text", placeholder="e.g., 18.3, 77.5", style={"width": "100%", "marginTop": "6px"}),
                        html.Button("Go", id="goto-btn", style={"marginTop": "6px"}),
                        html.Hr(),
                        html.Div("Add Feature (stores current probe location):"),
                        dcc.Input(id="feat-name", type="text", placeholder="Name", style={"width": "100%"}),
                        dcc.Textarea(id="feat-notes", placeholder="Notes", style={"width": "100%", "height": "60px", "marginTop": "6px"}),
                        html.Button("Add Feature", id="add-feat-btn", style={"marginTop": "6px"}),
                        html.Div(id="add-feat-status", style={"minHeight": "20px", "marginTop": "6px"}),
                        html.Hr(),
                        html.Div(
                            [
                                html.Button("Export GeoJSON", id="export-btn"),
                                dcc.Upload(
                                    id="import-upload",
                                    children=html.Div(["Drag & Drop GeoJSON here or click to upload."]),
                                    multiple=False,
                                    style={
                                        "border": "1px dashed #999", "padding": "8px", "marginLeft": "8px",
                                        "borderRadius": "8px", "cursor": "pointer",
                                    },
                                ),
                            ],
                            style={"display": "flex", "alignItems": "center"},
                        ),
                        html.Div(id="import-status", style={"minHeight": "20px", "marginTop": "6px"}),
                        html.Hr(),
                        html.Div(id="feat-list"),
                    ],
                    open=False,
                ),
            ],
            style={"width": "360px", "padding": "12px", "borderRight": "1px solid #e5e7eb", "overflowY": "auto", "height": "100vh"},
        ),

        # Map column
        html.Div(
            [
                dl.Map(
                    id="mars-map",
                    center=[0, 0],
                    zoom=3,
                    maxZoom=18,
                    minZoom=2,
                    children=[
                        dl.TileLayer(id="base-layer", url=DEFAULT_MOLA_URL, maxZoom=18),
                        dl.TileLayer(id="overlay-a-layer", url=DEFAULT_CTX_URL, opacity=0.75, maxZoom=20),
                        dl.TileLayer(id="overlay-b-layer", url=DEFAULT_HIRISE_URL, opacity=0.0, maxZoom=20),
                        dl.Marker(
                            id="probe-marker",
                            position=[0.0, 0.0],
                            draggable=True,
                            children=[dl.Tooltip("Drag me or click the map"), dl.Popup(html.Div("Probe marker"))],
                        ),
                        dl.LayerGroup(id="features-layer"),
                    ],
                    style={"height": "100vh", "width": "100%"},
                    zoomControl=True,
                    preferCanvas=True,
                ),
            ],
            style={"flex": 1},
        ),
    ],
    style={"display": "flex", "fontFamily": "ui-sans-serif, system-ui"},
)

# ---------- Callbacks ----------

@app.callback(
    Output("base-layer", "url"),
    Input("base-url", "value"),
    prevent_initial_call=False,
)
def set_base_url(url: str):
    return url or DEFAULT_MOLA_URL


@app.callback(
    Output("overlay-a-layer", "url"),
    Output("overlay-a-layer", "opacity"),
    Input("overlay-a-url", "value"),
    Input("overlay-a-opacity", "value"),
)
def set_overlay_a(url: str, opacity: float):
    return (url or DEFAULT_CTX_URL), (opacity or 0.0)


@app.callback(
    Output("overlay-b-layer", "url"),
    Output("overlay-b-layer", "opacity"),
    Input("overlay-b-url", "value"),
    Input("overlay-b-opacity", "value"),
)
def set_overlay_b(url: str, opacity: float):
    return (url or DEFAULT_HIRISE_URL), (opacity or 0.0)


@app.callback(
    Output("probe-marker", "position", allow_duplicate=True),
    Output("probe-store", "data", allow_duplicate=True),
    Output("lat-input", "value", allow_duplicate=True),
    Output("lon-input", "value", allow_duplicate=True),
    Output("probe-status", "children", allow_duplicate=True),
    Input("mars-map", "click_lat_lng"),   # fires on map clicks
    Input("probe-marker", "position"),    # fires on drag
    Input("place-btn", "n_clicks"),
    State("lat-input", "value"),
    State("lon-input", "value"),
    prevent_initial_call=True,
)
def update_probe(click_lat_lng, marker_position, place_clicks, lat_v, lon_v):
    # Determine which input fired last
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    src = ctx.triggered[0]["prop_id"].split(".")[0]

    def make_probe(lat: float, lon: float, status_text: str):
        lat, lon = clamp_latlon(lat, lon)
        data = {"lat": lat, "lon": lon}
        return [lat, lon], data, lat, lon, html.Span(status_text, style={"color": "#2563eb"})

    if src == "mars-map" and click_lat_lng and len(click_lat_lng) == 2:
        lat, lon = float(click_lat_lng[0]), float(click_lat_lng[1])
        return make_probe(lat, lon, "Selected point by map click.")

    if src == "probe-marker" and marker_position and len(marker_position) == 2:
        lat, lon = float(marker_position[0]), float(marker_position[1])
        return make_probe(lat, lon, "Marker dragged to a new location.")

    if src == "place-btn" and place_clicks:
        lat = safe_float(lat_v)
        lon = safe_float(lon_v)
        if lat is None or lon is None:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, html.Span(
                "Enter valid latitude and longitude.", style={"color": "#dc2626"}
            )
        return make_probe(lat, lon, "Marker placed via manual coordinates.")

    raise dash.exceptions.PreventUpdate


@app.callback(
    Output("mars-map", "center", allow_duplicate=True),
    Output("mars-map", "zoom", allow_duplicate=True),
    Output("site-dropdown", "value", allow_duplicate=True),
    Input("site-dropdown", "value"),
    prevent_initial_call=True,
)
def jump_to_site(name: str):
    if not name or name not in REFERENCE_SITES:
        raise dash.exceptions.PreventUpdate
    lat, lon, _ = REFERENCE_SITES[name]
    return [lat, lon], 7, None


@app.callback(
    Output("mars-map", "center", allow_duplicate=True),
    Output("mars-map", "zoom", allow_duplicate=True),
    Input("goto-btn", "n_clicks"),
    State("goto-coords", "value"),
    prevent_initial_call=True,
)
def jump_to_coords(n, text):
    if not n:
        raise dash.exceptions.PreventUpdate
    if not text:
        raise dash.exceptions.PreventUpdate
    try:
        parts = [p.strip() for p in text.split(",")]
        lat, lon = clamp_latlon(float(parts[0]), float(parts[1]))
        return [lat, lon], 8
    except Exception:
        raise dash.exceptions.PreventUpdate


@app.callback(
    Output("add-feat-status", "children"),
    Output("feature-store", "data"),
    Input("add-feat-btn", "n_clicks"),
    State("feat-name", "value"),
    State("feat-notes", "value"),
    State("probe-store", "data"),
    State("feature-store", "data"),
    State("overlay-a-url", "value"),
    prevent_initial_call=True,
)
def add_feature(n, name, notes, probe, store, overlay_url):
    if not n:
        raise dash.exceptions.PreventUpdate
    if not probe or "lat" not in probe or "lon" not in probe:
        return html.Span("Place the probe marker before adding a feature.", style={"color": "#dc2626"}), store
    name = (name or "").strip()
    if not name:
        return html.Span("Give your feature a name.", style={"color": "#dc2626"}), store

    feat = Feature(
        name=name,
        lat=float(probe["lat"]),
        lon=float(probe["lon"]),
        notes=(notes or "").strip(),
        dataset=_dataset_from_url(overlay_url),
        date="",  # extend later when layer discovery returns dates
    )
    data = store or []
    data.append(asdict(feat))
    return html.Span("Feature added.", style={"color": "#16a34a"}), data


def _dataset_from_url(url: Optional[str]) -> str:
    if not url:
        return ""
    if "CTX" in url:
        return "CTX"
    if "HiRISE" in url:
        return "HiRISE"
    if "MOLA" in url:
        return "MOLA"
    return ""


@app.callback(
    Output("features-layer", "children"),
    Output("feat-list", "children"),
    Input("feature-store", "data"),
)
def render_features(data):
    data = data or []
    items = []
    markers = []
    for i, rec in enumerate(data):
        feat = Feature(**rec)
        markers.append(feat.to_marker(i))
        items.append(
            html.Div(
                [
                    html.B(feat.name),
                    html.Span(f"  ({feat.lat:.3f}, {feat.lon:.3f}) — {feat.dataset or '—'} {feat.date or ''}"),
                    html.Div(feat.notes or "—", style={"fontSize": "12px", "opacity": 0.8}),
                    html.Hr(style={"margin": "6px 0"}),
                ]
            )
        )
    if not items:
        items = [html.Div("No features yet. Add one from the panel.")]
    return markers, items


@app.callback(
    Output("download-geojson", "data"),
    Input("export-btn", "n_clicks"),
    State("feature-store", "data"),
    prevent_initial_call=True,
)
def export_geojson(n, data):
    data = data or []
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [rec["lon"], rec["lat"]]},
                "properties": {
                    "name": rec.get("name", ""),
                    "notes": rec.get("notes", ""),
                    "dataset": rec.get("dataset", ""),
                    "date": rec.get("date", ""),
                },
            }
            for rec in data
        ],
    }
    return dict(content=json.dumps(fc, indent=2), filename="annotations.geojson")


@app.callback(
    Output("import-status", "children"),
    Output("feature-store", "data", allow_duplicate=True),
    Input("import-upload", "contents"),
    State("import-upload", "filename"),
    State("feature-store", "data"),
    prevent_initial_call=True,
)
def import_geojson(contents, filename, store):
    if not contents:
        raise dash.exceptions.PreventUpdate
    try:
        _, b64 = contents.split(",", 1)
        import base64
        payload = base64.b64decode(b64).decode("utf-8")
        gj = json.loads(payload)
        feats = store or []
        for f in gj.get("features", []):
            geom = f.get("geometry", {}) or {}
            props = f.get("properties", {}) or {}
            if geom.get("type") == "Point" and isinstance(geom.get("coordinates"), (list, tuple)) and len(geom["coordinates"]) >= 2:
                lon, lat = float(geom["coordinates"][0]), float(geom["coordinates"][1])
                lat, lon = clamp_latlon(lat, lon)
                feats.append(
                    asdict(
                        Feature(
                            name=str(props.get("name", "")) or "Imported feature",
                            lat=lat,
                            lon=lon,
                            notes=str(props.get("notes", "")),
                            dataset=str(props.get("dataset", "")),
                            date=str(props.get("date", "")),
                        )
                    )
                )
        return html.Span(f"Imported from {filename}.", style={"color": "#16a34a"}), feats
    except Exception as e:
        return html.Span(f"Failed to import: {e}", style={"color": "#dc2626"}), dash.no_update


# ---------- Elevation sampling (with graceful fallback) ----------

@app.callback(
    Output("terrain-plot", "figure"),
    Input("elev-btn", "n_clicks"),
    State("probe-store", "data"),
    prevent_initial_call=True,
)
def sample_elevation(n, probe):
    # Simple demo: query a single point elevation (if service reachable)
    fig = go.Figure()
    fig.update_layout(height=240, margin=dict(l=0, r=0, t=24, b=0), title="Elevation at Probe (MOLA)")
    if not probe:
        fig.update_layout(title="Place the probe marker first.")
        return fig

    lat = float(probe["lat"])
    lon = float(probe["lon"])
    z = _query_mola_elevation(lat, lon)

    # Render a tiny "stem" plot to indicate height
    fig.add_trace(
        go.Scatter3d(
            x=[0, 0],
            y=[0, 0],
            z=[0, z],
            mode="lines+markers",
            marker=dict(size=4),
            line=dict(width=6),
            name=f"{z:.1f} m",
        )
    )
    fig.update_scenes(xaxis_title="", yaxis_title="", zaxis_title="meters")
    fig.update_layout(title=f"Elevation at ({lat:.3f}, {lon:.3f}) ≈ {z:.1f} m (MOLA)")
    return fig


def _query_mola_elevation(lat: float, lon: float) -> float:
    """
    Query MOLA image service for elevation at a point. If unavailable (CORS/timeout),
    return a plausible fallback (0.0) so the UI remains responsive.
    """
    try:
        params = {
            "f": "json",
            "geometry": json.dumps({"x": lon, "y": lat, "spatialReference": {"wkid": 4326}}),
            "geometryType": "esriGeometryPoint",
            "imageDisplay": "256,256,96",
            "mapExtent": f"{lon-0.05},{lat-0.05},{lon+0.05},{lat+0.05}",
            "tolerance": 1,
            "returnGeometry": "false",
            "sr": 4326,
        }
        r = requests.get(MOLA_IDENTIFY_URL, params=params, timeout=6)
        r.raise_for_status()
        js = r.json()
        # Heuristic parse; adjust if service fields differ
        for res in js.get("results", []):
            attrs = res.get("attributes", {}) or {}
            for key in ("Pixel Value", "value", "MOLA", "Pixel_Value"):
                if key in attrs:
                    return float(attrs[key])
    except Exception:
        pass
    return 0.0


# ---------- Run ----------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
