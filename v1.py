"""Mars surface explorer using NASA WMTS sources (MOLA + MRO)."""

from __future__ import annotations

import datetime as dt
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import io
import dash
import numpy as np
import plotly.graph_objects as go
import requests
from PIL import Image
import json

from dash import Dash, Input, Output, State, dcc, html
from dash.dependencies import ALL
from dash.exceptions import PreventUpdate
import dash_leaflet as dl

MOLA_TILE_TEMPLATE = (
    "https://trek.nasa.gov/tiles/Mars/EQ/"
    "Mars_MGS_MOLA_ClrShade_merge_global_463m/1.0.0/default/default028mm/{z}/{y}/{x}.jpg"
)
SEARCH_ITEMS_URL = "https://trek.nasa.gov/mars/TrekServices/ws/index/eq/searchItems"
LAYER_SERVICES_URL = "https://trek.nasa.gov/mars/TrekServices/ws/index/getLayerServices"
ELEVATION_IDENTIFY_URL = (
    "https://trek.nasa.gov/mars/trekarcgis/rest/services/"
    "mola128_mola64_merge_90Nto90S_SimpleC_clon0/ImageServer/identify"
)
MAX_SEARCH_RESULTS = 8
DEFAULT_INSTRUMENT = "CTX"
USER_AGENT = "MarsExplorer/1.0 (NASA Space Apps prototype)"

DEFAULT_TERRAIN_PATCH_DEGREES = 4.0
TERRAIN_PIXEL_RESOLUTION = 160
MOLA_NODATA_THRESHOLD = -1e20

REQUEST_HEADERS = {"User-Agent": USER_AGENT}

REFERENCE_FEATURES: List[Dict[str, Any]] = [
    {
        "name": "Olympus Mons",
        "lat": 18.65,
        "lon": -133.8,
        "notes": "Reference peak for scale.",
        "time": "Global",
        "source": "reference",
        "dataset": "MOLA"
    },
    {
        "name": "Valles Marineris",
        "lat": -14.0,
        "lon": -65.0,
        "notes": "Canyon network stretching nearly 4,000 km.",
        "time": "Global",
        "source": "reference",
        "dataset": "MOLA"
    },
    {
        "name": "Gale Crater",
        "lat": -5.4,
        "lon": 137.8,
        "notes": "Curiosity rover landing site.",
        "time": "Global",
        "source": "reference",
        "dataset": "MOLA"
    },
]


def http_get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Fetch JSON with shared headers and error handling."""
    response = requests.get(url, params=params, timeout=20, headers=REQUEST_HEADERS)
    response.raise_for_status()
    return response.json()


def safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_dynamic_callback_index(expected_type: str) -> Optional[int]:
    """Return the triggered index for a Dash pattern-matching callback."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return None
    prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if not prop_id:
        return None
    try:
        trigger = json.loads(prop_id)
    except (TypeError, json.JSONDecodeError):
        return None
    if trigger.get("type") != expected_type:
        return None
    return trigger.get("index")


@lru_cache(maxsize=128)
def fetch_wmts_metadata(endpoint: str) -> Optional[Dict[str, Any]]:
    """Retrieve WMTS capabilities and return template plus zoom metadata."""
    import xml.etree.ElementTree as ET

    url = endpoint.rstrip("/") + "/1.0.0/WMTSCapabilities.xml"
    try:
        response = requests.get(url, timeout=20, headers=REQUEST_HEADERS)
        response.raise_for_status()
    except requests.RequestException:
        return None

    try:
        xml_root = ET.fromstring(response.content)
    except ET.ParseError:
        return None

    ns = {"wmts": "http://www.opengis.net/wmts/1.0", "ows": "http://www.opengis.net/ows/1.1"}
    resource = xml_root.find(".//wmts:ResourceURL", ns)
    tile_matrices = xml_root.findall(".//wmts:TileMatrix", ns)
    fmt = xml_root.find(".//wmts:Layer/wmts:Format", ns)

    if resource is None or fmt is None or not tile_matrices:
        return None

    template = resource.attrib.get("template")
    if not template:
        return None

    zoom_levels: List[int] = []
    for tm in tile_matrices:
        identifier = tm.find("ows:Identifier", ns)
        if identifier is None:
            continue
        try:
            zoom_levels.append(int(identifier.text))
        except (TypeError, ValueError):
            continue

    if not zoom_levels:
        return None

    return {
        "template": template,
        "format": fmt.text or "",
        "min_zoom": min(zoom_levels),
        "max_zoom": max(zoom_levels),
    }


def build_tile_template(endpoint: str) -> Optional[Dict[str, Any]]:
    """Convert WMTS template into a leaflet-friendly tile URL."""
    meta = fetch_wmts_metadata(endpoint)
    if not meta:
        return None

    template = meta["template"]
    template = template.replace("{Style}", "default")
    template = template.replace("{TileMatrixSet}", "default028mm")
    template = template.replace("{TileMatrix}", "{z}")
    template = template.replace("{TileRow}", "{y}")
    template = template.replace("{TileCol}", "{x}")
    template = template.replace('1.0.0//', '1.0.0/')
    return {
        "url": template,
        "format": meta["format"],
        "min_zoom": meta["min_zoom"],
        "max_zoom": meta["max_zoom"],
    }


def parse_date_label(value: Optional[str]) -> str:
    if not value:
        return "Unknown"
    try:
        dt_obj = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt_obj.strftime("%Y-%m-%d")
    except ValueError:
        return value


def parse_bbox(bbox_str: Optional[str]) -> Optional[List[List[float]]]:
    if not bbox_str:
        return None
    parts = [safe_float(part) for part in bbox_str.split(",")]
    if len(parts) != 4 or any(p is None for p in parts):
        return None
    min_lon, min_lat, max_lon, max_lat = [p for p in parts]  # type: ignore[misc]
    return [[min_lat, min_lon], [max_lat, max_lon]]


def clamp_bounds(bounds: Optional[Sequence[Sequence[float]]]) -> Optional[Tuple[float, float, float, float]]:
    if not bounds or len(bounds) != 2:
        return None
    sw, ne = bounds
    if len(sw) != 2 or len(ne) != 2:
        return None
    min_lat = max(-90.0, min(sw[0], ne[0]))
    max_lat = min(90.0, max(sw[0], ne[0]))
    min_lon = max(-180.0, min(sw[1], ne[1]))
    max_lon = min(180.0, max(sw[1], ne[1]))
    if min_lat >= max_lat or min_lon >= max_lon:
        return None
    return (min_lon, min_lat, max_lon, max_lat)


def search_wmts_products(bounds: Optional[Sequence[Sequence[float]]], instrument: str) -> List[Dict[str, Any]]:
    """Query Trek search service for WMTS mosaics intersecting the viewport."""
    clamped = clamp_bounds(bounds)
    params: Dict[str, Any] = {
        "serviceType": "Mosaic",
        "shape": "",
        "point": "",
        "radius": "",
        "key": "*",
        "proj": "",
        "start": 0,
        "rows": MAX_SEARCH_RESULTS,
        "facetKeys": "instrument",
        "facetValues": instrument,
    }
    if clamped:
        bbox = f"{clamped[0]},{clamped[1]},{clamped[2]},{clamped[3]}"
        params["bbox"] = bbox
    data = http_get_json(SEARCH_ITEMS_URL, params=params)
    docs = data.get("response", {}).get("docs", [])
    return docs


def resolve_wmts_layers(docs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Resolve WMTS endpoints and tile templates for the provided docs."""
    resolved: List[Dict[str, Any]] = []
    for doc in docs:
        uuid = doc.get("item_UUID")
        if not uuid:
            continue
        try:
            service_data = http_get_json(LAYER_SERVICES_URL, params={"uuid": uuid})
        except requests.RequestException:
            continue
        service_docs = service_data.get("response", {}).get("docs", [])
        wmts_service = next(
            (
                service
                for service in service_docs
                if service.get("serviceType", "").lower() == "mosaic"
                and service.get("protocol", "").upper() == "WMTS"
            ),
            None,
        )
        if not wmts_service:
            continue
        endpoint = wmts_service.get("endPoint")
        if not endpoint:
            continue
        tile_meta = build_tile_template(endpoint)
        if not tile_meta:
            continue

        resolved.append(
            {
                "uuid": uuid,
                "title": doc.get("title", "Unnamed mosaic"),
                "date": parse_date_label(doc.get("data_created_date")),
                "instrument": doc.get("instrument", ""),
                "description": doc.get("description", ""),
                "endpoint": endpoint,
                "tile_url": tile_meta["url"],
                "format": tile_meta["format"],
                "min_zoom": tile_meta["min_zoom"],
                "max_zoom": tile_meta["max_zoom"],
                "bbox": parse_bbox(doc.get("bbox")),
            }
        )
    resolved.sort(key=lambda entry: entry.get("date", ""))
    return resolved


def fetch_mola_elevation(lat: float, lon: float) -> Optional[float]:
    params = {
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "sr": 104905,
        "returnGeometry": "false",
        "f": "json",
    }
    try:
        data = http_get_json(ELEVATION_IDENTIFY_URL, params=params)
    except requests.RequestException:
        return None
    value = safe_float(data.get("value"))
    if value is None:
        return None
    return value


def clean_mola_array(array: np.ndarray) -> np.ndarray:
    cleaned = array.astype(float)
    cleaned[(cleaned < MOLA_NODATA_THRESHOLD) | (cleaned > 1e6)] = np.nan
    return cleaned


def fetch_mola_patch(lat: float, lon: float, size_deg: float, pixels: int) -> Dict[str, Any]:
    half = max(size_deg, 0.2) / 2.0
    min_lat = max(-90.0, lat - half)
    max_lat = min(90.0, lat + half)
    min_lon = max(-180.0, lon - half)
    max_lon = min(180.0, lon + half)
    bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"
    params = {
        "bbox": bbox,
        "bboxSR": 104905,
        "size": f"{pixels},{pixels}",
        "format": "tiff",
        "pixelType": "F32",
        "f": "image",
    }
    try:
        response = requests.get(
            ELEVATION_IDENTIFY_URL.replace('/identify', '/exportImage'),
            params=params,
            timeout=60,
            headers=REQUEST_HEADERS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to export MOLA patch: {exc}") from exc
    try:
        image = Image.open(io.BytesIO(response.content))
        elevation = clean_mola_array(np.array(image))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to decode MOLA raster: {exc}") from exc
    latitudes = np.linspace(max_lat, min_lat, elevation.shape[0])
    longitudes = np.linspace(min_lon, max_lon, elevation.shape[1])
    return {
        "latitudes": latitudes.tolist(),
        "longitudes": longitudes.tolist(),
        "elevations": elevation.tolist(),
        "center": [lat, lon],
    }


def build_empty_terrain_figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        title={"text": "Select a point to load terrain", "x": 0.5},
    )
    return fig


def build_terrain_figure(data: Dict[str, Any], exaggeration: float) -> go.Figure:
    latitudes = np.array(data.get("latitudes", []), dtype=float)
    longitudes = np.array(data.get("longitudes", []), dtype=float)
    elevations = np.array(data.get("elevations", []), dtype=float)
    if latitudes.size == 0 or longitudes.size == 0 or elevations.size == 0:
        return build_empty_terrain_figure()
    grid_lon, grid_lat = np.meshgrid(longitudes, latitudes)
    scaled = elevations * float(exaggeration)
    finite_mask = np.isfinite(elevations)
    if np.any(finite_mask):
        cmin = float(np.nanmin(elevations[finite_mask]))
        cmax = float(np.nanmax(elevations[finite_mask]))
    else:
        cmin, cmax = -8000.0, 25000.0
    surface = go.Surface(
        x=grid_lon,
        y=grid_lat,
        z=scaled,
        surfacecolor=elevations,
        colorscale="Cividis",
        cmin=cmin,
        cmax=cmax,
        colorbar={"title": "Elevation (m)"},
        hovertemplate="Lon %{x:.2f} deg<br>Lat %{y:.2f} deg<br>Elevation %{surfacecolor:.0f} m<extra></extra>",
    )
    fig = go.Figure(data=[surface])
    fig.update_layout(
        template="plotly_dark",
        scene={
            "xaxis": {"title": "Longitude (deg E)"},
            "yaxis": {"title": "Latitude (deg N)"},
            "zaxis": {"title": "Elevation (m)"},
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        title={"text": "MOLA terrain preview", "x": 0.5},
    )
    return fig


def build_probe_message(probe: Optional[Dict[str, Any]]) -> Any:
    if not probe:
        return html.Span("Click the map to sample elevation.", style={"opacity": 0.7})
    lat = safe_float(probe.get("lat")) if isinstance(probe, dict) else None
    lon = safe_float(probe.get("lon")) if isinstance(probe, dict) else None
    elevation = safe_float(probe.get("elevation")) if isinstance(probe, dict) else None
    if lat is None or lon is None:
        return html.Span("Click the map to sample elevation.", style={"opacity": 0.7})
    children: List[Any] = [html.Span(f"Lat {lat:.3f} deg, Lon {lon:.3f} deg")]
    if elevation is not None:
        children.extend([html.Br(), html.Span(f"MOLA elevation: {elevation:.0f} m")])
    return html.Div(children)


def build_feature_markers(features: Sequence[Dict[str, Any]]) -> List[dl.Marker]:
    markers: List[dl.Marker] = []
    for feature in features:
        lat = safe_float(feature.get("lat"))
        lon = safe_float(feature.get("lon"))
        if lat is None or lon is None:
            continue
        popup = html.Div(
            [
                html.H4(feature.get("name", "Feature")),
                html.P(feature.get("notes", "")),
                html.P(f"Dataset: {feature.get('dataset', 'N/A')}", style={"marginBottom": "0.25rem"}),
                html.P(f"Observed: {feature.get('time', 'All')}", style={"marginBottom": "0"}),
            ]
        )
        markers.append(
            dl.Marker(
                position=[lat, lon],
                children=[
                    dl.Tooltip(feature.get("name", "Feature")),
                    dl.Popup(popup),
                ],
            )
        )
    return markers


def build_observation_cards(observations: Sequence[Dict[str, Any]]) -> List[html.Div]:
    cards: List[html.Div] = []
    for idx, observation in enumerate(observations):
        cards.append(
            html.Div(
                [
                    html.Strong(f"{idx}: {observation['title']}", style={"display": "block"}),
                    html.Span(
                        f"Observed {observation['date']} ({observation.get('instrument', '')})",
                        style={"fontSize": "0.85rem", "opacity": 0.8},
                    ),
                    html.Span(
                        observation.get("endpoint", ""),
                        style={"fontSize": "0.7rem", "opacity": 0.6, "display": "block", "marginTop": "0.25rem"},
                    ),
                ],
                style={
                    "backgroundColor": "#14213d",
                    "border": "1px solid #1e293b",
                    "borderRadius": "0.5rem",
                    "padding": "0.6rem",
                },
            )
        )
    return cards


app = Dash(__name__)
app.title = "Martian Surface Explorer"

app.layout = html.Div(
    [
        html.H1("Martian Surface Explorer"),
        html.P(
            "Stream NASA MOLA elevation tiles with MRO overlays, explore multiple resolutions, and annotate points of interest.",
            style={"maxWidth": "70ch", "opacity": 0.85},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Observation instrument"),
                        dcc.Dropdown(
                            id="instrument-dropdown",
                            options=[
                                {"label": "MRO Context Camera (CTX)", "value": "CTX"},
                                {"label": "MRO HiRISE", "value": "HiRISE"},
                                {"label": "Global (no overlay)", "value": "NONE"},
                            ],
                            value=DEFAULT_INSTRUMENT,
                            clearable=False,
                        ),
                        html.Button(
                            "Search overlays in view",
                            id="search-overlay-btn",
                            n_clicks=0,
                            style={"marginTop": "0.5rem", "width": "100%"},
                        ),
                        html.Div(id="overlay-status", style={"marginTop": "0.5rem", "minHeight": "1.5rem"}),
                        html.Label("Observation timeline", style={"marginTop": "1rem"}),
                        dcc.Slider(
                            id="observation-slider",
                            min=0,
                            max=0,
                            step=1,
                            value=0,
                            marks={},
                            tooltip={"placement": "bottom"},
                            disabled=True,
                        ),
                        html.Label("Overlay opacity", style={"marginTop": "1rem"}),
                        dcc.Slider(
                            id="overlay-opacity",
                            min=0.0,
                            max=1.0,
                            step=0.05,
                            value=0.75,
                            marks={0.0: "0", 0.5: "0.5", 1.0: "1"},
                        ),
                        html.Div(
                            [
                                html.H3("Clicked point"),
                                html.Div(id="click-readout", style={"minHeight": "3.5rem"}),
                            ],
                            style={"marginTop": "1.2rem"},
                        ),
                        html.Div(
                            [
                                html.H3("Current overlay"),
                                html.Div(id="observation-summary", style={"minHeight": "4.5rem"}),
                            ],
                            style={"marginTop": "1rem"},
                        ),
                        html.Div(
                            [
                                html.H3("Overlay candidates"),
                                html.Div(id="observation-cards", style={"display": "grid", "gap": "0.4rem"}),
                            ],
                            style={"marginTop": "1rem"},
                        ),
                    ],
                    style={"flex": "1", "minWidth": "260px", "maxWidth": "320px"},
                ),
                html.Div(
                    [
                        dl.Map(
                            id="mars-map",
                            center=[0, 0],
                            zoom=3,
                            eventHandlers={"click": {"latlng": True}},
                            children=[
                                dl.TileLayer(id="base-layer", url=MOLA_TILE_TEMPLATE, maxZoom=12),
                                dl.TileLayer(id="overlay-layer", opacity=0.75, url='', minZoom=0, maxZoom=18),
                                dl.Marker(
                                    id="probe-marker",
                                    position=[0.0, 0.0],
                                    draggable=False,
                                    children=[dl.Tooltip("Click the map to position")],
                                ),
                                dl.LayerGroup(id="feature-layer"),
                            ],
                            style={"height": "70vh", "width": "100%", "cursor": "crosshair"},
                            zoomControl=True,
                            preferCanvas=True,
                        ),
                        html.Div(
                            [
                                html.H3("3D terrain preview", style={"marginTop": "1rem"}),
                                html.Div(
                                    "Click the map or drag the marker; coordinates below update automatically.",
                                    id="terrain-status",
                                    style={"minHeight": "1.4rem", "marginBottom": "0.5rem"},
                                ),
                                html.Div(
                                    [
                                        html.Label("Latitude (deg N)"),
                                        dcc.Input(
                                            id="terrain-lat-input",
                                            type="number",
                                            min=-90,
                                            max=90,
                                            step=0.05,
                                            placeholder="-90 to 90",
                                            style={"width": "100%"},
                                        ),
                                        html.Label("Longitude (deg E)"),
                                        dcc.Input(
                                            id="terrain-lon-input",
                                            type="number",
                                            min=-180,
                                            max=180,
                                            step=0.05,
                                            placeholder="-180 to 180",
                                            style={"width": "100%"},
                                        ),
                                        html.Div(
                                            [
                                                html.Button("Load terrain patch", id="terrain-load-btn", n_clicks=0, style={"flex": "1"}),
                                                html.Button("Drop marker at inputs", id="terrain-place-btn", n_clicks=0, style={"flex": "1", "marginLeft": "0.5rem"}),
                                            ],
                                            style={"display": "flex", "marginTop": "0.5rem"},
                                        ),
                                    ],
                                    style={"display": "grid", "gap": "0.35rem"},
                                ),
                                dcc.Graph(
                                    id="terrain-graph",
                                    figure=build_empty_terrain_figure(),
                                    style={"height": "52vh"},
                                    config={"displaylogo": False},
                                ),
                                html.Label("Patch width (degrees)", style={"marginTop": "0.5rem"}),
                                dcc.Slider(
                                    id="terrain-size-slider",
                                    min=1.0,
                                    max=10.0,
                                    step=0.5,
                                    value=DEFAULT_TERRAIN_PATCH_DEGREES,
                                    marks={1.0: "1 deg", 4.0: "4 deg", 7.0: "7 deg", 10.0: "10 deg"},
                                ),
                                html.Label("Vertical exaggeration", style={"marginTop": "0.5rem"}),
                                dcc.Slider(
                                    id="terrain-exaggeration-slider",
                                    min=0.5,
                                    max=4.0,
                                    step=0.1,
                                    value=2.0,
                                    marks={0.5: "0.5x", 1.0: "1x", 2.0: "2x", 3.0: "3x", 4.0: "4x"},
                                ),
                            ],
                            style={
                                "marginTop": "1rem",
                                "backgroundColor": "#111c34",
                                "padding": "1rem",
                                "borderRadius": "0.6rem",
                                "border": "1px solid #1e293b",
                            },
                        ),
                        html.Div(
                            [
                                html.H3("Add a feature tag"),
                                html.Label("Name"),
                                dcc.Input(id="feature-name", type="text", placeholder="Dust storm front", style={"width": "100%"}),
                                html.Label("Latitude (deg N)", style={"marginTop": "0.5rem"}),
                                dcc.Input(
                                    id="feature-lat",
                                    type="number",
                                    min=-90,
                                    max=90,
                                    step=0.1,
                                    value=0.0,
                                    style={"width": "100%"},
                                    readOnly=True,
                                ),
                                html.Label("Longitude (deg E)", style={"marginTop": "0.5rem"}),
                                dcc.Input(
                                    id="feature-lon",
                                    type="number",
                                    min=-180,
                                    max=180,
                                    step=0.1,
                                    value=0.0,
                                    style={"width": "100%"},
                                    readOnly=True,
                                ),
                                html.Label("Notes", style={"marginTop": "0.5rem"}),
                                dcc.Textarea(
                                    id="feature-notes",
                                    placeholder="Context for this observation",
                                    style={"width": "100%", "height": "4.5rem"},
                                ),
                                html.Button("Save feature", id="save-feature-btn", n_clicks=0, style={"marginTop": "0.75rem"}),
                                html.Div(id="save-status", style={"marginTop": "0.5rem", "minHeight": "1.5rem"}),
                            ],
                            style={
                                "marginTop": "1rem",
                                "backgroundColor": "#111c34",
                                "padding": "1rem",
                                "borderRadius": "0.6rem",
                                "border": "1px solid #1e293b",
                            },
                        ),
                        html.Div(
                            [
                                html.H3("Tracked features"),
                                html.Div(id="feature-list", style={"display": "grid", "gap": "0.75rem", "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))"}),
                            ],
                            style={"marginTop": "1.5rem"},
                        ),
                    ],
                    style={"flex": "2", "minWidth": "480px"},
                ),
            ],
            style={"display": "flex", "gap": "1.5rem", "flexWrap": "wrap"},
        ),
        dcc.Store(id="overlay-catalog-store", storage_type="memory"),
        dcc.Store(id="probe-store", storage_type="memory"),
        dcc.Store(id="terrain-store", storage_type="memory"),
        dcc.Store(id="feature-edit-index", storage_type="memory"),
        dcc.Store(id="feature-store", storage_type="memory", data=REFERENCE_FEATURES),
    ],
    style={
        "maxWidth": "1200px",
        "margin": "0 auto",
        "padding": "1.5rem",
        "backgroundColor": "#0f172a",
        "color": "#e2e8f0",
        "fontFamily": "Segoe UI, Roboto, sans-serif",
    },
)


@app.callback(
    Output("overlay-catalog-store", "data"),
    Output("overlay-status", "children"),
    Output("observation-slider", "max"),
    Output("observation-slider", "marks"),
    Output("observation-slider", "value"),
    Output("observation-slider", "disabled"),
    Output("observation-cards", "children"),
    Input("search-overlay-btn", "n_clicks"),
    State("instrument-dropdown", "value"),
    State("mars-map", "bounds"),
)
def refresh_overlay_catalog(n_clicks: int, instrument: str, bounds: Any):
    if not n_clicks or instrument == "NONE":
        message = html.Span("No overlay requested; showing global MOLA.", style={"opacity": 0.7})
        return [], message, 0, {}, 0, True, []

    try:
        docs = search_wmts_products(bounds, instrument)
        overlays = resolve_wmts_layers(docs)
    except requests.RequestException as exc:
        message = html.Span(f"Overlay search failed: {exc}", style={"color": "#f87171"})
        return dash.no_update, message, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    if not overlays:
        message = html.Span("No WMTS products found in this view.", style={"color": "#fbbf24"})
        return [], message, 0, {}, 0, True, []

    marks = {index: overlays[index]["date"] for index in range(len(overlays))}
    cards = build_observation_cards(overlays)
    message = html.Span(f"Loaded {len(overlays)} overlay option(s).", style={"color": "#34d399"})
    return overlays, message, len(overlays) - 1, marks, len(overlays) - 1, False, cards


@app.callback(
    Output("overlay-layer", "url"),
    Output("overlay-layer", "opacity"),
    Output("overlay-layer", "minZoom"),
    Output("overlay-layer", "maxZoom"),
    Output("observation-summary", "children"),
    Input("observation-slider", "value"),
    Input("overlay-opacity", "value"),
    State("overlay-catalog-store", "data"),
    State("instrument-dropdown", "value"),
)
def update_overlay_layer(index: int, opacity: float, catalog: Sequence[Dict[str, Any]], instrument: str):
    if instrument == "NONE" or not catalog:
        summary = html.Div(
            [
                html.Strong("Global MOLA hillshade"),
                html.Span("NASA MGS MOLA global hillshade at 463 m/pixel", style={"display": "block", "opacity": 0.7}),
            ]
        )
        return MOLA_TILE_TEMPLATE, 0.0, dash.no_update, dash.no_update, summary

    if index is None or index < 0 or index >= len(catalog):
        raise PreventUpdate

    observation = catalog[index]
    summary = html.Div(
        [
            html.Strong(observation.get("title", "Selected overlay")),
            html.Span(
                f"Observed {observation.get('date', 'Unknown')} - {observation.get('instrument', '')}",
                style={"display": "block", "opacity": 0.75},
            ),
            html.Span(observation.get("endpoint", ""), style={"fontSize": "0.75rem", "opacity": 0.6}),
        ]
    )
    tile_url = observation.get("tile_url")
    min_zoom = observation.get("min_zoom")
    max_zoom = observation.get("max_zoom")
    if not tile_url:
        tile_url = MOLA_TILE_TEMPLATE
        opacity = 0.0
    if min_zoom is None or max_zoom is None or min_zoom >= max_zoom:
        min_zoom = 0
        max_zoom = 22
    return (
        tile_url,
        opacity,
        min_zoom,
        max_zoom,
        summary,
    )


@app.callback(
    Output("probe-store", "data", allow_duplicate=True),
    Output("terrain-status", "children", allow_duplicate=True),
    Input("mars-map", "clickData"),
    Input("terrain-place-btn", "n_clicks"),
    State("terrain-lat-input", "value"),
    State("terrain-lon-input", "value"),
    State("probe-store", "data"),
    prevent_initial_call=True,
)
def handle_probe_source(
    click_data: Optional[Dict[str, Any]],
    place_clicks: Optional[int],
    lat_value: Any,
    lon_value: Any,
    current_probe: Optional[Dict[str, Any]],
):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger == "mars-map":
        latlng = (click_data or {}).get("latlng") if isinstance(click_data, dict) else None
        if not latlng:
            raise PreventUpdate
        try:
            lat = float(latlng.get("lat"))
            lon = float(latlng.get("lng"))
        except (TypeError, ValueError):
            raise PreventUpdate
        elevation = fetch_mola_elevation(lat, lon)
        status = html.Span(
            "Point selected on map. Adjust inputs if needed, then load terrain.",
            style={"color": "#60a5fa"},
        )
        return {"lat": lat, "lon": lon, "elevation": elevation}, status

    if trigger == "terrain-place-btn":
        if not place_clicks:
            raise PreventUpdate
        lat = safe_float(lat_value)
        lon = safe_float(lon_value)
        if lat is None or lon is None:
            status = html.Span(
                "Provide valid latitude and longitude to drop the marker.",
                style={"color": "#f87171"},
            )
            return dash.no_update, status
        lat = float(lat)
        lon = float(lon)
        elevation = fetch_mola_elevation(lat, lon)
        status = html.Span(
            "Marker placed using manual coordinates.",
            style={"color": "#34d399"},
        )
        return {"lat": lat, "lon": lon, "elevation": elevation}, status

    raise PreventUpdate


@app.callback(
    Output("probe-marker", "position", allow_duplicate=True),
    Output("terrain-lat-input", "value", allow_duplicate=True),
    Output("terrain-lon-input", "value", allow_duplicate=True),
    Output("feature-lat", "value", allow_duplicate=True),
    Output("feature-lon", "value", allow_duplicate=True),
    Input("probe-store", "data"),
    prevent_initial_call=True,
)
def sync_probe_targets(probe: Optional[Dict[str, Any]]):
    if not probe:
        raise PreventUpdate
    lat = safe_float((probe or {}).get("lat"))
    lon = safe_float((probe or {}).get("lon"))
    if lat is None or lon is None:
        raise PreventUpdate
    return [lat, lon], float(lat), float(lon), float(lat), float(lon)


@app.callback(
    Output("terrain-store", "data"),
    Output("terrain-status", "children", allow_duplicate=True),
    Output("probe-store", "data", allow_duplicate=True),
    Input("terrain-load-btn", "n_clicks"),
    State("terrain-lat-input", "value"),
    State("terrain-lon-input", "value"),
    State("terrain-size-slider", "value"),
    State("probe-store", "data"),
    prevent_initial_call=True,
)
def load_terrain_patch(
    n_clicks: Optional[int],
    lat_value: Any,
    lon_value: Any,
    patch_size: Optional[float],
    probe: Optional[Dict[str, Any]],
):
    if not n_clicks:
        raise PreventUpdate
    lat = safe_float(lat_value)
    lon = safe_float(lon_value)
    if lat is None or lon is None:
        status = html.Span("Provide valid latitude and longitude before loading.", style={"color": "#f87171"})
        return dash.no_update, status, dash.no_update
    lat = float(lat)
    lon = float(lon)
    size = float(patch_size or DEFAULT_TERRAIN_PATCH_DEGREES)
    try:
        patch = fetch_mola_patch(lat, lon, size, TERRAIN_PIXEL_RESOLUTION)
    except RuntimeError as exc:
        status = html.Span(str(exc), style={"color": "#f87171"})
        return dash.no_update, status, dash.no_update
    latitudes = patch.get("latitudes", [])
    longitudes = patch.get("longitudes", [])
    status = html.Span(
        f"Loaded terrain patch {len(latitudes)}x{len(longitudes)} around lat {lat:.2f} deg, lon {lon:.2f} deg",
        style={"color": "#34d399"},
    )
    elevation = None
    if probe and isinstance(probe, dict):
        probe_lat = safe_float(probe.get("lat"))
        probe_lon = safe_float(probe.get("lon"))
        if (
            probe_lat is not None
            and probe_lon is not None
            and abs(probe_lat - lat) < 1e-6
            and abs(probe_lon - lon) < 1e-6
        ):
            elevation = safe_float(probe.get("elevation"))
    if elevation is None:
        elevation = fetch_mola_elevation(lat, lon)
    updated_probe = {"lat": lat, "lon": lon, "elevation": elevation}
    return patch, status, updated_probe


@app.callback(
    Output("click-readout", "children"),
    Input("probe-store", "data"),
)
def render_probe_readout(probe: Optional[Dict[str, Any]]):
    return build_probe_message(probe)




@app.callback(
    Output("terrain-graph", "figure"),
    Input("terrain-store", "data"),
    Input("terrain-exaggeration-slider", "value"),
)
def render_terrain(data: Optional[Dict[str, Any]], exaggeration: Optional[float]) -> go.Figure:
    if not data:
        return build_empty_terrain_figure()
    return build_terrain_figure(data, float(exaggeration or 1.0))


@app.callback(
    Output("feature-store", "data", allow_duplicate=True),
    Output("save-status", "children", allow_duplicate=True),
    Output("feature-edit-index", "data", allow_duplicate=True),
    Output("feature-name", "value", allow_duplicate=True),
    Output("feature-notes", "value", allow_duplicate=True),
    Input("save-feature-btn", "n_clicks"),
    State("feature-store", "data"),
    State("feature-name", "value"),
    State("feature-lat", "value"),
    State("feature-lon", "value"),
    State("feature-notes", "value"),
    State("observation-slider", "value"),
    State("overlay-catalog-store", "data"),
    State("probe-store", "data"),
    State("feature-edit-index", "data"),
    prevent_initial_call=True,
)
def save_feature(
    n_clicks: int,
    current_features: Sequence[Dict[str, Any]],
    name: str,
    lat: Any,
    lon: Any,
    notes: str,
    obs_index: Optional[int],
    catalog: Sequence[Dict[str, Any]],
    probe: Optional[Dict[str, Any]],
    edit_index: Optional[int],
):
    if not n_clicks:
        raise PreventUpdate

    errors: List[str] = []
    lat_val = safe_float(lat)
    lon_val = safe_float(lon)
    if lat_val is None and probe:
        lat_val = safe_float((probe or {}).get("lat"))
    if lon_val is None and probe:
        lon_val = safe_float((probe or {}).get("lon"))
    if not name or not str(name).strip():
        errors.append("Name required")
    if lat_val is None or not -90.0 <= lat_val <= 90.0:
        errors.append("Latitude must be between -90 and 90")
    if lon_val is None or not -180.0 <= lon_val <= 180.0:
        errors.append("Longitude must be between -180 and 180")

    if errors:
        message = html.Span(" | ".join(errors), style={"color": "#f87171"})
        return current_features or [], message, edit_index, name, notes

    observation_title = "MOLA global"
    observation_date = "Global"
    if catalog and obs_index is not None and 0 <= obs_index < len(catalog):
        obs = catalog[obs_index]
        observation_title = obs.get("title", observation_title)
        observation_date = obs.get("date", observation_date)

    feature_data = {
        "name": str(name).strip(),
        "lat": lat_val,
        "lon": lon_val,
        "notes": str(notes).strip() if notes else "",
        "time": observation_date,
        "source": "user",
        "dataset": observation_title,
    }
    if probe and isinstance(probe, dict):
        feature_data["elevation"] = probe.get("elevation")

    updated = list(current_features or [])
    next_edit_index = None
    if isinstance(edit_index, int) and 0 <= edit_index < len(updated):
        updated[edit_index] = feature_data
        message = html.Span(
            f"Updated feature '{feature_data['name']}' at {lat_val:.2f}N, {lon_val:.2f}E",
            style={"color": "#60a5fa"},
        )
    else:
        updated.append(feature_data)
        message = html.Span(
            f"Saved feature '{feature_data['name']}' at {lat_val:.2f}N, {lon_val:.2f}E",
            style={"color": "#34d399"},
        )
    return updated, message, next_edit_index, "", ""


@app.callback(
    Output("feature-layer", "children"),
    Output("feature-list", "children"),
    Input("feature-store", "data"),
)
def render_feature_layers(features: Sequence[Dict[str, Any]]):
    if not features:
        empty = html.Div("No features saved yet.", style={"opacity": 0.7})
        return [], empty
    markers = build_feature_markers(features)

    cards: List[html.Div] = []
    for idx, feature in enumerate(features):
        cards.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Strong(feature.get("name", "Feature")),
                            html.Span(
                                f"Lat {feature.get('lat', 0):.2f}, Lon {feature.get('lon', 0):.2f}",
                                style={"fontSize": "0.9rem", "opacity": 0.8},
                            ),
                            html.Span(
                                f"Observed: {feature.get('time', 'N/A')} - {feature.get('dataset', 'Dataset')}",
                                style={"fontSize": "0.8rem", "opacity": 0.7},
                            ),
                            html.Span(feature.get("notes", ""), style={"fontSize": "0.8rem"}),
                        ]
                    ),
                    html.Div(
                        [
                            html.Button(
                                "Edit",
                                id={"type": "feature-edit", "index": idx},
                                n_clicks=0,
                                style={"flex": "1", "padding": "0.35rem", "backgroundColor": "#2563eb", "color": "#e2e8f0", "border": "none", "borderRadius": "0.3rem"},
                            ),
                            html.Button(
                                "Delete",
                                id={"type": "feature-delete", "index": idx},
                                n_clicks=0,
                                style={"flex": "1", "padding": "0.35rem", "backgroundColor": "#dc2626", "color": "#f8fafc", "border": "none", "borderRadius": "0.3rem"},
                            ),
                        ],
                        style={"display": "flex", "gap": "0.5rem", "marginTop": "0.6rem"},
                    ),
                ],
                style={
                    "backgroundColor": "#111c34",
                    "border": "1px solid #1e293b",
                    "borderRadius": "0.5rem",
                    "padding": "0.75rem",
                },
            )
        )
    return markers, cards


@app.callback(
    Output("probe-store", "data", allow_duplicate=True),
    Output("feature-name", "value", allow_duplicate=True),
    Output("feature-notes", "value", allow_duplicate=True),
    Output("feature-edit-index", "data"),
    Output("terrain-status", "children", allow_duplicate=True),
    Input({"type": "feature-edit", "index": ALL}, "n_clicks"),
    State("feature-store", "data"),
    prevent_initial_call=True,
)
def on_feature_edit(n_clicks, features):
    index = get_dynamic_callback_index("feature-edit")
    if (
        features is None
        or not isinstance(index, int)
        or index < 0
        or index >= len(features)
    ):
        raise PreventUpdate
    feature = features[index]
    lat = float(feature.get("lat", 0.0))
    lon = float(feature.get("lon", 0.0))
    name = feature.get("name", "")
    notes = feature.get("notes", "")
    probe = {"lat": lat, "lon": lon, "elevation": feature.get("elevation")}
    status = html.Span(
        f"Editing feature '{name}'. Adjust details and press Save feature to update.",
        style={"color": "#38bdf8"},
    )
    return probe, name, notes, index, status


@app.callback(
    Output("feature-store", "data", allow_duplicate=True),
    Output("save-status", "children", allow_duplicate=True),
    Output("feature-edit-index", "data", allow_duplicate=True),
    Input({"type": "feature-delete", "index": ALL}, "n_clicks"),
    State("feature-store", "data"),
    State("feature-edit-index", "data"),
    prevent_initial_call=True,
)
def on_feature_delete(n_clicks, features, edit_index):
    index = get_dynamic_callback_index("feature-delete")
    if (
        features is None
        or not isinstance(index, int)
        or index < 0
        or index >= len(features)
    ):
        raise PreventUpdate
    updated = list(features)
    removed = updated.pop(index)
    message = html.Span(f"Deleted feature '{removed.get('name', 'Feature')}'.", style={"color": "#f97316"})
    new_edit_index = edit_index
    if isinstance(edit_index, int):
        if edit_index == index:
            new_edit_index = None
        elif edit_index > index:
            new_edit_index = edit_index - 1
    return updated, message, new_edit_index


if __name__ == "__main__":
    app.run(debug=True)


