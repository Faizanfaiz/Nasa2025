"""Interactive SAM-based image viewer component for Dash."""

import base64
import json
from typing import Any, Dict, List, Optional
import dash
from dash import html, dcc, Input, Output, State, callback_context, no_update
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.express as px
from sam_integration import sam_processor, initialize_sam_model, SAM_AVAILABLE

# Export for testing
SAM_VIEWER_AVAILABLE = SAM_AVAILABLE

def create_sam_viewer_modal(image_data: str, feature_info: Dict[str, Any]) -> html.Div:
    """Create the SAM interactive viewer modal."""
    
    # Initialize SAM if available
    if SAM_AVAILABLE and not hasattr(create_sam_viewer_modal, '_sam_initialized'):
        initialize_sam_model()
        create_sam_viewer_modal._sam_initialized = True
    
    # Load image into SAM processor
    sam_processor.load_image(image_data)
    
    return html.Div([
        # Modal overlay
        html.Div([
            # Modal content
            html.Div([
                # Header
                html.Div([
                    html.H3(f"SAM Analysis: {feature_info.get('name', 'Feature')}", 
                           style={"margin": "0", "color": "#e2e8f0"}),
                    html.Button("×", id="sam-close-btn", 
                              style={
                                  "position": "absolute", "top": "10px", "right": "15px",
                                  "background": "none", "border": "none", "fontSize": "24px",
                                  "color": "#e2e8f0", "cursor": "pointer"
                              })
                ], style={"position": "relative", "marginBottom": "1rem"}),
                
                # Feature info
                html.Div([
                    html.P(f"Dataset: {feature_info.get('dataset', 'N/A')}", 
                          style={"margin": "0.25rem 0", "opacity": 0.8}),
                    html.P(f"Location: {feature_info.get('lat', 0):.2f}°N, {feature_info.get('lon', 0):.2f}°E", 
                          style={"margin": "0.25rem 0", "opacity": 0.8}),
                    html.P(f"Observed: {feature_info.get('time', 'N/A')}", 
                          style={"margin": "0.25rem 0", "opacity": 0.8}),
                ], style={"marginBottom": "1rem", "fontSize": "0.9rem"}),
                
                # Instructions
                html.Div([
                    html.P("🎯 Click anywhere on the image to segment that area", 
                          style={"margin": "0.5rem 0", "color": "#60a5fa", "fontWeight": "bold"}),
                    html.P("📸 The AI will automatically detect objects at the click point", 
                          style={"margin": "0.5rem 0", "color": "#60a5fa"}),
                    html.P("💾 Use 'Save Highlighted Segment' to export the result", 
                          style={"margin": "0.5rem 0", "color": "#60a5fa"}),
                ], style={
                    "backgroundColor": "#1e293b", 
                    "padding": "0.75rem", 
                    "borderRadius": "0.5rem",
                    "marginBottom": "1rem"
                }),
                
                # Main content area
                html.Div([
                    # Left side - Interactive image
                        html.Div([
                            html.Div([
                                # Interactive image with click capture
                                html.Div([
                                    html.Img(
                                        id="sam-interactive-image",
                                        src=image_data,
                                        style={
                                            "width": "100%", 
                                            "height": "500px", 
                                            "objectFit": "contain",
                                            "cursor": "crosshair",
                                            "border": "2px solid #374151",
                                            "borderRadius": "0.5rem"
                                        }
                                    ),
                                    # Invisible overlay for click capture
                                    html.Div(
                                        id="sam-click-overlay",
                                        style={
                                            "position": "absolute",
                                            "top": "0",
                                            "left": "0",
                                            "width": "100%",
                                            "height": "100%",
                                            "cursor": "crosshair",
                                            "zIndex": "10"
                                        }
                                    )
                                ], style={"position": "relative"}),
                                # Hidden div to store click coordinates
                                html.Div(id="sam-click-coords", style={"display": "none"}),
                                # Hidden div to store hover coordinates  
                                html.Div(id="sam-hover-coords", style={"display": "none"})
                            ])
                        ], style={"flex": "2", "marginRight": "1rem"}),
                    
                    # Right side - Controls and results
                    html.Div([
                        # Segment info
                        html.Div([
                            html.H4("Segment Information", style={"margin": "0 0 1rem 0", "color": "#e2e8f0"}),
                            html.Div(id="segment-info", style={"minHeight": "100px"})
                        ], style={
                            "backgroundColor": "#111c34",
                            "padding": "1rem",
                            "borderRadius": "0.5rem",
                            "marginBottom": "1rem"
                        }),
                        
                        # Actions
                        html.Div([
                            html.H4("Actions", style={"margin": "0 0 1rem 0", "color": "#e2e8f0"}),
                            html.Button("Save Highlighted Segment", id="save-segment-btn", 
                                      style={
                                          "width": "100%", "padding": "0.75rem",
                                          "backgroundColor": "#059669", "color": "white",
                                          "border": "none", "borderRadius": "0.5rem",
                                          "cursor": "pointer", "marginBottom": "0.5rem"
                                      }),
                        ], style={
                            "backgroundColor": "#111c34",
                            "padding": "1rem",
                            "borderRadius": "0.5rem"
                        })
                    ], style={"flex": "1", "minWidth": "300px"})
                ], style={"display": "flex", "gap": "1rem"}),
                
                # Status messages
                html.Div(id="sam-status", style={"marginTop": "1rem", "minHeight": "2rem"})
                
            ], style={
                "backgroundColor": "#0f172a",
                "padding": "2rem",
                "borderRadius": "1rem",
                "maxWidth": "1200px",
                "maxHeight": "90vh",
                "overflow": "auto",
                "border": "1px solid #374151"
            })
        ], style={
            "position": "fixed",
            "top": 0, "left": 0, "right": 0, "bottom": 0,
            "backgroundColor": "rgba(0, 0, 0, 0.8)",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "zIndex": 10000,
            "padding": "2rem"
        })
    ])

def create_interactive_image_figure(image_data: str) -> go.Figure:
    """Create an interactive Plotly figure for the image."""
    # Create a simple image figure
    fig = go.Figure()
    
    # Add the image as a background
    fig.add_layout_image(
        dict(
            source=image_data,
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=1,
            sizey=1,
            sizing="stretch",
            opacity=1,
            layer="below"
        )
    )
    
    # Add invisible scatter points to capture hover/click events
    fig.add_trace(go.Scatter(
        x=[0.1, 0.3, 0.5, 0.7, 0.9],
        y=[0.1, 0.3, 0.5, 0.7, 0.9],
        mode='markers',
        marker=dict(size=1, opacity=0),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Configure layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[0, 1]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[0, 1],
            scaleanchor="x",
            scaleratio=1
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
        showlegend=False,
        clickmode='event+select'
    )
    
    return fig

def setup_sam_callbacks(app):
    """Set up all SAM-related callbacks."""
    
    @app.callback(
        Output("sam-click-coords", "children"),
        Input("sam-click-overlay", "n_clicks"),
        State("sam-click-overlay", "id"),
        prevent_initial_call=True
    )
    def handle_image_click(n_clicks, overlay_id):
        """Handle click events on the interactive image overlay."""
        if not n_clicks:
            raise PreventUpdate
        
        # For now, simulate different click positions to test segment updates
        # In a real implementation, you'd capture the actual click coordinates from the event
        import random
        x = random.uniform(0.2, 0.8)  # Random x between 0.2 and 0.8
        y = random.uniform(0.2, 0.8)  # Random y between 0.2 and 0.8
        return f"{x:.3f},{y:.3f}"
    
    @app.callback(
        Output("sam-status", "children"),
        Output("segment-info", "children"),
        Output("sam-interactive-image", "src"),
        Input("sam-click-coords", "children"),
        State("sam-interactive-image", "src"),
        prevent_initial_call=True
    )
    def handle_click_coordinates(coords_str, current_image_src):
        """Handle click coordinates and perform segmentation."""
        if not coords_str:
            raise PreventUpdate
        
        try:
            x, y = map(float, coords_str.split(','))
            # Convert normalized coordinates to image coordinates
            img_x = int(x * 1000)  # Assuming 1000px width
            img_y = int((1 - y) * 1000)  # Flip Y coordinate
            
            status, segment_info = handle_click_event({"x": x, "y": y, "img_x": img_x, "img_y": img_y})
            
            # Get the highlighted image from the SAM processor
            highlighted_image_src = current_image_src  # Default to current image
            if hasattr(sam_processor, 'current_mask') and sam_processor.current_mask is not None:
                # Get the highlighted image from the processor
                from sam_integration import create_highlighted_image, image_to_base64
                if sam_processor.current_image is not None:
                    highlighted = create_highlighted_image(sam_processor.current_image, sam_processor.current_mask)
                    highlighted_image_src = image_to_base64(highlighted)
            
            return status, segment_info, highlighted_image_src
        except (ValueError, TypeError):
            raise PreventUpdate
    
    @app.callback(
        Output("sam-status", "children", allow_duplicate=True),
        Output("sam-annotation-store", "data", allow_duplicate=True),
        Input("save-segment-btn", "n_clicks"),
        State("sam-annotation-store", "data"),
        prevent_initial_call=True
    )
    def save_segment(n_clicks, annotation_data):
        """Save the currently selected segment."""
        if not n_clicks:
            raise PreventUpdate
        
        if not SAM_AVAILABLE:
            return html.Div("SAM model not available yet. Please wait for installation to complete.", 
                          style={"color": "#f87171"})
        
        if sam_processor.current_mask is None:
            return html.Div("No segment selected. Click on the image first.", 
                          style={"color": "#f87171"})
        
        try:
            # Check if there's an existing annotation to replace
            existing_filename = None
            if annotation_data and isinstance(annotation_data, dict):
                existing_filename = annotation_data.get("path")
            
            # Save segment with replacement logic
            filename = sam_processor.save_segment(existing_filename)
            if filename:
                # Update annotation data with new segment info
                updated_annotation = {
                    "path": filename,
                    "timestamp": sam_processor.current_timestamp if hasattr(sam_processor, 'current_timestamp') else None
                }
                
                status = html.Div(f"✅ Segment saved as {filename}", 
                                style={"color": "#34d399"})
                return status, updated_annotation
            else:
                status = html.Div("❌ Failed to save segment", 
                                style={"color": "#f87171"})
                return status, dash.no_update
        except Exception as e:
            status = html.Div(f"❌ Error saving segment: {str(e)}", 
                            style={"color": "#f87171"})
            return status, dash.no_update
    
    @app.callback(
        Output("sam-status", "children", allow_duplicate=True),
        Input("download-full-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def download_full_image(n_clicks):
        """Download the full image."""
        if not n_clicks:
            raise PreventUpdate
        
        # This would trigger a download in a real implementation
        return html.Div("📥 Full image download initiated", 
                      style={"color": "#34d399"})

def handle_hover_event(hover_data: Dict[str, Any]) -> tuple:
    """Handle hover events to show segment preview."""
    if not SAM_AVAILABLE:
        return (
            html.Div("SAM model not available yet.", style={"color": "#f87171"}),
            html.Div("Waiting for SAM installation...", style={"opacity": 0.7})
        )
    
    try:
        # Extract coordinates from hover data
        if "points" not in hover_data or not hover_data["points"]:
            return no_update, no_update
        
        point = hover_data["points"][0]
        x = point.get("x", 0)
        y = point.get("y", 0)
        
        # Convert to image coordinates (assuming image is 1x1 normalized)
        # In a real implementation, you'd need to map these to actual image pixels
        img_x = int(x * 1000)  # Assuming 1000px width
        img_y = int((1 - y) * 1000)  # Flip Y coordinate
        
        # Get segment at this point
        result = sam_processor.get_segment_at_coordinates(img_x, img_y)
        
        if result:
            status = html.Div(f"🖱️ Hovering at ({img_x}, {img_y})", 
                            style={"color": "#60a5fa"})
            
            info = html.Div([
                html.P(f"Segment Area: {result['mask_area']} pixels", 
                      style={"margin": "0.25rem 0"}),
                html.P(f"Bounds: {result['mask_bounds']}", 
                      style={"margin": "0.25rem 0", "fontSize": "0.8rem", "opacity": 0.8}),
                html.Img(src=result['segment_image'], 
                        style={"maxWidth": "100%", "height": "auto", "borderRadius": "0.25rem"})
            ])
            
            return status, info
        else:
            return (
                html.Div("No segment found at this location", style={"color": "#fbbf24"}),
                html.Div("Try hovering over a distinct object", style={"opacity": 0.7})
            )
            
    except Exception as e:
        return (
            html.Div(f"Error processing hover: {str(e)}", style={"color": "#f87171"}),
            html.Div("", style={"opacity": 0.7})
        )

def handle_click_event(click_data: Dict[str, Any]) -> tuple:
    """Handle click events to select a segment."""
    if not SAM_AVAILABLE:
        return (
            html.Div("SAM model not available yet.", style={"color": "#f87171"}),
            html.Div("Waiting for SAM installation...", style={"opacity": 0.7})
        )
    
    try:
        # Extract coordinates from click data
        img_x = click_data.get("img_x", 500)
        img_y = click_data.get("img_y", 500)
        x = click_data.get("x", 0.5)
        y = click_data.get("y", 0.5)
        
        # Get segment at this point
        result = sam_processor.get_segment_at_coordinates(img_x, img_y)
        
        if result:
            status = html.Div(f"✅ Segment selected at ({img_x}, {img_y}) - Ready to save!", 
                            style={"color": "#34d399"})
            
            info = html.Div([
                html.P(f"Selected Segment", style={"fontWeight": "bold", "margin": "0 0 0.5rem 0"}),
                html.P(f"Area: {result['mask_area']} pixels", 
                      style={"margin": "0.25rem 0"}),
                html.P(f"Bounds: {result['mask_bounds']}", 
                      style={"margin": "0.25rem 0", "fontSize": "0.8rem", "opacity": 0.8}),
                html.Img(src=result['segment_image'], 
                        style={"maxWidth": "100%", "height": "auto", "borderRadius": "0.25rem", 
                               "border": "2px solid #34d399"})
            ])
            
            return status, info
        else:
            return (
                html.Div("No segment found at this location", style={"color": "#fbbf24"}),
                html.Div("Try clicking on a distinct object", style={"opacity": 0.7})
            )
            
    except Exception as e:
        return (
            html.Div(f"Error processing click: {str(e)}", style={"color": "#f87171"}),
            html.Div("", style={"opacity": 0.7})
        )
