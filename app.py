import os
import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import time
import folium
from folium.plugins import HeatMap
from sklearn.ensemble import RandomForestClassifier

# Initialize the Dash application frameworks
app = dash.Dash(__name__, title="SIH26001 Resilient Dashboard")
application = app.server
server = app.server

# ==========================================
# 1. CORE MACHINE LEARNING ENGINE
# ==========================================
FEATURE_NAMES = ['slope_angle', 'rainfall_24h', 'soil_moisture', 'elevation', 'deforestation_idx']

def initialize_ml_pipeline():
    np.random.seed(42)
    samples = 1500
    X_mock = pd.DataFrame({
        'slope_angle': np.random.uniform(15, 60, samples),
        'rainfall_24h': np.random.uniform(10, 350, samples),
        'soil_moisture': np.random.uniform(20, 95, samples),
        'elevation': np.random.uniform(500, 3000, samples),
        'deforestation_idx': np.random.uniform(0.1, 0.9, samples)
    })
    y_mock = ((X_mock['rainfall_24h']/350 + X_mock['slope_angle']/60 + X_mock['soil_moisture']/95) > 1.2).astype(int)
    pipeline = RandomForestClassifier(n_estimators=50, random_state=42)
    pipeline.fit(X_mock, y_mock)
    return pipeline

ml_pipeline = initialize_ml_pipeline()

# Preset Monitor Region Coordinates and Topography Profiles
GEO_PROFILES = {
    'Cherrapunji': {'lat': 25.2702, 'lon': 91.7325, 'slope': 48.0, 'elevation': 1480, 'name': 'Cherrapunji (Meghalaya)'},
    'Aizawl': {'lat': 23.7271, 'lon': 92.7176, 'slope': 38.5, 'elevation': 1120, 'name': 'Aizawl (Mizoram)'},
    'Kohima': {'lat': 25.6751, 'lon': 94.1086, 'slope': 42.0, 'elevation': 1440, 'name': 'Kohima (Nagaland)'}
}

# ==========================================
# 2. UI DASHBOARD LAYOUT CONFIGURATION
# ==========================================
app.layout = html.Div(style={'fontFamily': 'Segoe UI, Arial, sans-serif', 'padding': '20px', 'backgroundColor': '#f4f6f9'}, children=[
    
    # Official Hackathon Header Banner
    html.Div(style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#2c3e50', 'borderRadius': '8px', 'color': 'white', 'marginBottom': '20px'}, children=[
        html.H1("🌋 MDoNER AI Landslide Early Warning Engine", style={'margin': '0 0 5px 0', 'fontSize': '26px'}),
        html.P("SIH26001 Multi-Source Fusion Platform • Live IMD Rainfall & ISRO Bhuvan Topography Engine Map", style={'margin': '0', 'opacity': '0.8', 'fontSize': '13px'})
    ]),
    
    # Left Column: Inputs & Media Upload Node
    html.Div(style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap'}, children=[
        html.Div(style={'flex': '1', 'minWidth': '340px', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px', 'boxShadow': '0 4px 10px rgba(0,0,0,0.05)'}, children=[
            html.H3("📡 Telemetry Input Control Node", style={'borderBottom': '2px solid #f1f3f5', 'paddingBottom': '8px', 'marginTop': '0', 'color': '#2c3e50'}),
            
            html.Label("Target Region Base Profile Selection:", style={'fontWeight': '600', 'display': 'block', 'marginTop': '10px'}),
            dcc.Dropdown(
                id='region-selector',
                options=[{'label': v['name'], 'value': k} for k, v in GEO_PROFILES.items()],
                value='Cherrapunji',
                clearable=False
            ),
            
            html.Hr(style={'margin': '15px 0', 'border': '0', 'borderTop': '1px solid #eee'}),
            
            html.Label("IMD Rainfall Data Saturation (24h mm):", style={'fontWeight': '600'}),
            dcc.Slider(id='rain-slider', min=10, max=450, value=240, step=10, marks={50: '50', 150: '150', 250: '250', 350: '350', 450: '450'}),
            
            html.Label("IoT Field Soil Moisture Sensor Data (%):", style={'fontWeight': '600', 'marginTop': '10px', 'display': 'block'}),
            dcc.Slider(id='moist-slider', min=10, max=100, value=75, step=5, marks={20: '20%', 40: '40%', 60: '60%', 80: '80%', 100: '100%'}),
            
            html.H3("📸 Geotagged Field Evidence Tagger", style={'borderBottom': '2px solid #f1f3f5', 'paddingBottom': '8px', 'marginTop': '25px', 'color': '#2c3e50'}),
            html.Label("Media Source Classification:", style={'fontWeight': '600'}),
            dcc.RadioItems(id='media-type-selector', options=[{'label': ' Geotagged Photo  ', 'value': 'Photo'}, {'label': ' Drone Video Footage', 'value': 'Video'}], value='Photo', style={'marginBottom': '10px'}),
            
            html.Label("Incident Observation Note:", style={'fontWeight': '600'}),
            dcc.Input(id='media-note-input', type='text', value='Soil displacement and rock cracking observed.', style={'width': '93%', 'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc', 'marginBottom': '10px'}),
            
            html.Label("Incident Coordinate Offsets (Latitude / Longitude):", style={'fontWeight': '600'}),
            html.Div(style={'display': 'flex', 'gap': '10px', 'marginBottom': '15px'}, children=[
                dcc.Input(id='offset-lat', type='number', value=0.015, step=0.001, style={'width': '45%', 'padding': '6px'}),
                dcc.Input(id='offset-lon', type='number', value=-0.020, step=0.001, style={'width': '45%', 'padding': '6px'}),
            ]),
            
            html.Button("💾 Append Report Marker Layer", id='submit-report-btn', n_clicks=0, style={'width': '100%', 'padding': '10px', 'backgroundColor': '#3498db', 'color': 'white', 'border': 'none', 'borderRadius': '4px', 'fontWeight': 'bold', 'cursor': 'pointer'})
        ]),
        
        # Right Column: AI Diagnostics & Map Rendering Engine Canvas
        html.Div(style={'flex': '2', 'minWidth': '460px', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px', 'boxShadow': '0 4px 10px rgba(0,0,0,0.05)'}, children=[
            html.H3("📊 AI Pipeline Engine Diagnostics", style={'borderBottom': '2px solid #f1f3f5', 'paddingBottom': '8px', 'marginTop': '0', 'color': '#2c3e50'}),
            
            html.Div(id='risk-badge', style={'padding': '15px', 'borderRadius': '6px', 'color': 'white', 'fontSize': '18px', 'fontWeight': 'bold', 'textAlign': 'center', 'marginBottom': '15px'}),
            html.Div(id='risk-text-summary', style={'fontSize': '14px', 'color': '#555', 'backgroundColor': '#f8f9fa', 'padding': '12px', 'borderRadius': '6px', 'borderLeft': '4px solid #34495e', 'marginBottom': '20px'}),
            
            html.H3("🗺️ Live Geographic Threat Heatmap View", style={'borderBottom': '2px solid #f1f3f5', 'paddingBottom': '8px', 'color': '#2c3e50'}),
            
            # Interactive Map View via Local Web Container Frame Slicing
            html.Iframe(id='heatmap-map-frame', style={'width': '100%', 'height': '460px', 'border': 'none', 'borderRadius': '6px'})
        ])
    ]),
    
    dcc.Store(id='stored-report-records', data=[
        {'lat': 25.2810, 'lon': 91.7290, 'type': 'Photo', 'note': 'Initial Base Structural Fissure Discovered.', 'time': '10:14'}
    ])
])

# ==========================================
# 3. DISASTER CONTROL CALLBACK HANDLERS
# ==========================================
@app.callback(
    [Output('stored-report-records', 'data'),
     Output('media-note-input', 'value')],
    [Input('submit-report-btn', 'n_clicks')],
    [State('region-selector', 'value'),
     State('media-type-selector', 'value'),
     State('media-note-input', 'value'),
     State('offset-lat', 'value'),
     State('offset-lon', 'value'),
     State('stored-report-records', 'data')]
)
def handle_media_ingestion(n_clicks, region, m_type, note, d_lat, d_lon, current_store):
    if n_clicks > 0 and note:
        base_c = GEO_PROFILES[region]
        new_entry = {
            'lat': round(base_c['lat'] + d_lat, 5),
            'lon': round(base_c['lon'] + d_lon, 5),
            'type': m_type,
            'note': note,
            'time': time.strftime("%H:%M")
        }
        current_store.append(new_entry)
        return current_store, ""
    return current_store, note


@app.callback(
    [Output('risk-badge', 'children'),
     Output('risk-badge', 'style'),
     Output('risk-text-summary', 'children'),
     Output('heatmap-map-frame', 'srcDoc')],
    [Input('region-selector', 'value'),
     Input('rain-slider', 'value'),
     Input('moist-slider', 'value'),
     Input('stored-report-records', 'data')]
)
def compute_disaster_risk_matrix(region, rainfall, moisture, current_reports):
    profile = GEO_PROFILES[region]
    
    # 1. Evaluate Core Base Station Point Risk Level
    input_df = pd.DataFrame([[profile['slope'], float(rainfall), float(moisture), profile['elevation'], 0.65]], columns=FEATURE_NAMES)
    risk_percentage = round(float(ml_pipeline.predict_proba(input_df)) * 100, 2)
    
    if risk_percentage < 35.0:
        badge_text = "🟩 GREEN STATUS: SYSTEM SAFE"
        badge_color = '#2ecc71'
        directive = f"Live operational monitoring active at {profile['name']}. Remote sensors showing normal telemetry parameters."
    elif risk_percentage < 70.0:
        badge_text = "🟨 YELLOW STATUS: HAZARD WATCH LEVEL"
        badge_color = '#f1c40f'
        directive = f"Warning watch active for {profile['name']}. IMD precipitation or moisture tracking is rising. Restrict hillside engineering works."
    else:
        badge_text = "🟥 RED ALERT STATUS: EMERGENCY EVACUATION CRITICAL"
        badge_color = '#e74c3c'
        directive = f"🚨 EMERGENCY DIRECTIVE: High landslide probability near {profile['name']}! Initialize sirens and deploy S&R teams immediately."

    badge_style = {'padding': '15px', 'borderRadius': '6px', 'color': 'white', 'fontSize': '18px', 'fontWeight': 'bold', 'textAlign': 'center', 'backgroundColor': badge_color}
    summary_text = html.Div([html.B(f"Calculated Center Landslide Probability: {risk_percentage}%"), html.P(directive)])
    
    # 2. Build Folium Core Map Layer Object Canvas
