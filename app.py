import os
import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Initialize the Dash application framework
app = dash.Dash(__name__)
server = app.server  # Required by Render's WSGI server to scale on the cloud

# ==========================================
# 1. CORE MACHINE LEARNING ENGINE (SIH26001)
# ==========================================
FEATURE_NAMES = ['slope_angle', 'rainfall_24h', 'soil_moisture', 'elevation', 'deforestation_idx']

def initialize_ml_pipeline():
    """Generates a synthetic matrix and pre-trains the pipeline model."""
    np.random.seed(42)
    samples = 1500
    X_mock = pd.DataFrame({
        'slope_angle': np.random.uniform(15, 60, samples),
        'rainfall_24h': np.random.uniform(10, 350, samples),
        'soil_moisture': np.random.uniform(20, 95, samples),
        'elevation': np.random.uniform(500, 3000, samples),
        'deforestation_idx': np.random.uniform(0.1, 0.9, samples)
    })
    # Landslide trigger threshold function logic
    y_mock = ((X_mock['rainfall_24h']/350 + X_mock['slope_angle']/60 + X_mock['soil_moisture']/95) > 1.2).astype(int)
    pipeline = RandomForestClassifier(n_estimators=50, random_state=42)
    pipeline.fit(X_mock, y_mock)
    return pipeline

ml_pipeline = initialize_ml_pipeline()

# ==========================================
# 2. UI LAYOUT & BRANDING CONFIGURATION
# ==========================================
app.layout = html.Div(style={'fontFamily': 'Segoe UI, Arial, sans-serif', 'padding': '40px', 'backgroundColor': '#f4f6f9'}, children=[
    
    # Official Hackathon Header Banner Module
    html.Div(style={'textAlign': 'center', 'marginBottom': '30px', 'padding': '20px', 'backgroundColor': '#2c3e50', 'borderRadius': '8px', 'color': 'white'}, children=[
        html.H1("🌋 MDoNER AI Landslide Early Warning Engine", style={'margin': '0 0 10px 0', 'fontSize': '28px'}),
        html.P("Smart India Hackathon Cloud Prototype Node • Problem Statement Statement ID: SIH26001", style={'margin': '0', 'opacity': '0.8', 'fontSize': '14px'})
    ]),
    
    # Dashboard Grid Splitting Panels
    html.Div(style={'display': 'flex', 'gap': '30px', 'flexWrap': 'wrap'}, children=[
        
        # Left Grid Section: Interactive Telemetry Controls
        html.Div(style={'flex': '1', 'minWidth': '320px', 'backgroundColor': 'white', 'padding': '25px', 'borderRadius': '8px', 'boxShadow': '0 4px 10px rgba(0,0,0,0.05)'}, children=[
            html.H3("📡 Live Telemetry Input Hub", style={'borderBottom': '2px solid #f1f3f5', 'paddingBottom': '10px', 'color': '#343a40', 'marginTop': '0'}),
            
            html.Div(style={'marginBottom': '25px'}, children=[
                html.Label("Target NER Monitor Region Cluster Selection:", style={'fontWeight': '600', 'color': '#495057', 'display': 'block', 'marginBottom': '8px'}),
                dcc.Dropdown(
                    id='region-dropdown',
                    options=[
                        {'label': 'Cherrapunji (Meghalaya) [Cluster Node 1]', 'value': 'Cherrapunji'},
                        {'label': 'Aizawl (Mizoram) [Cluster Node 2]', 'value': 'Aizawl'},
                        {'label': 'Kohima (Nagaland) [Cluster Node 3]', 'value': 'Kohima'}
                    ],
                    value='Cherrapunji',
                    clearable=False
                )
            ]),
            
            html.Div(style={'marginBottom': '25px'}, children=[
                html.Label("IMD Live Rainfall Saturation (24h mm):", style={'fontWeight': '600', 'color': '#495057'}),
                dcc.Slider(id='rain-slider', min=10, max=450, value=240, step=10, marks={50: '50mm', 150: '150mm', 250: '250mm', 350: '350mm', 450: '450mm'}),
            ]),
            
            html.Div(style={'marginBottom': '10px'}, children=[
                html.Label("IoT Field Soil Moisture Sensor Data Node (%):", style={'fontWeight': '600', 'color': '#495057'}),
                dcc.Slider(id='moist-slider', min=10, max=100, value=75, step=5, marks={20: '20%', 40: '40%', 60: '60%', 80: '80%', 100: '100%'}),
            ]),
        ]),
        
        # Right Grid Section: AI Decision Outputs
        html.Div(style={'flex': '1.5', 'minWidth': '400px', 'backgroundColor': 'white', 'padding': '25px', 'borderRadius': '8px', 'boxShadow': '0 4px 10px rgba(0,0,0,0.05)'}, children=[
            html.H3("📊 AI Model Prediction Analysis", style={'borderBottom': '2px solid #f1f3f5', 'paddingBottom': '10px', 'color': '#343a40', 'marginTop': '0'}),
            
            # Dynamic Target Status Block Component
            html.Div(id='alert-badge', style={'padding': '18px', 'borderRadius': '6px', 'color': 'white', 'fontSize': '20px', 'fontWeight': 'bold', 'textAlign': 'center', 'transition': 'all 0.3s ease'}),
            
            html.Div(style={'marginTop': '25px', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '6px', 'borderLeft': '4px solid #adb5bd'}, children=[
                html.H4("💡 Diagnostic Vector Summary", style={'margin': '0 0 10px 0', 'color': '#495057'}),
                html.Div(id='risk-percentage-text', style={'fontSize': '16px', 'fontWeight': '600', 'marginBottom': '8px', 'color': '#212529'}),
                html.Div(id='directive-text', style={'fontSize': '14px', 'color': '#495057', 'lineHeight': '1.5'})
            ])
        ])
    ])
])

# ==========================================
# 3. DYNAMIC RE-CALCULATION CALLBACKS
# ==========================================
@app.callback(
    [Output('alert-badge', 'children'),
     Output('alert-badge', 'style'),
     Output('risk-percentage-text', 'children'),
     Output('directive-text', 'children')],
    [Input('region-dropdown', 'value'),
     Input('rain-slider', 'value'),
     Input('moist-slider', 'value')]
)
def update_diagnostics(region, rainfall, moisture):
    # Setup varying localized environmental constant profiles to simulate geographic traits
    geo_constants = {'Cherrapunji': [48.0, 1480], 'Aizawl': [38.5, 1120], 'Kohima': [42.0, 1440]}
    slope_angle, elevation = geo_constants.get(region, [40.0, 1200])
    
    # Structure features into an explicit DataFrame to satisfy scikit-learn requirements
    input_df = pd.DataFrame([[slope_angle, float(rainfall), float(moisture), elevation, 0.65]], columns=FEATURE_NAMES)
    
    # Calculate probability via model pipeline
    risk_prob = float(ml_pipeline.predict_proba(input_df))
    risk_percentage = round(risk_prob * 100, 2)
    
    # Automated Alert Threshold Trigger Logic
    if risk_percentage < 35.0:
        badge_text = "🟩 GREEN STATUS: NORMAL OPERATIONS"
        badge_style = {'backgroundColor': '#2ecc71', 'padding': '18px', 'borderRadius': '6px', 'color': 'white', 'fontWeight': 'bold', 'textAlign': 'center'}
        directive = f"Target zone status within acceptable parameters. Live connection node active for regional array at {region}. No active evacuation protocols required."
    elif risk_percentage < 70.0:
        badge_text = "🟨 YELLOW STATUS: WATCH NOTICE TRIGGERED"
        badge_style = {'backgroundColor': '#f1c40f', 'padding': '18px', 'borderRadius': '6px', 'color': 'white', 'fontWeight': 'bold', 'textAlign': 'center'}
        directive = f"Warning status active for {region} cluster. Heavy hillside excavation, blasting, and road-widening works should be temporarily halted. Civil control cells notified."
    else:
        badge_text = "🟥 RED ALERT STATUS: EMERGENCY EVACUATION CRITICAL"
        badge_style = {'backgroundColor': '#e74c3c', 'padding': '18px', 'borderRadius': '6px', 'color': 'white', 'fontWeight': 'bold', 'textAlign': 'center'}
        directive = f"🚨 CRITICAL ACTION REQUIRED: High probability of structural slope failure calculated near the {region} node! Trigger automated public address sirens and mobilize local disaster management teams immediately."
        
    return badge_text, badge_style, f"Calculated Landslide Probability Matrix: {risk_percentage}%", directive

if __name__ == '__main__':
    # Grab port mapped by cloud container hosting stack environment variables
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host='0.0.0.0', port=port, debug=False)
