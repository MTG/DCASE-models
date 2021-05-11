import dash
import dash_bootstrap_components as dbc

# Define app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SPACELAB],
    suppress_callback_exceptions=False
)

