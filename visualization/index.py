from .app import app
from .layout import layout

# Define layout
app.layout = layout

from . import callbacks

# Run the app
if (__name__ == '__main__'):
    app.run_server(debug=False)
