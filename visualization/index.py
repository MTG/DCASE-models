from .app import app
from .layout import layout
from . import callbacks

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Define layout
app.layout = layout

# Run the app
if (__name__ == '__main__'):
    app.run_server(debug=False)
