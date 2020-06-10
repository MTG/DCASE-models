from app import app
from layout import layout
import callbacks

# Define layout
app.layout = layout

# Run the app
if (__name__ == '__main__'):
    app.run_server(debug=False)
