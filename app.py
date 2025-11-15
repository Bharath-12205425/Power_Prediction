import os
import logging
from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix
from action_logger import action_logger

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "power-grid-analysis-secret-key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Import routes after app creation
import routes

app.logger.info("Power Grid Analysis App initialized successfully")
action_logger.log_action("APP_INITIALIZATION", "Power Grid Analysis App started")

if __name__ == "__main__":
    app.run(debug=True)
