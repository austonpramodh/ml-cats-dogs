from flask import Flask
from .config import config
from .cat_dog.routes import bp as cate_dogs_bp


def create_app(config_name):
    """Construct the core application."""
    app = Flask(__name__)
    app.config.from_object(config.get(config_name or 'default'))

    with app.app_context():
        # Imports
        # from . import routes
        app.register_blueprint(cate_dogs_bp)

        return app
