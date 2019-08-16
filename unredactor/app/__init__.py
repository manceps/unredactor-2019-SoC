from flask import Flask, send_from_directory
from config import Config
from flask_bootstrap import Bootstrap


app = Flask(__name__, static_url_path='')
app.config.from_object(Config)

bootstrap = Bootstrap(app)

from app import routes
