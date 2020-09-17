from flask import Flask

from flask import Blueprint

from config import Config
from deep_net.blueprint import deep_net

app = Flask(__name__)
app.config.from_object(Config)
app.register_blueprint(deep_net, url_prefix='/deep_net')

from views import *


if __name__=='__main__':
    app.run(debug=True)

