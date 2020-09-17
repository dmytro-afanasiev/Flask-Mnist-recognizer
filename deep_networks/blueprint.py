from flask import Blueprint

deep_net = Blueprint('deep_net', __name__, template_folder='templates', static_folder='static')


@deep_net.route('/')
def deep_index():
    return "deep hello"