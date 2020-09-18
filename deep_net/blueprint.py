from flask import Blueprint
from flask import render_template
from flask import request
from flask import make_response
from flask import abort

deep_net = Blueprint('deep_net', __name__, template_folder='templates', static_folder='static')


def make_predict(data : dict) -> int:
    print(data)



@deep_net.route('/')
def simple_paint():
    return render_template('deep_net/index.html')


@deep_net.route('/send', methods=['POST', 'GET'])
def send():
    if request.method == 'POST':
        make_predict(request.json)
        res = make_response('correct')
        return res
    if request.method == 'GET':
        abort(404)

@deep_net.errorhandler(404)
def error_404(error):
    return "<h1>Error 404</h1><br><p>Not found</p>", 404