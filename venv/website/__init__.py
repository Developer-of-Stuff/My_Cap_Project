from flask import Flask


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'fj0a89sdf9a8iusdhf9hjf0912j30rfvxk12543'

    from .views import views

    app.register_blueprint(views, url_prefix='/')

    return app
