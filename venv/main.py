from website import create_app
from website.data_loading import MatrixFactorization, Loader

app = create_app()

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
