from website import create_app
from website.data_loading import MatrixFactorization, Loader

app = create_app()

if __name__ == '__main__':
    app.run()
