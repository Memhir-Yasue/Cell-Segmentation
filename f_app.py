from flask import Flask
from segmentation.seg import run_segmenter


app = Flask(__name__)


@app.route('/')
def simple():
    img, count = run_segmenter(img_path='images/01.png')
    return f"Cells: {count}"


if __name__ == '__main__':
    app.run(debug=True)
