from flask import Flask
from flask import url_for, render_template
from segmentation.seg import run_segmenter



app = Flask(__name__)


@app.route('/')
def simple():
    cell_count = run_segmenter(img_path='images/00.png',w_count=False)
    return render_template('index.html', cell_count=cell_count)
 #   return f"Cells: {count}"


if __name__ == '__main__':
    app.run(debug=True)
