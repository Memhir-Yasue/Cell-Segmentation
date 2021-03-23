import os

from flask import Flask, flash, redirect,  render_template, request, url_for, send_from_directory
from segmentation.seg import run_segmenter, save_plt
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/input'
OUTPUT_FOLDER = 'static/output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER


@app.route('/upload/<filename>')
def uploaded_file(filename):
    
    print(app.config['UPLOAD_FOLDER'])
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/processed/<filename>')
def processed_image(filename):
    f_path = app.config['OUTPUT_FOLDER']+"/"+filename
    print(f_path)
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


@app.route('/process/<filename>', methods=['GET'])
def process_image(filename):
    
    img = run_segmenter(img_path='static/input/'+filename)
    extension = filename.rsplit('.', 1)[1].lower()
    output_f_name = "output."+extension
    save_plt(img, output_f_name)

    return redirect(url_for('processed_image', filename=output_f_name))


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            extension = filename.rsplit('.', 1)[1].lower()
            neo_file_name =  f"input.{extension}"
            f_path = os.path.join(app.config['UPLOAD_FOLDER'], neo_file_name)
            file.save(f_path)
            return redirect(url_for('process_image', filename=neo_file_name))
            # return redirect(url_for('uploaded_file', filename=neo_file_name))
        else:
            return render_template('upload_failed.html')

    return render_template('upload.html')


# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>')
def run_segmentation(filename):
    cell_count = run_segmenter(img_path=filename, w_count=False)
    return render_template('index.html', cell_count=cell_count)


if __name__ == '__main__':
    app.run(debug=True)
