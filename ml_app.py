import onnx
import onnxruntime
from flask import Flask
from flask import request, make_response,  render_template, jsonify
import base64
import numpy as np
from torchvision.io import read_image, ImageReadMode
from predict import make_prediction


app = Flask('image-classification')


@app.route('/forward', methods=['GET', 'POST'])
def upload_img():
    if request.method == 'GET':
        form = """
        <form method='POST' enctype="multipart/form-data">
            <input type="file" name="image" accept=".jpg, .jpeg, .png">
            <input type="submit" name="submit" value="Upload">
        </form>
        """
        return form
    elif request.method == 'POST':

        try:
            # processing request
            file = request.files['image']
            with open(f"./examples/{file.filename}", 'rb') as img:
                source_image = base64.b64encode(img.read()).decode('utf-8')
            img_np = read_image(f"./examples/{file.filename}", mode=ImageReadMode.GRAY)

        except FileNotFoundError:
            response = make_response("bag request", 400)
            return response

        else:
            # if request is OK, make prediction
            pred = make_prediction(ort_session, img_np)[0]

            if pred is None:
                return make_response("модель не смогла обработать данные", 403)

            return render_template('view_image.html', prediction=int(np.argmax(pred)), string=source_image)


@app.route('/metadata')
def get_metadata():
    model = onnx.load("models/model.onnx")
    metadata = model.metadata_props
    response = {
        "commit": metadata[0].value,
        "date": metadata[1].value,
        "experiment": metadata[2].value
    }

    return jsonify(response)


if __name__ == '__main__':
    ort_session = onnxruntime.InferenceSession("models/model.onnx")

    app.run()
