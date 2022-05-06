"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""

import argparse
import io
import os
import json
import glob
from PIL import Image
from uuid import uuid4
import torch
from flask import Flask, render_template, request, redirect,url_for
import numpy as np

def DeleteAllFiles(filePath):
    if os.path.exists(filePath):
        for file in os.scandir(filePath):
            os.remove(file.path)
            
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        DeleteAllFiles('C:\\web\\yolov5-flask\\static\\aft')  #파일삭제
        DeleteAllFiles('C:\\web\\yolov5-flask\\static\\bef')
        
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files.getlist("file")
        if not file:
            return

        pf=[]
        for file in file:
            filename = file.filename.rsplit("/")[0]
            print("진행 중 파일 :", filename)

            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
            # print(img)
            img.save(f"static/bef/{filename}", format="JPEG")
            print('원본 저장')

            results = model(img, size=640)
            results.render()  # updates results.imgs with boxes and labels
            data = results.pandas().xyxy[0][['name']].values.tolist()
            print("데이터:",data)

            for img in results.imgs:
                img_base64 = Image.fromarray(img)
                img_base64.save(f"static/aft/{filename}", format="JPEG")
                print('디텍트 저장')

            if len(data) == 0:
                pf.append("PASS")
            if len(data) != 0:
                pf.append("FAIL")
            print(pf)
            root = "static/aft"
            if not os.path.isdir(root):
                return "Error : not found!"
            files = []
            for file in glob.glob("{}/*.*".format(root)):
                fname = file.split(os.sep)[-1]
                files.append(fname)
            print("파일스 :",files)
            firstimage = "static/aft/"+files[0]
            print("파일1 :",firstimage)
        return render_template("imageshow.html",files=files,pf=pf,firstimage=firstimage,enumerate=enumerate,len=len)
    return render_template("index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load(
        'ultralytics/yolov5', 'custom', path='C:\web\yolov5-flask\yolov5l.pt', autoshape=True
    )  # force_reload = recache latest code
    model.eval()

    flask_options = dict(
        host='0.0.0.0',
        debug=True,
        port=args.port,
        threaded=True,
    )

    app.run(**flask_options)
    
    
    
# @app.route("/")
# def index():
#     return render_template("index.html")
#
# @app.route("/upload", methods=["POST"])
# def upload():
#     """Handle the upload of a file."""
#     form = request.form
#     # Create a unique "session ID" for this particular batch of uploads.
#     upload_key = str(uuid4())
#
#     # Is the upload using Ajax, or a direct POST by the form?
#     is_ajax = False
#     if form.get("__ajax", None) == "true":
#         is_ajax = True
#
#     # Target folder for these uploads.
#     target = "uploadr/static/uploads/{}".format(upload_key)
#     try:
#         os.mkdir(target)
#     except:
#         if is_ajax:
#             return ajax_response(False, "Couldn't create upload directory: {}".format(target))
#         else:
#             return "Couldn't create upload directory: {}".format(target)
#
#     print("=== Form Data ===")
#     for key, value in list(form.items()):
#         print(key, "=>", value)
#
#     for upload in request.files.getlist("file"):
#         filename = upload.filename.rsplit("/")[0]
#         print("Accept incoming file:", filename)
#
#         destination = "/".join([target, filename])
#         print("Save it to:", destination)
#         upload.save(destination)
#
#     if is_ajax:
#         return ajax_response(True, upload_key)
#     else:
#         return redirect(url_for("upload_complete", uuid=upload_key))
#
# @app.route("/files/<uuid>")
# def upload_complete(uuid):
#     """The location we send them to at the end of the upload."""
#
#     # Get their files.
#     root = "uploadr/static/uploads/{}".format(uuid)
#     if not os.path.isdir(root):
#         return "Error: UUID not found!"
#
#     files = []
#     for file in glob.glob("{}/*.*".format(root)):
#         fname = file.split(os.sep)[-1]
#         files.append(fname)
#
#     return render_template("files.html",
#         uuid=uuid,
#         files=files,
#     )
#
#
# def ajax_response(status, msg):
#     status_code = "ok" if status else "error"
#     return json.dumps(dict(
#         status=status_code,
#         msg=msg,
#     ))
#







        # for debugging
        # data = results.pandas().xyxy[0][['name']].values.tolist()
        # data1 = results.pandas().xyxy[0][['confidence']].values.round(2).tolist()
        #
        # # return data
        #
        # results.render()  # updates results.imgs with boxes and labels
        # for img in results.imgs:
        #     img_base64 = Image.fromarray(img)
        #     img_base64.save("static/image0.jpg", format="JPEG")
        # # return redirect("static/image0.jpg")
        



