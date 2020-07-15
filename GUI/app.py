from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
from werkzeug.utils import secure_filename
import os
from datetime import timedelta
import sys
import os

path = os.path.dirname(os.getcwd())
sys.path.append(path + r"/SourceCode")
sys.path.append(path + r"/SourceCode/CardPositioning")
sys.path.append(path + r"/SourceCode/CardPositioning/checkpoints")

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)

app.send_file_max_age_default = timedelta(seconds=1)

@app.route('/', methods=['POST' , 'GET'])
def upload():
    print(request.method)
    if request.method == 'POST':
        f = request.files['file0']
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、jpeg"})

        user_input = request.form.get("name")

        basepath = os.path.dirname(__file__)  # 当前文件所在路径

        upload_path = os.path.join(basepath, 'static/images_test', secure_filename(f.filename))
        f.save(upload_path)

        print("成功读取")
        print(upload_path)

        bankCard = str(loadSystem(""+(upload_path)))
        if(bankCard != None):
            json_cardNumber = {
                "银行卡卡号:" : bankCard
            }
            print(json_cardNumber)
            return jsonify(json_cardNumber)
        else:
            json_cardNumber = {
                "银行卡卡号:": "未识别成功"
            }

            return jsonify(json_cardNumber)

    return render_template('yhk5.html')


if __name__ == '__main__':
    from demo import loadSystem
    load_result = str(loadSystem("./static/yhk.png"))
    print(load_result)
    app.run()

