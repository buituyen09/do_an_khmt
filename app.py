from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

try:
    model = joblib.load('trained_model.pkl')
except Exception as e:
    model = None
    print(f"Lỗi khi tải mô hình: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            if not all(f'feature{i}' in request.form for i in range(1, 13)):
                raise ValueError("Thiếu dữ liệu đầu vào!")

            # Lấy dữ liệu từ form và chuyển đổi sang kiểu số
            features = [float(request.form[f'feature{i}']) for i in range(1, 13)]
            features = np.array(features).reshape(1, -1)

            if model is None:
                raise ValueError("Mô hình chưa được tải!")

            prediction = model.predict(features)[0]
        except ValueError as ve:
            prediction = f"Lỗi đầu vào: {str(ve)}"
        except Exception as e:
            prediction = f"Lỗi không xác định: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
