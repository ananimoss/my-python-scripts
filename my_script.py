pip install ccxt filterpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Conv1D , LSTM , Dropout , Dense
from filterpy.kalman import KalmanFilter
from google.colab import drive
import os
import json

#______________________________________________________________________________

epochss     = 100 # تعداد اپوک ها

#______________________________________________________________________________

drive.mount('/content/drive')

# نام فایل اصلی در Google Drive
array_file_path = '/content/drive/MyDrive/data_xrp_4h.json'
# نام فایل جدید(مقدار جدید) در Google Drive
new_file_path = '/content/drive/MyDrive/ninety_eight_candles_xrp_4h.json'

def read_data(array_file_path):
    if os.path.exists(array_file_path):
        with open(array_file_path, 'r') as file:
            data = json.load(file)
        return data
    else:
        print("File does not exist.")
        return None

# اجرای تابع برای خواندن داده‌ها
data = read_data(array_file_path)

if data is not None:
    print("Data read successfully:")
    print(data)

def read_data_2(new_file_path):
    if os.path.exists(new_file_path):
        with open(new_file_path, 'r') as file:
            ninth_candles = json.load(file)
        return ninth_candles
    else:
        print("File does not exist.")
        return None

# اجرای تابع برای خواندن داده‌ها
ninth_candles = read_data_2(new_file_path)

#______________________________________________________________________________

symbols_t    = ['XRP/USD']
timeframe_t  = ['4h']  # تایم فریم
limit        = 100

# لیستی برای ذخیره داده‌ها
data_t = []
ninth_candles_t = []

for time_t in timeframe_t:
    for symbol_t in symbols_t:
        candles_t = exchange.fetch_ohlcv(symbol_t, time_t, limit=limit)
        symbol_data_t = []  # لیستی برای داده‌های هر ارز

    # جمع‌آوری داده‌های اول تا نهمین کندل
        for i in range(limit):
        # پروسس اطلاعات کندل
            open_price_t = candles_t[i][1]
            high_price_t = candles_t[i][2]
            low_price_t = candles_t[i][3]
            close_price_t = candles_t[i][4]
            volume_price_t = candles_t[i][5]

        # اضافه کردن قیمت‌ها به آرایه داده‌ها با ترتیب مشخص
            if i < limit - 2:  # تا کندل نهم
                symbol_data_t.append([open_price_t, high_price_t, low_price_t, close_price_t, volume_price_t])
            if i == limit - 2:  # نهمین کندل
                ninth_candles_t.append([close_price_t])

    # داده‌های کندل‌های ۸ تایی را به داده‌های کل اضافه می‌کنیم
        data_t.append(symbol_data_t)


#______________________________________________________________________________

# تبدیل خروجی به فرمت مناسب
X_train = np.array(data)
y_train = np.array(ninth_candles)

# نرمال‌سازی داده‌ها
scaler_X = MinMaxScaler(feature_range=(0, 1))
X_train_reshaped = X_train.reshape(X_train.shape[0] * X_train.shape[1], X_train.shape[2])  # تغییر شکل برای نرمال‌سازی
X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(X_train.shape)

scaler_y = MinMaxScaler(feature_range=(0, 1))
y_train_reshaped = y_train.reshape(-1, 1)  # تبدیل y_train به آرایه دو بعدی
y_train_scaled = scaler_y.fit_transform(y_train_reshaped)

# ایجاد مدل
model = Sequential()

# لایه‌های CNN
model.add(Conv1D(filters=64,

kernel_size=2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

# لایه‌های LSTM
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))

# لایه خروجی
model.add(Dense(1))

# کامپایل مدل
model.compile(optimizer='adam', loss='mean_squared_error')

# آموزش مدل
model.fit(X_train_scaled, y_train_scaled, epochs = epochss, batch_size=1 , verbose =1)

# تعریف ورودی تست از داده‌های جمع‌آوری شده
# برای هر ارز در data، نهمین کندل را به عنوان ورودی تست استفاده خواهیم کرد
X_test = np.array(data_t)

# نرمال‌ساز

X_test_reshaped = X_test.reshape(X_test.shape[0] * X_test.shape[1], X_test.shape[2])  # تغییر شکل برای نرمال‌سازی
X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(X_test.shape)

# پیش‌بینی
predictions = model.predict(X_test_scaled)

# در اینجا پیش‌بینی‌های نهمین کندل (y_train) را برای تغییر یابند
predictions_inverse = scaler_y.inverse_transform(predictions)

# استفاده از فیلتر کالمن برای بهینه‌سازی پیش‌بینی
kf = KalmanFilter(dim_x=2, dim_z=1)
kf.x = np.array([[predictions_inverse[0][0]], [0]])  # حالت اولیه
kf.P *= 1000.  # عدم قطعیت اولیه
kf.F = np.array([[1, 1], [0, 1]])  # مدل حرکت
kf.H = np.array([[1, 0]])  # ماتریس مشاهده
kf.Q = np.array([[1, 0], [0, 1]])  # کوواریانس نویز فرایند
kf.R = np.array([[5]])  # کوواریانس نویز اندازه گیری

filtered_predictions = []

for observation in predictions_inverse:
    kf.predict()
    kf.update(observation)
    filtered_predictions.append(kf.x[0, 0])

# نمایش نتایج
plt.plot(y_train, label='asl', marker='o')
plt.plot(predictions_inverse, label='pishbini', marker='x')
plt.plot(filtered_predictions, label='پیش‌بینی تصحیح شده با فیلتر کالمنلتر کالمن' , marker = 'o')
plt.xlabel('نمونه‌ها')
plt.ylabel('قیمت')
plt.legend()
plt.show()

# نمایش پیش‌بینی‌های نهایی
print(f"pishbini ha: {predictions_inverse.flatten()}")
print(f"pishbini haye tashih shodeh  : {filtered_predictions}")
print("قیمت واقعی")
print(str(ninth_candles_t))
