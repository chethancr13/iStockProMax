from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def results():
    start_year = request.form['start_year']
    end_year = request.form['end_year']
    future_date = request.form['future_date']
    
    # Convert dates
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    
    # Fetch data
    stock_data = yf.download('AAPL', start=start_date, end=end_date)
    stock_data.dropna(inplace=True)
    stock_data['Daily Return'] = stock_data['Close'].pct_change()
    stock_data['5 Day Moving Avg'] = stock_data['Close'].rolling(window=5).mean()
    stock_data['10 Day Moving Avg'] = stock_data['Close'].rolling(window=10).mean()
    stock_data['20 Day Moving Avg'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['Volatility'] = stock_data['Daily Return'].rolling(window=5).std()
    stock_data['Volume Change'] = stock_data['Volume'].pct_change()
    stock_data['Future Price'] = stock_data['Close'].shift(-1)
    stock_data.dropna(inplace=True)

    # Features and target
    features = ['Close', 'Volume', '5 Day Moving Avg', '10 Day Moving Avg', '20 Day Moving Avg', 'Volatility', 'Volume Change']
    X = stock_data[features]
    y = stock_data['Future Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    rf_regressor = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf_regressor.fit(X_train, y_train)

    # Predict future price
    result = predict_stock_movement(stock_data, rf_regressor, future_date, features)
    return render_template('results.html', result=result)

def predict_stock_movement(stock_data, model, input_date, features):
    input_datetime = datetime.strptime(input_date, "%d/%m/%Y")
    if input_datetime in stock_data.index:
        date_data = stock_data.loc[input_datetime]
        features_data = date_data[features].values.reshape(1, -1)
        predicted_price = model.predict(features_data)[0]
    else:
        last_available_day = stock_data.iloc[-1]
        days_to_future = (input_datetime - stock_data.index[-1]).days
        last_available_day_data = last_available_day[features].values.reshape(1, -1)
        predicted_price = last_available_day['Close']
        for _ in range(days_to_future):
            predicted_price = model.predict(last_available_day_data)[0]
            last_available_day_data = np.array([predicted_price] + list(last_available_day_data[0][1:])).reshape(1, -1)
    last_price = stock_data['Close'].iloc[-1]
    percentage_change = float(((predicted_price - last_price)) / last_price) * 100
    movement = 'Up' if percentage_change > 0 else 'Down'
    return {
        "future_date": input_date,
        "predicted_price": round(predicted_price, 2),
        "percentage_change": round(percentage_change, 2),
        "movement": movement
    }

if __name__ == '__main__':
    app.run(debug=True)
