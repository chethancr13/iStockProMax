from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

# Keep Chrome from exiting instantly
chrome_options = webdriver.ChromeOptions()
chrome_options.add_experimental_option("detach", True)

driver = webdriver.Chrome(options=chrome_options)
driver.get("https://investor.apple.com/stock-price/")

data = driver.find_element(By.XPATH, value='//*[@id="_ctrl0_ctl42_divModuleContainer"]/div/div/div/div[2]/div[2]/span')
change = driver.find_element(By.XPATH, value='//*[@id="_ctrl0_ctl42_divModuleContainer"]/div/div/div/div[2]/div[3]/span/span')
dprice = data.text
dchange = change.text

driver.quit()


app = Flask(__name__)

@app.route('/')
def index():
    # try:
    #     stock_data = yf.download(company_name="AAPL", start=start_date, end=end_date)
    #     if stock_data.empty:
    #         raise ValueError(f"No data found for {company_name}. Check the ticker.")
    # except Exception as e:
    #     return render_template('results.html', result={"error": str(e)})
    # last_price = stock_data['Close'].iloc[-1]
    return render_template('index.html', price_close=dprice, change_price=dchange)

@app.route('/results', methods=['POST'])
def results():
    # Get user inputs
    company_name = request.form['company_name']
    start_year = request.form['start_year']
    end_year = request.form['end_year']
    future_date = request.form['future_date']
    
    # Construct date range
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    
    # Fetch stock data dynamically
    try:
        stock_data = yf.download(company_name, start=start_date, end=end_date)
        if stock_data.empty:
            raise ValueError(f"No data found for {company_name}. Check the ticker.")
    except Exception as e:
        return render_template('results.html', result={"error": str(e)})
    
    # Prepare data
    stock_data.dropna(inplace=True)
    stock_data['Daily Return'] = stock_data['Close'].pct_change()
    stock_data['5 Day Moving Avg'] = stock_data['Close'].rolling(window=5).mean()
    stock_data['10 Day Moving Avg'] = stock_data['Close'].rolling(window=10).mean()
    stock_data['20 Day Moving Avg'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['Volatility'] = stock_data['Daily Return'].rolling(window=5).std()
    stock_data['Volume Change'] = stock_data['Volume'].pct_change()
    stock_data['Future Price'] = stock_data['Close'].shift(-1)
    stock_data.dropna(inplace=True)

    # Define features and target
    features = ['Close', 'Volume', '5 Day Moving Avg', '10 Day Moving Avg', '20 Day Moving Avg', 'Volatility', 'Volume Change']
    X = stock_data[features]
    y = stock_data['Future Price']
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Predict future price
    result = predict_stock_movement(stock_data, model, future_date, features, company_name)
    return render_template('results.html', result=result)

def predict_stock_movement(stock_data, model, input_date, features, company_name):
    input_datetime = datetime.strptime(input_date, "%d-%m-%Y")
    try:
        # Ensure the date exists in the dataset or use the last row for prediction
        if input_datetime in stock_data.index:
            date_data = stock_data.loc[input_datetime]
            features_data = date_data[features].values.reshape(1, -1)
            predicted_price = model.predict(features_data)[0]
        else:
            # Use the most recent data row for prediction if future_date is outside range
            last_row = stock_data.iloc[-1]
            features_data = last_row[features].values.reshape(1, -1)
            predicted_price = model.predict(features_data)[0]

        # Calculate percentage change
        last_price = stock_data['Close'].iloc[-1]
        percentage_change = float(((predicted_price - last_price) / last_price) * 100)
        movement = "Up" if percentage_change > 0 else "Down"
        
        return {
            "company_name": company_name.upper(),
            "future_date": input_date,
            "predicted_price": round(predicted_price, 2),
            "percentage_change": round(percentage_change, 2),
            "movement": movement,
            "price_close": dprice,
            "change_price" : dchange
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    app.run(debug=True)
