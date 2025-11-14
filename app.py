from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from datetime import datetime
from finta import TA
import pickle


model_path = 'bitcoin_price_predictor.keras'
scaler_path = 'full_scaler.pkl'
data_filename = 'btc_1d_data_2018_to_2025.csv'
date_col = 'Open time'
n_past = 60


base_features = ['close', 'high', 'low', 'volume']
derived_features = ['SMA_20', 'RSI_14']
features = base_features + derived_features
n_features = len(features)
PREDICT_COL_INDEX = features.index('close')

app = Flask(__name__)
CORS(app)

model = None
scaler = None
full_dataset_df = None

def load_data_and_model():
    global model, scaler, full_dataset_df

    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    print("Model loaded.")
    
    print(f"Loading scaler from {scaler_path}...")
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    print("Scaler loaded.")

    print(f"Loading full dataset from {data_filename}...")
    try:
        full_dataset_df = pd.read_csv(data_filename)
        
        full_dataset_df[date_col] = pd.to_datetime(full_dataset_df[date_col])
        
        
        if full_dataset_df[date_col].dt.tz is not None:
            print("Converting dates to naive (removing timezone)...")
            full_dataset_df[date_col] = full_dataset_df[date_col].dt.tz_localize(None)
      
        
        full_dataset_df = full_dataset_df.sort_values(by=date_col)
        
        
        full_dataset_df.rename(columns={"Close": "close", "High": "high", "Low": "low", "Volume": "volume"}, inplace=True)
        
        full_dataset_df = full_dataset_df.set_index(date_col)

        
        print("Calculating indicators for full dataset...")
        full_dataset_df['SMA_20'] = TA.SMA(full_dataset_df, period=20)
        full_dataset_df['RSI_14'] = TA.RSI(full_dataset_df, period=14)
        full_dataset_df = full_dataset_df.dropna()
        
        print("Data and features loaded.")

    except Exception as e:
        print(f"Error loading data: {e}")
        exit()


@app.route('/predict', methods=['POST'])
def predict_date():
    try:
        data = request.get_json(force=True)
        target_date_str = data['date']
        target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
        
        last_real_date = full_dataset_df.index.max()

        if target_date <= last_real_date:
            
            print(f"Request for PAST date: {target_date_str}")
            
            if target_date not in full_dataset_df.index:
                return jsonify({'error': f'No data for {target_date_str}.'}), 404
            
            actual_price = float(full_dataset_df.loc[target_date][features[PREDICT_COL_INDEX]])
            
            target_index = full_dataset_df.index.get_loc(target_date)
            if target_index < n_past:
                return jsonify({'error': 'Not enough historical data.'}), 400
                
            past_n_days_values = full_dataset_df[features].iloc[target_index - n_past : target_index].values
            
            past_n_scaled = scaler.transform(past_n_days_values)
            X_pred = np.reshape(past_n_scaled, (1, n_past, n_features))
            
            pred_scaled = model.predict(X_pred)
            predicted_all_features = scaler.inverse_transform(pred_scaled)
            predicted_price = float(predicted_all_features[0][PREDICT_COL_INDEX])

            return jsonify({
                'type': 'past', 'prediction': predicted_price,
                'actual': actual_price, 'date': target_date_str
            })

        else:
            
            print(f"Request for FUTURE date: {target_date_str}")
            days_to_predict = (target_date - last_real_date).days
            if days_to_predict > 180:
                return jsonify({'error': 'Cannot predict more than 180 days.'}), 400

            
            current_batch_scaled = scaler.transform(full_dataset_df[features].iloc[-n_past:].values)
            for i in range(days_to_predict):
                X_pred = np.reshape(current_batch_scaled, (1, n_past, n_features))
                pred_scaled = model.predict(X_pred) 
                current_batch_scaled = np.append(current_batch_scaled[1:], pred_scaled, axis=0)
            
            final_prediction_all_features = scaler.inverse_transform(pred_scaled)
            final_prediction_price = float(final_prediction_all_features[0][PREDICT_COL_INDEX])

            return jsonify({
                'type': 'future', 'prediction': final_prediction_price,
                'date': target_date_str, 'days_predicted': days_to_predict
            })

    except Exception as e:
        print(f"Server Error in /predict: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/backtest', methods=['POST'])
def backtest_range():
    try:
        data = request.get_json(force=True)
        start_date_str = data['start_date']
        end_date_str = data['end_date']
        
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        
        print(f"Processing back-test from {start_date_str} to {end_date_str}...")

        
        start_index = full_dataset_df.index.searchsorted(start_date, side='left')
        
        
        end_index = full_dataset_df.index.searchsorted(end_date, side='right') - 1
        
       

        if start_index > end_index:
             return jsonify({'error': 'No data found for the selected date range.'}), 404
             
        if (start_index - n_past) < 0:
             return jsonify({'error': f'Not enough data before {start_date_str}. Need {n_past} days prior.'}), 400

        dates_list = []
        predictions_list = []
        actuals_list = []
        
        
        for i in range(start_index, end_index + 1):
            
           
            past_n_days_values = full_dataset_df[features].iloc[i - n_past : i].values
            
           
            past_n_scaled = scaler.transform(past_n_days_values)
            X_pred = np.reshape(past_n_scaled, (1, n_past, n_features))
            
            pred_scaled = model.predict(X_pred)
            predicted_all_features = scaler.inverse_transform(pred_scaled)
            predicted_price = float(predicted_all_features[0][PREDICT_COL_INDEX])

            
            actual_price = float(full_dataset_df[features].iloc[i][PREDICT_COL_INDEX])
            current_date = full_dataset_df.index[i]

            
            predictions_list.append(predicted_price)
            actuals_list.append(actual_price)
            dates_list.append(current_date.strftime('%Y-%m-%d'))
            
        print(f"Back-test complete. Returning {len(dates_list)} data points.")
        
        if not dates_list:
            return jsonify({'error': 'No data processed for this range.'}), 404
            
        
        mae = np.mean(np.abs(np.array(actuals_list) - np.array(predictions_list)))

        return jsonify({
            'dates': dates_list,
            'predictions': predictions_list,
            'actuals': actuals_list,
            'mae': float(mae),
            'start_date_pred': predictions_list[0],
            'start_date_actual': actuals_list[0],
            'end_date_pred': predictions_list[-1],
            'end_date_actual': actuals_list[-1]
        })

    except Exception as e:
        print(f"Server Error in /backtest: {e}")
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    load_data_and_model() 
    print("Starting Flask server... Access at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000)