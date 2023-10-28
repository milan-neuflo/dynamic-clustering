import pandas as pd
from flask import Flask, jsonify

app = Flask(__name__)

# Load the rfm.csv file into a DataFrame
data = pd.read_csv('rfm.csv')
remaining_data = data.copy()

@app.route('/stream-data', methods=['GET'])
def stream_data():
    global remaining_data
    if not remaining_data.empty:
        data_to_send = remaining_data.iloc[:50]
        remaining_data = remaining_data.iloc[50:]  # Remove sent data from remaining_data
        return jsonify({"data": data_to_send.to_dict(orient='records'), "stop": False})
    else:
        return jsonify({"data": [], "stop": True})

if __name__ == "__main__":
    app.run(port=5000)