<!DOCTYPE html>
<html>
<head>
  <title>AQI Predictor</title>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.2.0"></script>
</head>
<body>
  <h1>Air Quality Index (AQI) Predictor</h1>

  <form id="aqiForm">
    <label>PM2.5: <input type="number" id="pm25" required></label><br>
    <label>PM10: <input type="number" id="pm10" required></label><br>
    <label>NO2: <input type="number" id="no2" required></label><br>
    <label>SO2: <input type="number" id="so2" required></label><br>
    <label>CO: <input type="number" id="co" required></label><br>
    <label>O3: <input type="number" id="o3" required></label><br>
    <label>Temperature (°C): <input type="number" id="temp" required></label><br>
    <label>Humidity (%): <input type="number" id="humidity" required></label><br>
    <label>Wind Speed (m/s): <input type="number" id="wind" required></label><br>
    <button type="submit">Predict AQI</button>
  </form>

  <h2 id="result"></h2>

  <script>
    let model;

    async function loadModel() {
      model = await tf.loadLayersModel('model/model.json'); // hosted TF.js model
      console.log("Model loaded.");
    }

    function getInputData() {
      return [
        parseFloat(document.getElementById("pm25").value),
        parseFloat(document.getElementById("pm10").value),
        parseFloat(document.getElementById("no2").value),
        parseFloat(document.getElementById("so2").value),
        parseFloat(document.getElementById("co").value),
        parseFloat(document.getElementById("o3").value),
        parseFloat(document.getElementById("temp").value),
        parseFloat(document.getElementById("humidity").value),
        parseFloat(document.getElementById("wind").value)
      ];
    }

    document.getElementById("aqiForm").addEventListener("submit", async function(e) {
      e.preventDefault();
      const input = tf.tensor2d([getInputData()]);
      const prediction = model.predict(input);
      const aqi = (await prediction.data())[0].toFixed(2);
      document.getElementById("result").innerText = `Predicted AQI: ${aqi}`;
    });

    loadModel();
  </script>
</body>
</html>
