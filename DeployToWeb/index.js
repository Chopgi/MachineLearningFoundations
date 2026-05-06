// index.js – Housing price predictor using DLnet_housingData.onnx

async function runHousingPrediction() {
  // Read the six input features from the HTML inputs
  const features = [
    'housing_median_age',
    'total_rooms',
    'total_bedrooms',
    'population',
    'households',
    'median_income'
  ].map(id => parseFloat(document.getElementById(id).value) || 0);

  // Create input tensor of shape [1, 6]
  const tensorX = new ort.Tensor('float32', new Float32Array(features), [1, 6]);

  try {
   
    const session = await ort.InferenceSession.create(
      './DLnet_housingData.onnx?v=' + Date.now()
    );
    const results = await session.run({ input1: tensorX });
    const scaledPred = results.output1.data[0]; // standardized output

    // Restore the actual dollar value using the training set’s mean and std
    const yMean = 206704.0312;
    const yStd  = 115474.7812;
    const actualPrice = scaledPred * yStd + yMean;

    document.getElementById('predictionResult').textContent =
      `Predicted Median House Value: $${actualPrice.toFixed(2)}`;
  } catch (e) {
    console.error('ONNX runtime error:', e);
    alert('Error: ' + e.message);
  }
}