# Time Series Analysis and Forecasting

This project demonstrates the implementation of various techniques for time series data analysis and forecasting, including outlier detection, trend removal, and polynomial regression.

## Features

1. **Data Handling**:
   - Ability to load real-world stock data or generate simulated data with customizable outlier ratio.
   - Handling of missing values and data preprocessing.

2. **Outlier Detection**:
   - Employs different outlier detection methods such as Isolation Forests, Local Outlier Factor, Z-score, and DBSCAN.
   - Removes detected outliers to clean the input data.

3. **Trend Removal**:
   - Attempts to remove the underlying trend from the time series data using polynomial regression.
   - Utilizes the Least Squares method to find the best-fitting polynomial coefficients.

4. **Recursive Smoothing**:
   - Implements a recursive smoothing algorithm based on the alpha-beta-gamma exponential smoothing model.
   - Optimizes the smoothing hyperparameters (alpha, beta, gamma) using gradient descent and RMSE loss function.

5. **Polynomial Regression**:
   - Determines the best-fitting polynomial order for the cleaned time series data.
   - Computes the R-squared and RMSE metrics to evaluate the model performance.
   - Generates future forecasts using the identified polynomial model.

6. **Visualization**:
   - Provides interactive plots to visualize the input data, outlier detection, trend removal, smoothing, and forecasting results.

## Dependencies

- Python 3.7 or higher
- pandas
- numpy
- matplotlib
- scikit-learn

## Usage

1. Clone the repository:
   ```
   git clone https://github.com/your-username/time-series-analysis-and-forecasting.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the main script:
   ```
   python main.py
   ```

4. Follow the on-screen prompts to select the data source, outlier cleaning method, and polynomial order.

5. Observe the generated plots and analysis results.

## Contributing

If you find any issues or have suggestions for improvements, feel free to open a new issue or submit a pull request. Contributions are always welcome!

## License

This project is licensed under the [MIT License](LICENSE).
