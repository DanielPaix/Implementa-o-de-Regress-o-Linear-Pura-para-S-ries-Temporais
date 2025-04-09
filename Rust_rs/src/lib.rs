// src/lib.rs

/// Estrutura que representa os coeficientes da regressão linear
pub struct LinearRegression {
    pub slope: f64,
    pub intercept: f64,
}

impl LinearRegression {
    /// Cria uma nova regressão linear a partir de uma série temporal
    pub fn fit(series: &[f64]) -> Option<Self> {
        let n = series.len();
        if n == 0 {
            return None;
        }

        let x_vals: Vec<f64> = (0..n).map(|x| x as f64).collect();
        let y_vals: Vec<f64> = series.to_vec();

        let mean_x = mean(&x_vals);
        let mean_y = mean(&y_vals);

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..n {
            numerator += (x_vals[i] - mean_x) * (y_vals[i] - mean_y);
            denominator += (x_vals[i] - mean_x).powi(2);
        }

        if denominator == 0.0 {
            return None;
        }

        let slope = numerator / denominator;
        let intercept = mean_y - slope * mean_x;

        Some(Self { slope, intercept })
    }

    /// Realiza uma previsão de valores futuros para `n_future` períodos
    pub fn predict(&self, last_index: usize, n_future: usize) -> Vec<f64> {
        (1..=n_future)
            .map(|i| self.slope * (last_index + i) as f64 + self.intercept)
            .collect()
    }

    /// Calcula o erro quadrático médio (MSE)
    pub fn mse(&self, series: &[f64]) -> f64 {
        let n = series.len();
        let predictions: Vec<f64> = (0..n)
            .map(|i| self.slope * i as f64 + self.intercept)
            .collect();

        predictions
            .iter()
            .zip(series.iter())
            .map(|(pred, actual)| (pred - actual).powi(2))
            .sum::<f64>()
            / n as f64
    }

    /// Calcula o coeficiente de determinação (R²)
    pub fn r_squared(&self, series: &[f64]) -> f64 {
        let mean_y = mean(series);
        let ss_tot: f64 = series.iter().map(|y| (y - mean_y).powi(2)).sum();
        let ss_res: f64 = (0..series.len())
            .map(|i| {
                let y_pred = self.slope * i as f64 + self.intercept;
                (series[i] - y_pred).powi(2)
            })
            .sum();

        1.0 - (ss_res / ss_tot)
    }
}

/// Calcula a média de um slice de f64
fn mean(data: &[f64]) -> f64 {
    let sum: f64 = data.iter().sum();
    sum / data.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fit_and_predict() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let model = LinearRegression::fit(&series).unwrap();
        assert!((model.slope - 1.0).abs() < 1e-6);
        assert!((model.intercept - 1.0).abs() < 1e-6);

        let predictions = model.predict(series.len() - 1, 2);
        assert_eq!(predictions.len(), 2);
    }

    #[test]
    fn test_mse_and_r_squared() {
        let series = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let model = LinearRegression::fit(&series).unwrap();
        let mse = model.mse(&series);
        let r2 = model.r_squared(&series);
        assert!(mse < 1e-6);
        assert!((r2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_input() {
        let series: Vec<f64> = vec![];
        assert!(LinearRegression::fit(&series).is_none());
    }
}