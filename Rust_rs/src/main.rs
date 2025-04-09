mod lib;

use lib::LinearRegression;

fn main() {
    let data = vec![2.0, 4.0, 6.0, 8.0, 10.0];

    let model = LinearRegression::fit(&data).expect("Falha ao ajustar o modelo");

    println!("Coeficiente angular (slope): {}", model.slope);
    println!("Intercepto: {}", model.intercept);

    let preds = model.predict(data.len() - 1, 3); // por exemplo, 3 passos futuros
    println!("Previsões: {:?}", preds);

    let r2 = model.r_squared(&data);
    let mse = model.mse(&data);

    println!("R²: {:.4}", r2);
    println!("MSE: {:.4}", mse);
}

