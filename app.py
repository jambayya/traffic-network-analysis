from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle file upload and data processing
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Load the dataset
    data = pd.read_csv(file)

    # Feature columns and target
    features = ['num_failed_logins', 'hot', 'num_access_files', 'attack_type']
    target = 'label'

    # Extract features and target
    X = data[features]
    y = data[target]

    # Calculate the sum of features
    column_sums = X.sum().to_dict()

    # Bar plot for feature sums
    bar_plot = io.BytesIO()
    plt.figure(figsize=(8, 5))
    pd.Series(column_sums).plot(kind='bar', color=['blue', 'green', 'orange', 'red'])
    plt.title('Sum of Features')
    plt.ylabel('Sum')
    plt.xlabel('Feature')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(bar_plot, format='png')
    plt.close()
    bar_plot.seek(0)
    bar_plot_base64 = base64.b64encode(bar_plot.getvalue()).decode()

    # Heatmap for correlations
    heatmap = io.BytesIO()
    plt.figure(figsize=(10, 6))
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(heatmap, format='png')
    plt.close()
    heatmap.seek(0)
    heatmap_base64 = base64.b64encode(heatmap.getvalue()).decode()

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Feature importance plot
    importance_plot = io.BytesIO()
    feature_importances = pd.Series(model.feature_importances_, index=features)
    feature_importances.sort_values().plot(kind='barh', color='teal')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(importance_plot, format='png')
    plt.close()
    importance_plot.seek(0)
    importance_plot_base64 = base64.b64encode(importance_plot.getvalue()).decode()

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save the model
    joblib.dump(model, 'model/random_forest_model.pkl')

    # Return results to the UI
    return jsonify({
        'accuracy': f'{accuracy * 100:.2f}%',
        'bar_plot': bar_plot_base64,
        'heatmap': heatmap_base64,
        'importance_plot': importance_plot_base64,
        'data_preview': data.head().to_html(classes='table table-bordered', index=False),
        'column_sums': column_sums,
    })

if __name__ == '__main__':
    app.run(debug=True)
