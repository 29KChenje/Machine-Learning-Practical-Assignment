# Student Dropout Prediction Model

A machine learning model built with Python and Streamlit to predict student dropout risk based on various academic and personal factors.

## Links

- GitHub repository: <https://github.com/29KChenje/Machine-Learning-Practical-Assignment>
- Live Streamlit app: <https://machine-learning-practical-assignment.streamlit.app>
- Deploy on Streamlit Cloud: <https://share.streamlit.io/deploy?repository=https://github.com/29KChenje/Machine-Learning-Practical-Assignment&branch=main&mainModule=app.py>

## Features

- **Student Dropout Prediction**: Predicts whether a student is likely to dropout based on input parameters
- **Interactive Web Interface**: Built with Streamlit for easy use
- **Fast Cloud Startup**: Uses the trained model values directly in the app

## Installation

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/29KChenje/Machine-Learning-Practical-Assignment.git
cd Machine-Learning-Practical-Assignment
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the app locally:
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Deployment on Streamlit Cloud

1. Go to [Streamlit Cloud](https://share.streamlit.io) or open the deploy link above
2. Sign in with your GitHub account
3. Click **"New app"**
4. Configure:
   - **Repository**: `29KChenje/Machine-Learning-Practical-Assignment`
   - **Branch**: `main`
   - **Main file**: `app.py`
5. Click **"Deploy"**

After deployment, Streamlit Cloud will give you the public app URL.

## Model Features

The model uses the following features to make predictions:
- Age
- Family Income
- Transport Time
- Study Hours
- Attendance Rate
- LMS Logins
- Stress Level

## Files

- `app.py` - Main Streamlit application
- `cleaned_dataset.csv` - Training dataset used during model development
- `requirements.txt` - Python dependencies
- `runtime.txt` - Python version specification
- `.streamlit/config.toml` - Streamlit configuration

## Technologies

- **Python 3.11**
- **Streamlit** - Web framework
- **Scikit-learn** - Machine learning
- **Pandas** - Data processing
- **NumPy** - Numerical computing

## Author

Chenje Raikos Kosmas

## License

MIT
