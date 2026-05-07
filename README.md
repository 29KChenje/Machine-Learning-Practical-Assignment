# Student Dropout Prediction Model

A machine learning model built with Python and Streamlit to predict student dropout risk based on various academic and personal factors.

## Features

- **Student Dropout Prediction**: Predicts whether a student is likely to dropout based on input parameters
- **Interactive Web Interface**: Built with Streamlit for easy use
- **Real-time Model Training**: Trains on the provided dataset at runtime

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

1. Go to [Streamlit Cloud](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Configure:
   - **Repository**: `29KChenje/Machine-Learning-Practical-Assignment`
   - **Branch**: `main`
   - **Main file**: `app.py`
5. Click **"Deploy"**

Your app will be live at: `https://share.streamlit.io/29KChenje/Machine-Learning-Practical-Assignment/main/app.py`

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
- `cleaned_dataset.csv` - Training dataset
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
