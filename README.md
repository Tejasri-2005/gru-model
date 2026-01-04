# 🔧 PrognosAI – Remaining Useful Life (RUL) Prediction

PrognosAI is a deep learning–based predictive maintenance system that estimates the **Remaining Useful Life (RUL)** of turbofan aircraft engines using NASA’s **CMAPSS dataset**.  
The project uses **GRU (Gated Recurrent Unit)** neural networks to detect engine degradation and generate **maintenance alerts**.



## 📌 Project Overview

Predictive maintenance helps identify failures before they occur, reducing downtime and maintenance costs.  
This project:

- Predicts Remaining Useful Life (RUL) for each engine
- Classifies engines as **Healthy**, **Warning**, or **Critical**
- Generates actionable maintenance recommendations
- Supports all four CMAPSS datasets (**FD001–FD004**)
- Includes an interactive **Streamlit dashboard**

---

## 📊 Dataset

- **Source:** NASA CMAPSS Turbofan Engine Degradation Dataset

### Dataset Domains
- **FD001:** Single operating condition, single fault
- **FD002:** Multiple operating conditions, single fault
- **FD003:** Single operating condition, multiple faults
- **FD004:** Multiple operating conditions, multiple faults

Each dataset contains:
- Engine unit number
- Time cycles
- Operational settings
- Sensor measurements
- True RUL values for test data



## 🧠 Model Architecture

- Model Type: **GRU (Gated Recurrent Unit)**
- Sequence Length: **50 cycles**
- Output: Predicted Remaining Useful Life
- Loss Function: **Mean Squared Error (MSE)**
- Metrics: **RMSE, MAE**

Each dataset has:
- A dedicated trained GRU model
- A dataset-specific feature scaler




## 🖥️ Streamlit Application Features

- Dataset selection (FD001–FD004)
- Upload test and RUL files
- RUL prediction per engine
- Fleet health visualization
- Maintenance alert generation
- Downloadable CSV reports
- Interactive Plotly and Matplotlib charts

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
▶️ Run the Application
streamlit run project.py

Open the URL shown in the terminal (usually http://localhost:8501).




