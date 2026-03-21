# ---------- Base image ----------
FROM python:3.11-slim

# ---------- Workdir ----------
WORKDIR /app

# ---------- Install Python deps ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Copy source ----------
COPY . .

# ---------- Cloud Run port ----------
EXPOSE 8080

# ---------- Start Streamlit ----------
CMD ["streamlit", "run", "app.py","--server.port=8080","--server.address=0.0.0.0", "--server.enableCORS=false","--server.enableXsrfProtection=false", "--server.headless=true"]
