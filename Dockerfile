# Prvi stage: Instaliraj zavisnosti u čistom okruženju
FROM python:3.9-slim as builder

# Postavljamo radni direktorijum unutar containera
WORKDIR /app

# Kopiramo fajlove za instalaciju
COPY requirements.txt .

# Instaliramo zavisnosti
RUN pip install --no-cache-dir -r requirements.txt -t /app/dependencies

# Drugi stage: Minimalan finalni image
FROM python:3.9-slim

# Postavljamo radni direktorijum
WORKDIR /app

# Kopiramo samo potrebne fajlove
COPY app.py .
COPY pricing_agent.zip .
COPY --from=builder /app/dependencies /usr/local/lib/python3.9/site-packages

# Expose port 5000
EXPOSE 5000

# Komanda za pokretanje aplikacije
CMD ["python", "app.py"]
