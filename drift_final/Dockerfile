# 1. Python'un hafif bir sürümünü temel al
FROM python:3.9-slim

# 2. Çalışma klasörünü ayarla
WORKDIR /app

# 3. Kütüphane listesini kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Kodların tamamını (app.py, model vb.) içeri kopyala
COPY . .

# 5. Dış dünyaya 5000 portunu aç (Flask buradan yayın yapacak)
EXPOSE 5000

# 6. Konteyner başladığında API'yi çalıştır
CMD ["python", "app.py"]