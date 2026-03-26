Otonom Drift Yönetimi (Prototip)

Bu klasör, geçen dönem kurduğumuz altyapının üzerine inşa edilen akıllı denetim mekanizmasını içerir.

### Teknik Özellikler:
- **Model:** Variational Autoencoder (VAE)
- **Denetim:** Input Level (Veri girişi aşamasında kontrol)
- **Metrik:** Reconstruction Error ($MSE$)
- **Otonom Yapı:** Eşik değer (Threshold) aşıldığında sistem otomatik olarak retraining başlatır.

### Sonuç:
Prototipimiz, sensör driftini saniyeler içinde yakalayıp modeli güncelleyerek operasyonel sürekliliği sağlamaktadır.
