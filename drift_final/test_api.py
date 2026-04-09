import unittest
import os

class TestMLOpsProject(unittest.TestCase):
    
    def test_model_exists(self):
        """1. KONTROL: Eğitilmiş model dosyası var mı?"""
        # Senin model isminle kontrol ediyoruz
        model_file = 'model_adwin_final.pkl'
        print(f">>> Test 1: {model_file} aranıyor...")
        self.assertTrue(os.path.exists(model_file), "HATA: Model dosyası (pkl) bulunamadı!")
        
    def test_dockerfile_exists(self):
        """2. KONTROL: Dockerfile var mı?"""
        print(">>> Test 2: Dockerfile kontrol ediliyor...")
        self.assertTrue(os.path.exists('Dockerfile'), "HATA: Dockerfile eksik!")

    def test_requirements_exists(self):
        """3. KONTROL: Kütüphane listesi var mı?"""
        print(">>> Test 3: requirements.txt kontrol ediliyor...")
        self.assertTrue(os.path.exists('requirements.txt'), "HATA: requirements.txt eksik!")

if __name__ == '__main__':
    print("--- MLOps Otomatik Testleri Başlatılıyor ---")
    unittest.main()
