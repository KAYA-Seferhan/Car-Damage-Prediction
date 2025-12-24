# CNN Tabanlı Araç Hasar Tespiti

Bu projede, araç görüntülerinden **hasarlı / hasarsız** durumunun otomatik olarak tespit edilmesini sağlayan **derin öğrenme tabanlı bir görüntü sınıflandırma sistemi** geliştirilmiştir.  
Proje, **Derin Öğrenme ve Bilgisayarla Görme** alanında akademik bir çalışma olarak hazırlanmıştır.

---

## Proje Özeti

Araç hasar tespiti süreçleri çoğunlukla manuel olarak yürütülmekte ve zaman alıcı olabilmektedir. Bu proje, bu süreci otomatikleştirerek araç görüntülerinin **hasarlı** veya **hasarsız** olarak sınıflandırılmasını amaçlamaktadır.

Model, **Convolutional Neural Network (CNN)** mimarisi kullanılarak **sıfırdan eğitilmiştir** ve herhangi bir hazır veya önceden eğitilmiş model kullanılmamıştır.

---

## Kullanılan Yöntem

- Model Türü: Convolutional Neural Network (CNN)
- Yaklaşım: İkili sınıflandırma (Hasarlı / Hasarsız)
- Kayıp Fonksiyonu: Binary Cross Entropy (BCE)
- Optimizasyon Algoritması: AdamW
- Aktivasyon Fonksiyonu: ReLU

Model performansı aşağıdaki metriklerle değerlendirilmiştir:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Ayrıca, modelin hasarlı araçları kaçırmaması amacıyla **karar eşiği (threshold) optimizasyonu** uygulanmıştır.

---

## Veri Seti Yapısı

Veri seti `ImageFolder` formatına uygun şekilde düzenlenmiştir:

```text
dataset/
 ├── train/
 │    ├── damaged/
 │    └── undamaged/
 ├── val/
 │    ├── damaged/
 │    └── undamaged/
 └── test/
      ├── damaged/
      └── undamaged/
```

---

## Model Eğitimi

Modeli eğitmek için:

```bash
python train.py
```

---

## Web Arayüzü (Gradio)

```bash
python serve.py
```
