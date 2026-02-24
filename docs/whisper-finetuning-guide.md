# Whisper Fine-Tuning ile Kuran Tilaveti Tanima

**Tilavet Projesi — Kapsamli Egitim Dokumani**

> Bu dokuman, OpenAI Whisper modelini LoRA ile fine-tune ederek Kuran-i Kerim
> tilavetini tanitan sistemi A'dan Z'ye anlatir. AYBU Teknik Bilimler MYO
> ogrencileri icin hazirlanmistir.
>
> **Yazar:** Tarik Ismet ALKAN — AYBU Teknik Bilimler MYO

---

## Icindekiler

1. [Giris ve Motivasyon](#1-giris-ve-motivasyon)
2. [Platformlar ve Araclar](#2-platformlar-ve-araclar)
3. [Temel Kavramlar](#3-temel-kavramlar)
4. [Veri Seti Analizi](#4-veri-seti-analizi)
5. [Model Secimi ve Konfigurasyon](#5-model-secimi-ve-konfigurasyon)
6. [Egitim Sureci (Adim Adim)](#6-egitim-sureci-adim-adim)
7. [Karsilasilan 17 Hata ve Cozumleri](#7-karsilasilan-17-hata-ve-cozumleri)
8. [Alternatif Yaklasimlar](#8-alternatif-yaklasimlar)
9. [Performans ve Sonuclar](#9-performans-ve-sonuclar)
10. [Uretim Ortamina Gecis](#10-uretim-ortamina-gecis)
11. [Maliyet Analizi](#11-maliyet-analizi)
12. [Sozluk (Glossary)](#12-sozluk-glossary)
13. [Pratik Egzersizler](#13-pratik-egzersizler)
14. [Sik Sorulan Sorular (FAQ)](#14-sik-sorulan-sorular-faq)
15. [Kaynaklar ve Ileri Okuma](#15-kaynaklar-ve-ileri-okuma)
16. [Sonuc ve Oneriler](#16-sonuc-ve-oneriler)

---

## 1. Giris ve Motivasyon

### Proje: Tilavet

Tilavet, acik kaynakli bir Kuran-i Kerim konusmadan metne (Speech-to-Text / STT) ve
ceviri platformudur. Amacimiz:

- Kuran tilavetini dinleyip, hangi ayetin okundugunu otomatik olarak tanimak
- 108 dilde ceviri sunmak
- Gelistiriciler icin acik API saglamak

### Neden Onemli?

- **1.8 milyar Musluman** dunyada Kuran okur ve dinler
- Mevcut cozumler ya kapali kaynakli ya da yetersiz kalitede
- Erisilebilirlik: Gorme engelli kullanicilar icin ses tabanli Kuran erisimi
- Egitim: Kuran okumayi ogrenenler icin anlık geri bildirim

### Hedef: WER < %3

Piyasadaki en iyi sistem Tarteel AI, %5.75 WER degerine sahip. Bizim hedefimiz
bunu gecmek ve %3'un altina inmek. Bu dokumanda, bu hedefe nasil ulastigimizi
adim adim anlatacagiz.

> **Not:** WER (Word Error Rate) konusunu Bolum 3'te detayli aciklayacagiz.

---

## 2. Platformlar ve Araclar

### 2.1 HuggingFace

HuggingFace, yapay zeka topluluginin GitHub'i gibidir. Model paylasimi, dataset
barindirma ve egitim araclari sunar.

**HuggingFace Hub** — https://huggingface.co

Model ve dataset deposudur. Herkes modelini yukleyebilir, indirebilir ve kullanabilir.

- **500.000+** model barindiriyor
- **100.000+** dataset mevcut
- Ucretsiz hesapla kullanilabilir
- Model kartlari: Her modelin ne yaptigi, nasil kullanilacagi, performans metrikleri

Biz su modeli kullaniyoruz:
```
https://huggingface.co/openai/whisper-large-v3
```

Ve su dataset'i:
```
https://huggingface.co/datasets/tarteel-ai/everyayah
```

**HuggingFace Kutuphaneleri:**

| Kutuphane | Ne Ise Yarar | Biz Nasil Kullandik |
|---|---|---|
| `transformers` | Model yukleme, egitim, inference | Whisper modelini yukleme ve egitme |
| `datasets` | Dataset yukleme ve isleme | everyayah dataset'ini streaming ile yukleme |
| `peft` | Parameter-efficient fine-tuning (LoRA) | LoRA adapter'i ekleme |
| `evaluate` | Metrik hesaplama | WER olcumu |
| `accelerate` | Coklu GPU, mixed precision | fp16 egitim hizlandirma |
| `huggingface_hub` | Hub'a model yukleme/indirme | Login ve model paylasimi |
| `tokenizers` | Metin tokenizasyonu | Whisper tokenizer (dahili) |

**Kurulum:**
```python
pip install transformers peft datasets accelerate evaluate huggingface_hub
```

**HuggingFace Login:**
```python
from huggingface_hub import login
login()  # Token girmenizi ister
```

> **Neden login gerekli?** Bazi dataset'ler (tarteel-ai/everyayah dahil) login
> gerektirmektedir. Ayrica egitilen modeli Hub'a yuklemek icin de login sart.
> https://huggingface.co/settings/tokens adresinden token alinabilir.

**HuggingFace Spaces:**
Spaces, web uygulamalari barindirma servisidir. Gradio veya Streamlit ile demo
olusturabilirsiniz. Ornegin, egittigimiz modelle canli bir "Tilavet Tanima" demosu
yapilabilir.

### 2.2 Google Colab

Google Colab, tarayicida calisan bulut tabanli Jupyter notebook ortamidir.

**Ucretsiz vs Pro vs Pro+:**

| Ozellik | Ucretsiz | Pro ($10/ay) | Pro+ ($50/ay) |
|---|---|---|---|
| GPU | T4 (15GB) | T4, V100 (16GB) | T4, V100, **A100 (80GB)** |
| RAM | 12 GB | 25 GB | **83 GB** |
| Calisma suresi | ~4 saat | ~12 saat | ~24 saat |
| Oncelik | Dusuk | Yuksek | En yuksek |

**Biz neden Pro+ kullandik?**
- Whisper-large-v3 modeli (1.5B parametre) 6+ GB GPU bellegi gerektiriyor
- LoRA egitimi sirasinda ~20 GB GPU bellegi kullanildi
- A100 80 GB, yeterli alani rahatca sagliyor
- 24 saatlik calisma suresi, ~12-13 saatlik egitimimize yeterli

**GPU Tipleri:**

| GPU | VRAM | FP16 TFLOPS | Fiyat (Colab) |
|---|---|---|---|
| T4 | 16 GB | 65 | Ucretsiz/Pro |
| V100 | 16 GB | 125 | Pro |
| A100 40GB | 40 GB | 312 | Pro+ |
| **A100 80GB** | **80 GB** | **312** | **Pro+** |

> **T4 vs A100:** A100 yaklasik 5 kat daha hizli. T4'te 60+ saat surecek egitim,
> A100'de ~12-13 saat suruyor.

**Google Drive Entegrasyonu:**
```python
from google.colab import drive
drive.mount("/content/drive")
```
Bu sayede checkpoint'lar Drive'a kaydedilir ve runtime yeniden basladiginda
egitim kaldigi yerden devam eder.

### 2.3 TensorBoard

TensorBoard, egitim metriklerini gorsel olarak takip etmeye yarayan bir aractir.
Google tarafindan gelistirilmistir.

- **Loss egrisi:** Egitimin ogreniyor mu, yoksa takildi mi?
- **WER egrisi:** Kalite nasil ilerliyor?
- **Learning rate:** Scheduler duzgun calisiyor mu?

Notebook'ta su sekilde aktif edilir:
```python
report_to="tensorboard"
```

Colab'de goruntuleme:
```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/tilavet/whisper-large-v3-quran-lora/runs
```

---

## 3. Temel Kavramlar

### 3.1 Whisper Nedir?

Whisper, OpenAI tarafindan gelistirilen cok dilli bir ses tanima (ASR / Automatic
Speech Recognition) modelidir.

**Temel Ozellikler:**
- 680.000 saat ses verisiyle egitilmis
- 99 dil destekliyor (Arapca dahil)
- Hem ses tanima hem ceviri yapabilir
- Encoder-decoder mimarisi kullanir

**Mimari: Encoder-Decoder**

```
Ses Dosyasi
    |
    v
[Mel Spectrogram] -----> [ENCODER] -----> [DECODER] -----> Metin Ciktisi
   (ses -> gorsel)        (anlamlandirma)   (metin uretimi)
```

1. **Mel Spectrogram:** Ses dalgasi sayisal bir "resme" donusturulur. 80 frekans
   bandinda, her 10ms'de bir deger uretilir. Sonuc: 80 x 3000 boyutunda bir matris
   (30 saniyelik ses icin).

2. **Encoder:** Bu matrisi isler ve "anlam vektorleri" olusturur. Dikkat
   mekanizmasi (attention) ile sesin hangi kisimlarin onemli oldugunu ogenir.

3. **Decoder:** Encoder'in ciktisini alir ve kelime kelime metin uretir. Her
   adimda bir onceki kelimeye bakarak sirali tahmin yapar (autoregressive).

**Whisper Ailesi:**

| Model | Parametreler | VRAM (fp32) | Relatif Hiz | Ingilizce WER | Cok Dilli WER |
|---|---|---|---|---|---|
| tiny | 39M | ~1 GB | 32x | ~7.6% | ~14% |
| base | 74M | ~1 GB | 16x | ~5.0% | ~11% |
| small | 244M | ~2 GB | 6x | ~3.4% | ~8% |
| medium | 769M | ~5 GB | 2x | ~2.7% | ~6% |
| large | 1.5B | ~10 GB | 1x | ~2.2% | ~5% |
| large-v2 | 1.5B | ~10 GB | 1x | ~2.1% | ~4.5% |
| **large-v3** | **1.5B** | **~10 GB** | **1x** | **~1.8%** | **~4%** |

> **Biz neden large-v3 sectik?** En dusuk WER degerine sahip. Buyuk model,
> daha fazla bilgi tutar ve fine-tuning'de daha iyi sonuc verir. Dezavantaji:
> daha fazla GPU bellegi ve egitim suresi gerektirir.

**Desteklenen Diller (99):**
Arapca, Turkce, Ingilizce, Farsca, Urduca, Endonezce, Malayca ve 90+ dil.
Whisper ozellikle Arapca'da guclu bir baz performansa sahiptir.

### 3.2 Transfer Learning

Transfer learning, bir gorevi ogrenmis bir modeli alip baska bir goreve uyarlamaktir.

```
ADIM 1 — Pre-training (OpenAI yapti):
  680.000 saat genel ses verisi -> Whisper-large-v3 (genel ASR modeli)

ADIM 2 — Fine-tuning (biz yapiyoruz):
  234.000 Kuran tilaveti ornegi -> Tilavet modeli (Kuran'a ozel ASR)
```

**Neden sifirdan egitmiyoruz?**
- Sifirdan egitim icin yuz binlerce GPU-saat gerekir (milyonlarca dolar)
- Whisper zaten Arapca biliyor, sadece Kuran tanitlara "ozellestirilmesi" gerekiyor
- Fine-tuning cok daha hizli: ~12-13 saat, ~$35-40

**Domain Adaptation:**
Whisper genel konusmada iyidir ama Kuran tilaveti ozel bir alandir:
- Tecvid kurallari (uzatmalar, idgam, ihfa vs.)
- Harekeli Arapca (gunluk konusmada harekeler soylenmiyor)
- Ozel ses tonlari ve makamlar

Fine-tuning, modeli bu ozelliklere adapte eder.

### 3.3 LoRA (Low-Rank Adaptation)

LoRA, buyuk modelleri verimli sekilde fine-tune etmenin yoludur.

**Problem:** Whisper-large-v3'un 1.5 milyar parametresi var. Hepsini egitmek
(full fine-tuning) icin 80+ GB GPU bellegi gerekir ve cok yavastir.

**Cozum:** LoRA, modelin sadece kucuk bir kismini egitir.

**Nasil Calisiyor?**

Normal bir katman (layer) su islemi yapar:
```
y = W * x    (W: buyuk bir agirlik matrisi, ornegin 1024 x 1024)
```

LoRA bu katmana kucuk bir "ek" koyar:
```
y = W * x + (B * A) * x

Burada:
  W  = orijinal matris (1024 x 1024) — DONDURULMUS, egitilmez
  A  = kucuk matris (1024 x r)       — EGITILIR
  B  = kucuk matris (r x 1024)       — EGITILIR
  r  = rank (bizde r=32)
```

**Neden "Low-Rank"?**
- Orijinal W: 1024 x 1024 = 1.048.576 parametre
- LoRA (r=32): (1024 x 32) + (32 x 1024) = 65.536 parametre
- **16 kat daha az parametre!**

**Parametreler:**

| Parametre | Deger | Aciklama |
|---|---|---|
| `r` | 32 | Rank. Buyukse daha cok ogrenir, daha cok bellek |
| `alpha` | 64 | Olcekleme faktoru. Genelde alpha = 2*r |
| `dropout` | 0.05 | Regularizasyon. Overfitting'i onler |
| `target_modules` | 6 modul | Hangi katmanlar LoRA alacak |

**Target Modules:**
```python
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
```

- `q_proj`, `k_proj`, `v_proj`: Attention mekanizmasinin query, key, value projeksiyonlari
- `out_proj`: Attention ciktisi
- `fc1`, `fc2`: Feed-forward network katmanlari

> **Neden bu 6 modul?** Attention katmanlari, modelin "neye dikkat ettigi"ni
> belirler. Kuran tilavetine ozellestirilmis dikkat oruntuleri ogrenmesi icin
> bu katmanlar secildi. fc1 ve fc2, her attention blogunun ardindan gelen
> feed-forward katmanlaridir ve ek kapasite saglar.

**Sonuc:**
```
Toplam parametreler: 1,543,534,592  (1.5B)
Egitilen parametreler: 13,107,200    (~13M)
Egitilen oran: %0.85
```

Yani modelin sadece %0.85'ini egitiyoruz, geri kalan %99.15 donuk kaliyor.
Bu, 80 GB GPU ile rahatca yapilabilir.

**r Degeri Farklı Olsaydi:**

| r | Egitilen Param | GPU Bellek | Kalite |
|---|---|---|---|
| 8 | ~3.3M | ~15 GB | Yeterli olmayabilir |
| 16 | ~6.6M | ~17 GB | Orta |
| **32** | **~13M** | **~20 GB** | **Iyi (biz bunu sectik)** |
| 64 | ~26M | ~25 GB | Daha iyi ama daha yavas |

### 3.4 QLoRA (Quantized LoRA)

QLoRA, modeli 4-bit'e quantize edip LoRA egitimi yapmayi saglayan bir tekniktir.

**Quantization Nedir?**
Normal bir model fp32 (32-bit) ile saklanir. Her parametre 4 byte yer kaplar.
Quantization, bunu azaltir:

| Format | Bit | Byte/Param | 1.5B Model Boyutu |
|---|---|---|---|
| fp32 | 32 | 4 | ~6 GB |
| fp16 | 16 | 2 | ~3 GB |
| 8-bit | 8 | 1 | ~1.5 GB |
| **4-bit (NF4)** | **4** | **0.5** | **~0.75 GB** |

**NF4 (Normal Float 4):**
QLoRA'nin kullandigi ozel 4-bit formati. Normal dagılıma optimize edilmis
4-bit temsil. Bilgi kaybi minimumda tutulur.

**Neden Biz QLoRA Kullanmadik?**

Denedik ve su hatayi aldik:
```
RuntimeError: Input type (float) and bias type (c10::Half) should be the same
  File "torch/nn/modules/conv.py", line 459, in _conv_forward
```

**Sorun:** Whisper'in ilk katmani `Conv1d`'dir. Bu katman ses sinyalini
(mel spectrogram) isler. QLoRA modeli fp16'ya cevirdikten sonra, Conv1d'nin
bias'i da fp16 oldu. Ancak giris mel spectrogram'i hala fp32. Bu uyumsuzluk
PyTorch'ta hata veriyor.

**Neden cozmek yerine vazgectik?**
- fp32 modelle zaten 20/80 GB GPU kullaniyoruz (rahat)
- QLoRA'nin Conv1d uyumsuzlugu risk tasior
- Kalite bizim icin hizdan daha onemli
- fp32 hassasiyet, 4-bit'ten daha iyi sonuc verir

> **Ders:** Her optimizasyon her modelde calismaz. Whisper'in Conv1d'si QLoRA
> ile sorunludur. T4 gibi kucuk GPU'larda calismaniz gerekirse bu sorunu
> cozmek sart olur, ancak A100 80GB varsa fp32 ile devam etmek daha guvenli.

### 3.5 Gradient Checkpointing

Gradient checkpointing, GPU bellegini azaltmanin en etkili yontemlerinden biridir.

**Normal Egitim:**
```
Ileri gecis (forward pass):
  Girdi -> Katman1 -> Katman2 -> ... -> Katman32 -> Kayip

  Her katmanin ciktisi (aktivasyonlar) bellekte TUTULUR
  Neden? Geri gecis (backward pass) icin gerekli
  Sorun: 32 katmanin aktivasyonlari cok yer kaplar (~30-40 GB)
```

**Gradient Checkpointing ile:**
```
Ileri gecis:
  Girdi -> Katman1 -> [SIL] -> Katman2 -> [SIL] -> ... -> Kayip

  Aktivasyonlar hemen silinir (bellek tasarrufu)

Geri gecis:
  Gerekli aktivasyonlar YENIDEN HESAPLANIR (tekrar ileri gecis)
  ~%30 daha yavas ama ~%60 daha az bellek
```

**Bizim durumumuz:**
- Gradient checkpointing **OLMADAN:** ~50+ GB GPU (A100 bile yetmez, OOM)
- Gradient checkpointing **ILE:** ~20 GB GPU (A100'un 80 GB'inin %25'i)

```python
gradient_checkpointing=True  # TrainingArguments'ta
```

> **Trade-off:** %30 daha yavas egitim, ama %60 daha az bellek kullanimi.
> A100 80 GB'da bu olmazsa olmaz bir ayar.

### 3.6 Mixed Precision Training (fp16)

Normal hesaplamalar fp32 (32-bit) ile yapilir. fp16 (16-bit) kullanmak:
- **2x daha az bellek** kullanir
- **A100'un Tensor Core'larini** aktive eder (buyuk hiz artisi)
- Kalite kaybi ihmal edilebilir (loss scaling ile onlenir)

**Loss Scaling:**
fp16'da cok kucuk sayilar sifira yuvarlanabilir (underflow). Loss scaling,
kayip degerini buyuk bir sayi ile carpar, geri gecisten sonra tekrar boler.
Bu PyTorch'ta otomatik yapilir (`fp16=True`).

```python
fp16=True,           # Egitimde fp16
fp16_full_eval=True, # Eval'da da fp16 (bellek tasarrufu)
```

> **fp16_full_eval olmadan ne olur?** Eval sirasinda model fp32'ye donusturulur.
> Bu 80 GB bellegin tamamini doldurabilir ve OOM hatasina yol acar.
> Biz bu hatayi yasadik (Hata #15).

### 3.7 Degerlendirme Metrikleri

#### WER (Word Error Rate) — Ana Metrik

WER, ses tanima sistemlerinin standart degerlendirme metrigidir.

**Formul:**
```
WER = (S + D + I) / N

S = Substitution (Degistirme) — Yanlis kelime
D = Deletion (Silme)          — Eksik kelime
I = Insertion (Ekleme)        — Fazla kelime
N = Toplam kelime sayisi (referansta)
```

**Ornek Hesaplama:**

```
Referans:    بسم    الله    الرحمن    الرحيم
Tahmin:      بسم    الهه    الرحمن

Karsilastirma:
  بسم     -> بسم     (DOGRU)
  الله     -> الهه     (S — yanlis kelime)
  الرحمن   -> الرحمن   (DOGRU)
  الرحيم   -> ---      (D — eksik kelime)

S = 1, D = 1, I = 0, N = 4
WER = (1 + 1 + 0) / 4 = %50
```

**WER Degerleri Ne Anlama Gelir:**

| WER | Anlam |
|---|---|
| %0 | Mukemmel — hic hata yok |
| %1-3 | Cok iyi — profesyonel duzey |
| %5-10 | Iyi — kullanilabilir |
| %10-20 | Orta — iyilestirme gerekir |
| %20+ | Kotu — ciddi sorunlar var |
| %100 | Hicbir kelime dogru degil |

**Bizim Hedefimiz:** WER < %3 (Tarteel AI: %5.75)

#### Harekeli vs Harekesiz WER

Arapca'da harekeler (fatha, damma, kasra vs.) kelimelerin telaffuzunu belirler.

**Harekeli (diacritized) WER:**
```
Referans: بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ
Tahmin:   بِسْمِ اللّهِ الرَّحْمَنِ الرَّحِيمِ
                  ^^ (shadda eksik)

Bu bile hata sayilir. Cok katı bir metrik.
```

**Harekesiz (normalized) WER:**
```
Referans: بسم الله الرحمن الرحيم  (harekeler silindi)
Tahmin:   بسم الله الرحمن الرحيم  (harekeler silindi)

Mukemmel eslesme. Sadece harflere bakilir.
```

**Neden ikisini birden olcuyoruz?**
- `wer_diacritized`: Tam dogruluk — hareke hatalari dahil
- `wer_normalized`: Pratik dogruluk — ana metrigimiz

> Ana metrik `wer_normalized` cunku kullanicilar genellikle hareke hassasiyeti
> istemez, ayetin dogru taninmasi yeterlidir.

**Hareke Temizleme Kodu:**
```python
import re

def strip_harakat(text: str) -> str:
    """Arapca harekeleri kaldir."""
    return re.sub(r'[\u064B-\u065F\u0670]', '', text).strip()

# Ornek
harekeli = "بِسْمِ اللَّهِ"
harekesiz = strip_harakat(harekeli)  # "بسم الله"
```

**Unicode Hareke Araliklari:**

| Hareke | Unicode | Aciklama |
|---|---|---|
| Fatha | U+064E | Ustteki kisa cizgi (a sesi) |
| Damma | U+064F | Ustteki kucuk vav (u sesi) |
| Kasra | U+0650 | Alttaki kisa cizgi (i sesi) |
| Sukun | U+0652 | Ustteki kucuk daire (sessiz) |
| Shadda | U+0651 | Ustteki W sekli (cift uretim) |
| Tanwin Fatha | U+064B | Ustteki cift cizgi (-en) |
| Tanwin Damma | U+064C | Ustteki cift vav (-un) |
| Tanwin Kasra | U+064D | Alttaki cift cizgi (-in) |

#### CER (Character Error Rate)

WER'in karakter bazli versiyonu. Kelime yerine karakter karsilastirir.
Arapca gibi bitisik yazilan dillerde WER'den daha ince ayrim yapabilir.

```
WER = kelime bazli hata orani
CER = karakter bazli hata orani
```

Biz ana metrik olarak WER kullaniyoruz cunku STT alaninda standart WER'dir.

#### BLEU ve METEOR — Neden Kullanmadik?

BLEU ve METEOR, **makine cevirisi** icin tasarlanmis metriklerdir. STT icin
uygun degildir cunku:
- STT'de tek bir "dogru cevap" vardir (orijinal metin)
- Ceviri'de birden fazla dogru cevap olabilir
- WER, STT icin standart ve en bilgilendirici metriktir

#### jiwer Kutuphanesi

WER hesaplamak icin Python kutuphanesi:
```python
pip install jiwer

from jiwer import wer
error_rate = wer("referans metin", "tahmin metin")
print(f"WER: {error_rate * 100:.2f}%")
```

#### HuggingFace evaluate Kutuphanesi

HuggingFace'in metrik ekosistemi. Birden fazla metrigi standart API ile kullanmayi saglar:
```python
import evaluate
wer_metric = evaluate.load("wer")
result = wer_metric.compute(predictions=["tahmin"], references=["referans"])
```

### 3.8 SDPA (Scaled Dot-Product Attention)

SDPA, PyTorch 2.1+ ile gelen optimize edilmis attention uygulamasidir.

**Standart Attention:**
```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d)) * V
```

Bu islem normalde Python/CUDA'da adim adim yapilir. Yavastir.

**SDPA ile:**
```python
torch.nn.functional.scaled_dot_product_attention(Q, K, V)
```

Tek bir PyTorch fonksiyon cagrisinda yapilir. GPU kernel seviyesinde
optimize edilmistir. **%20-40 hiz artisi** saglar.

```python
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3",
    attn_implementation="sdpa",  # <-- bu satir
)
```

**Flash Attention 2 Nedir?**
Flash Attention 2, SDPA'nin daha da hizli bir versiyonudur. Ancak **Whisper'da
calismaz** cunku Whisper'in cross-attention yapisi Flash Attention 2'nin
beklentileriyle uyumlu degildir.

> **A100 Tensor Core:** A100'un donanim hizlandirma birimleri SDPA ile
> otomatik olarak kullanilir. Ayrica fp16 ile Tensor Core'lar aktif olur
> ve matris carpimlarinda buyuk hiz artisi saglar.

---

## 4. Veri Seti Analizi

### 4.1 tarteel-ai/everyayah

| Ozellik | Deger |
|---|---|
| **Kaynak** | https://huggingface.co/datasets/tarteel-ai/everyayah |
| **Boyut** | ~234.000 ornek |
| **Okuyucu** | 37 farkli kari (Kuran okuyucusu) |
| **Lisans** | MIT (acik kaynak, ticari kullanim serbest) |
| **Format** | Audio (WAV/MP3) + Metin (harekeli Arapca) |
| **Split** | Train / Test |

**Neden bu dataset?**
- Kuran tilavetine ozel en buyuk acik dataset
- 37 farkli kari = farkli ses tonlari ve okuyus usluplari
- Harekeli metin = hassas degerlendirme mumkun
- MIT lisans = ozgurce kullanilabilir

**HuggingFace'te Nasil Bulunur?**
1. https://huggingface.co/datasets adresine git
2. "quran" veya "everyayah" ara
3. Dataset kartini oku (boyut, lisans, ornek sayisi)
4. "Use in datasets" butonundan kod ornegini al

**Yukleme:**
```python
from datasets import load_dataset

# Streaming: disk kullanmaz, internet uzerinden okur
dataset = load_dataset("tarteel-ai/everyayah", split="train", streaming=True)

# Normal: diske indirir (onbellek)
dataset = load_dataset("tarteel-ai/everyayah", split="train")
```

**Her Ornek Iceriyor:**
```python
{
    "audio": {
        "array": [0.012, -0.003, 0.008, ...],  # Ses dalgasi (float array)
        "sampling_rate": 16000                   # 16 kHz
    },
    "text": "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ"  # Harekeli Arapca
}
```

### 4.2 Streaming vs Download

| | Streaming | Download |
|---|---|---|
| **Disk** | 0 GB | ~50+ GB |
| **Baslama suresi** | Aninda | Uzun indirme |
| **Tekrar erisilir mi?** | Her epoch yeniden okunur | Diskte kalir |
| **Risk** | Internet kopma / timeout | Disk dolu olabilir |
| **Shuffle** | Buffer ile (yaklasik) | Tam karıstirma |

**Biz neden streaming sectik?**
- Colab'da disk alani sinirli (~100 GB, model + checkpoint'lar yer kaplar)
- 234K ornek icin indirme cok uzun surebilir
- Streaming'de `buffer_size=10_000` ile yeterli karistirma yapilir

```python
train_dataset = (
    train_stream
    .filter(lambda x: len(x["audio"]["array"]) / x["audio"]["sampling_rate"] <= 30.0)
    .map(speed_perturb)
    .map(prepare_sample, remove_columns=["audio", "text"])
    .shuffle(buffer_size=10_000, seed=42)
)
```

**Timeout Riskleri:**
HuggingFace CDN'den baglanti kopma yaşanabilir:
```
ProtocolError: RemoteDisconnected('Remote end closed connection')
```
Bu normal bir durumdur. HuggingFace datasets kutuphanesi otomatik yeniden dener.
Ek onlem olarak:
```python
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 300 saniye timeout
```

### 4.3 Data Augmentation (Veri Artirma)

Data augmentation, mevcut veriden yeni varyasyonlar olusturarak modelin daha
genel ogrenebilmesini saglar.

**Speed Perturbation (Hiz Degisimi):**
```python
def speed_perturb(batch):
    speed = random.choice([0.9, 1.0, 1.1])
    if speed != 1.0:
        audio = np.array(batch["audio"]["array"], dtype=np.float32)
        new_len = int(len(audio) / speed)
        indices = np.linspace(0, len(audio) - 1, new_len)
        batch["audio"] = {
            "array": np.interp(indices, np.arange(len(audio)), audio),
            "sampling_rate": batch["audio"]["sampling_rate"],
        }
    return batch
```

- **0.9x:** Ses yavaslar (uzar), daha agir okuyus
- **1.0x:** Orijinal hiz (degisiklik yok)
- **1.1x:** Ses hizlanir (kisalir), daha hizli okuyus

**Neden bu teknik?**
- Kari'ler farkli hizlarda okur
- Model hiz farkliliklarina dayanikli olur
- Basit ama etkili bir augmentation

**Alternatifleri:**
| Teknik | Ne Yapar | Neden Kullanmadik |
|---|---|---|
| Noise injection | Gurultu ekler | Kuran tilavetinde temiz ses onemli |
| Pitch shift | Ses tonu degistirir | Kari sesi bozulabilir |
| SpecAugment | Mel spectrogram'a maskeleme | Denenebilir (gelecek is) |
| Room reverb | Oda yankisi ekler | Studio kayitlarinda gereksiz |

### 4.4 Arapca Metin Zorluklari

Arapca, NLP ve STT icin ozel zorluklar barindirir:

**Harekeler:**
Arapca harfler ustteki/alttaki isareterle farkli okunur. Kuran'da tum harekeler
yazilir (diger Arapca metinlerde genellikle yazilmaz).

```
ك = kaf (sadece harf)
كَ = ke (fatha ile)
كُ = ku (damma ile)
كِ = ki (kasra ile)
كْ = k  (sukun ile — sessiz)
```

**WER'e Etkisi:**
Harekeler eklendiginde kelime sayisi ayni kalir ama hata olasıligi artar.
Bir harekenin yanlis taninmasi bile WER'i arttirir. Bu yuzden iki metrik
kullaniyoruz: `wer_diacritized` (katı) ve `wer_normalized` (hareke yok).

---

## 5. Model Secimi ve Konfigurasyon

### 5.1 Neden whisper-large-v3?

| Kriter | whisper-small | whisper-medium | **whisper-large-v3** |
|---|---|---|---|
| Parametreler | 244M | 769M | **1.5B** |
| Arapca WER (baz) | ~15% | ~8% | **~4%** |
| GPU (LoRA egitim) | ~5 GB | ~12 GB | **~20 GB** |
| Egitim suresi | ~3-4 saat | ~6-8 saat | **~12-13 saat** |
| Fine-tune sonrasi WER | ~5-8% | ~3-5% | **<%2** |

**Buyuk model avantajlari:**
- Daha fazla bilgi tasiyor (1.5B param = daha zengin ic temsiller)
- Fine-tuning'de daha hizli ve daha iyi yakinsama
- Zaten iyi bir baz performans var (Arapca %4 WER)

**Buyuk model dezavantajlari:**
- Daha fazla GPU gerekli (A100 sart)
- Egitim daha uzun (12-13 saat)
- Inference daha yavas (ancak optimizasyonlarla dengelenebilir)

> **Karar:** Kalite oncelikli oldugu icin en buyuk modeli sectik.

### 5.2 LoRA Konfigurasyonu

```python
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    bias="none",
)
```

**r=32 secimi:**
- r=8: Cok az parametre, yetersiz ogrenim kapasitesi
- r=16: Orta, basit gorevler icin yeterli
- **r=32: Denge nokti — yeterli kapasite, makul bellek**
- r=64: Daha cok parametre ama diminishing returns (azalan getiri)

**alpha=64 secimi:**
- Genel kural: alpha = 2 * r
- alpha/r orani = 2 → LoRA agirliklarinin ogrenme oranini belirler
- Daha buyuk alpha = daha agresif guncelleme

**dropout=0.05:**
- Egitim sirasinda rastgele %5 agirlik sifirlanir
- Overfitting'i onler
- Kucuk dataset'lerde daha yuksek dropout onerilir

**bias="none":**
- Bias terimlerini egitmez
- Bellek tasarrufu + LoRA paper'in onerisi

**Target Modules — 6 Modul:**

Whisper'in her transformer blogu sunlari icerir:
```
[Self-Attention]     -> q_proj, k_proj, v_proj, out_proj  (4 modul)
[Cross-Attention]    -> q_proj, k_proj, v_proj, out_proj  (4 modul, encoder'da yok)
[Feed-Forward]       -> fc1, fc2                          (2 modul)
```

Biz attention + feed-forward'u hedefliyoruz = 6 modul tipi.
Bu, her transformer blogunun LoRA uyarlamasi almasini saglar.

### 5.3 Egitim Hiperparametreleri

```python
BATCH_SIZE    = 32       # GPU basina
GRAD_ACCUM    = 2        # Gradient birikimi
EFFECTIVE     = 64       # 32 x 2 = 64
LEARNING_RATE = 3e-4     # 0.0003
WARMUP_STEPS  = 200      # Isitma adimlari
EPOCHS        = 5        # Tam gecis
MAX_STEPS     = 18_281   # 5 * 234000 / 64
```

**Learning Rate: 3e-4**
- LoRA icin standart oran (LoRA paper'da onerilmistir)
- Full fine-tuning'de 1e-5 kullanilir (cok daha kucuk)
- LoRA'da daha yuksek LR kullanilabilir cunku sadece kucuk bir kisim guncelleniyor

**Cosine Scheduler:**
```
LR
 |  /\
 | /  \__
 |/      \___
 +-----------> Steps
   warmup  cosine decay
```

- Ilk 200 adim: LR 0'dan 3e-4'e cıkar (warmup)
- Sonra yavas yavas 0'a yaklasir (cosine decay)

**Neden linear degil cosine?**
- Cosine, egitimin sonunda LR'yi daha yavas azaltir
- Son adimlarda daha ince ayarlamalar yapilir
- Pratikte linear'dan daha iyi sonuc verir

**Batch Size: 64 (32 x 2):**
- 32 sample GPU'ya ayni anda yuklenir
- 2 kez bu tekrarlanir (gradient accumulation)
- Gradientler toplanir ve tek seferde guncelleme yapilir
- Sonuc: 64 sample'lik efektif batch size

**Neden 64?**
| Batch | Steps | Avantaj | Dezavantaj |
|---|---|---|---|
| 16 | ~73K | Daha ince guncelleme | Cok yavas |
| 32 | ~36K | Iyi denge | Orta hiz |
| **64** | **~18K** | **Optimal denge** | — |
| 128 | ~9K | Cok hizli | Az adim, yetersiz ogrenim |

> 18K adim, modelin 5 epoch boyunca veriyi gormesi icin yeterli.
> 128'de adim sayisi yarisina duser ve model yeterince ogrenemeyebilir.

**Epochs: 5**
- 3 epoch: Model henuz tam ogrenmemis olabilir
- **5 epoch: Kuran metinleri icin yeterli (tekrarli yapilar)**
- 10 epoch: Overfitting riski artar

---

## 6. Egitim Sureci (Adim Adim)

Asagida notebook'un her adimini (hucresini) ve neden o islemin yapildigini
acikliyoruz.

### Adim 1: Environment (Ortam Kontrolu)

```python
import os, shutil

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
os.environ["HF_DATASETS_CACHE"] = "/content/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/content/model_cache"

from google.colab import drive
drive.mount("/content/drive")

import torch
assert torch.cuda.is_available(), "GPU yok!"

gpu = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu}, VRAM: {vram:.1f} GB")
```

**Ne yapiyor?**
1. HuggingFace indirme timeout'unu 300 saniyeye cikarir
2. Cache klasorlerini belirler
3. Google Drive'i baglar (checkpoint kaydi icin)
4. GPU kontrolu yapar

**Neden?**
- Timeout artirmak, buyuk model indirmede baglanti kopmalarini onler
- Drive baglantisi, egitimi kaldigi yerden devam ettirmeyi saglar
- GPU yoksa devam etmenin anlami yok

### Adim 2: Dependencies (Bagimliliklar)

```python
!pip install -q transformers==4.47.0 peft==0.14.0 datasets==3.2.0 accelerate==1.2.1
!pip install -q evaluate jiwer soundfile librosa tensorboard huggingface_hub
!pip cache purge
```

**Neden sabit versiyonlar?**
- `transformers==4.47.0`: Test edilmis, calisan versiyon
- Versiyon sabitleme, "dun calisiyordu, bugun bozuldu" problemini onler
- `pip cache purge`: Colab'da sinirli diski temiz tutar

> **Onemli:** Pip bagimliliklari bazen catisir. `--quiet (-q)` ile
> uyari mesajlari gizlenir, ama hata varsa gorulur.

### Adim 3: HuggingFace Login

```python
from huggingface_hub import login
login()
```

- Dataset erisimi icin gerekli (tarteel-ai/everyayah)
- Model Hub'a yuklemek icin de sart
- Token: https://huggingface.co/settings/tokens

### Adim 4: Checkpoint Resume (Kaldigi Yerden Devam)

```python
from transformers.trainer_utils import get_last_checkpoint

OUTPUT_DIR = "/content/drive/MyDrive/tilavet/whisper-large-v3-quran-lora"
resume_from = get_last_checkpoint(OUTPUT_DIR)
if resume_from:
    print(f"Checkpoint bulundu: {resume_from}")
else:
    print("Sifirdan baslanacak.")
```

**Neden onemli?**
- Colab baglantisi kopabilir (12-13 saatlik egitim)
- Drive'a kaydedilen checkpoint'lardan egitim devam eder
- Koptugunda: Runtime > Run All > otomatik devam

**get_last_checkpoint ne yapar?**
OUTPUT_DIR'deki checkpoint klasorlerini tarar (checkpoint-2000, checkpoint-4000 vs.)
ve en son olani dondurur.

### Adim 5: Imports ve Config

```python
MODEL_ID       = "openai/whisper-large-v3"
LORA_R         = 32
LORA_ALPHA     = 64
BATCH_SIZE     = 32
GRAD_ACCUM     = 2        # 32 x 2 = 64 effective
LEARNING_RATE  = 3e-4
MAX_STEPS      = 18_281   # 5 epoch
EVAL_STEPS     = 2000
SAVE_STEPS     = 2000
```

**Neden EVAL_STEPS = SAVE_STEPS = 2000?**
- `load_best_model_at_end=True` kullaniyoruz (en iyi modeli sec)
- Bu, save_steps'in eval_steps'in tam kati olmasini gerektirir
- Her 2000 adimda: eval yap + checkpoint kaydet + en iyiyi sec

**Neden 2000?**
- 500 adimda eval: Cok sik, egitimi yavaslatir (eval ~10 dk surer)
- 2000 adimda eval: 18K adimda 9 eval = makul izleme
- 5000 adimda eval: Cok seyrek, sorunlar gec farkedilir

### Adim 6: Model + LoRA Yukleme

```python
# Processor: ses -> mel spectrogram, metin -> tokenlar
processor = WhisperProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.set_prefix_tokens(language="ar", task="transcribe")

# Model: fp32 + SDPA (optimized attention)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID,
    attn_implementation="sdpa",
)
model.config.use_cache = False
model.generation_config.language = "ar"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

# LoRA ekleme
lora_config = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    bias="none",
)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()
```

**Kritik satirlar:**

1. `attn_implementation="sdpa"`: Optimize attention (%20-40 hiz artisi)
2. `use_cache=False`: Gradient checkpointing ile uyumsuz, kapatilmali
3. `forced_decoder_ids=None`: Whisper'in zorunlu token'larini devre disi birakir
4. `enable_input_require_grads()`: PEFT + gradient checkpointing icin **ZORUNLU**

> **enable_input_require_grads() neden gerekli?** LoRA modeldeki cogu katmani
> dondurur (requires_grad=False). Gradient checkpointing ise tum girdilerin
> requires_grad=True olmasini bekler. Bu fonksiyon, girdi embedding'lerine
> grad aktive eder ve ikisi arasindaki uyumsuzlugu cozer.

> **task_type neden yok?** PEFT kutuphanesinin 1988 numarali bug'i. LoraConfig'e
> `task_type="SEQ_2_SEQ_LM"` eklenirse, Whisper model `input_ids` bekler
> (metin modeli gibi). Ancak Whisper ses modeli — `input_features` bekler.
> task_type'i kaldirmak bu sorunu cozer.

### Adim 7: Dataset Yukleme

```python
# Train: streaming (disk 0)
train_stream = load_dataset("tarteel-ai/everyayah", split="train", streaming=True)

# Eval: 2000 sample RAM'e yukle
eval_stream = load_dataset("tarteel-ai/everyayah", split="test", streaming=True)
eval_raw = list(eval_stream.take(2000))
```

**Neden eval streaming degil?**
- Eval sirasinda tum veri uzerinde metrik hesaplanmali
- Streaming ile bu yapilamaz (veri bir kez okunur)
- 2000 sample'i RAM'e almak yeterli ve pratik

### Adim 8: Preprocessing (On Isleme)

**Pipeline:**
```
Ham ses -> Filtre (<=30sn) -> Speed perturb -> Mel spectrogram + Token -> Shuffle
```

1. **Filtre:** 30 saniyeden uzun kayitlar atilir (Whisper siniri)
2. **Speed perturbation:** 0.9x / 1.0x / 1.1x rastgele hiz degisimi
3. **prepare_sample:** Ses -> mel spectrogram, metin -> token ID
4. **Shuffle:** buffer_size=10,000 ile karistirma

```python
features = processor(audio["array"], sampling_rate=16000, return_tensors="np")
batch["input_features"] = features.input_features[0]   # [80, 3000] mel spectrogram
batch["labels"] = processor.tokenizer(batch["text"]).input_ids  # [N] token ids
```

### Adim 9: Data Collator + Metrikler

**Data Collator:**
Farkli uzunluktaki ornekleri batch'ler haline getirir:
- input_features: Padding ile ayni boyuta getirilir
- labels: Padding + -100 ile maskelenir (kayip hesabinda yok sayilir)

```python
labels = labels_batch["input_ids"].masked_fill(
    labels_batch.attention_mask.ne(1), -100
)
```

> **-100 neden onemli?** PyTorch'un CrossEntropyLoss fonksiyonu -100 degerli
> etiketleri yok sayar. Bu, padding token'larinin kayip hesabini bozmasini onler.

**Metrik Hesaplama:**
```python
def compute_metrics(pred):
    pred_str = processor.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)

    wer_diac = wer_metric.compute(predictions=pred_str, references=label_str)
    wer_norm = wer_metric.compute(
        predictions=[strip_harakat(p) for p in pred_str],
        references=[strip_harakat(l) for l in label_str],
    )
    return {"wer_diacritized": wer_diac * 100, "wer_normalized": wer_norm * 100}
```

### Adim 10: Egitim

```python
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    max_steps=18281,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=3e-4,
    lr_scheduler_type="cosine",
    warmup_steps=200,
    fp16=True,
    fp16_full_eval=True,
    gradient_checkpointing=True,
    eval_strategy="steps",
    eval_steps=2000,
    save_steps=2000,
    save_total_limit=3,
    predict_with_generate=True,
    load_best_model_at_end=True,
    metric_for_best_model="wer_normalized",
    greater_is_better=False,
    dataloader_num_workers=2,
    remove_unused_columns=False,
    label_names=["labels"],
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

trainer.train(resume_from_checkpoint=resume_from)
```

**Kritik Parametreler:**

| Parametre | Deger | Neden |
|---|---|---|
| `fp16=True` | Mixed precision | 2x hiz, yariya bellek |
| `fp16_full_eval=True` | Eval'da da fp16 | OOM onleme |
| `gradient_checkpointing=True` | Bellek optimizasyonu | 50GB -> 20GB |
| `predict_with_generate=True` | WER icin metin uretimi | Eval'da gerekli |
| `remove_unused_columns=False` | Sutun silmeyi kapat | Streaming dataset uyumu |
| `label_names=["labels"]` | Etiket sutunu | Whisper icin gerekli |
| `save_total_limit=3` | Max 3 checkpoint | Disk tasarrufu |
| `dataloader_num_workers=2` | 2 paralel veri yukleme | CPU/GPU dengelemesi |

### Adim 11: Kaydet + Test

```python
# LoRA adapter'i kaydet
FINAL = OUTPUT_DIR + "/final"
model.save_pretrained(FINAL)
processor.save_pretrained(FINAL)

# 5 ornekle test
for sample in test_stream.take(5):
    inputs = processor(sample["audio"]["array"], return_tensors="pt").to(model.device)
    ids = model.generate(**inputs, max_length=225)
    pred = processor.batch_decode(ids, skip_special_tokens=True)[0]
    print(f"Ref:  {sample['text']}")
    print(f"Pred: {pred}")
```

**Not:** Kaydedilen sadece LoRA adapter'dir (~50 MB), tam model degil (~6 GB).
Kullanim sirasinda baz model + adapter birlestirilir.

---

## 7. Karsilasilan 17 Hata ve Cozumleri

Bu bolum, egitim surecinde karsilastigimiz tum hatalari, nedenlerini ve
cozumlerini anlatir. Her hata bir ders iceriyor.

### Hata 1: torchaudio pip Sorunu

**Hata:**
```
ERROR: pip's dependency resolver does not currently take into account
all the packages that are installed.
```

**Neden:** Colab'daki onceden yuklu `torchaudio` versiyonu, yeni yuklenen
`torch` versiyonu ile catisiyor.

**Cozum:** Uyari (warning) — gercek bir hata degil. Pip bagimliliklari
cakissa bile paketler dogru calisiyor. Guvende yok sayilabilir.

**Ders:** Her pip uyarisi bir hata degildir. "ERROR" yazsa bile gercek
hatayi ayirt etmek onemli.

### Hata 2: torch_dtype conv1d Mismatch

**Hata:**
```
RuntimeError: Input type (float) and bias type (c10::Half) should be the same
```

**Neden:** `torch_dtype=torch.float16` ile model yuklendiginde, Conv1d
katmaninin bias'i fp16 olur. Ancak mel spectrogram girdisi fp32'dir.

**Cozum:** `torch_dtype` parametresini model yuklemeden kaldirdik. Model fp32
olarak yuklenip, egitim sirasinda fp16'ya cevriliyor (`fp16=True` ile).

```python
# YANLIS:
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16)

# DOGRU:
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
```

**Ders:** `torch_dtype` tum katmanlari cevirir. Whisper'in Conv1d'si bununla
uyumlu degil. Mixed precision (fp16=True) daha guvenli cunku sadece hesaplama
fp16 olur, agirliklar fp32 kalir.

### Hata 3: total_mem AttributeError

**Hata:**
```
AttributeError: 'NoneType' object has no attribute 'total_mem'
```

**Neden:** GPU bellek sorgulama kodu Colab'da farkli calisiyor. Bazi
fonksiyonlar None donduruyor.

**Cozum:**
```python
# YANLIS:
torch.cuda.mem_get_info(0).total_mem

# DOGRU:
torch.cuda.get_device_properties(0).total_memory
```

**Ders:** GPU bilgi sorgulama API'lari platformdan platforma degisebilir.
Daima resmi PyTorch dokumanini kontrol edin.

### Hata 4: Eval Disk Full

**Hata:**
```
OSError: [Errno 28] No space left on device
```

**Neden:** Eval sırasında predict_with_generate=True kullanılırken, uretilen
tahminler diske yaziliyor. Colab'in ~100 GB diski, model + checkpoint + cache
ile doluyor.

**Cozum:**
1. `pip cache purge` ile pip onbellegini temizle
2. Gereksiz dosyalari sil
3. `save_total_limit=3` ile sadece son 3 checkpoint tut

**Ders:** Colab disk alani sinirli. Her zaman disk kullanimini izleyin:
```python
shutil.disk_usage("/content")
```

### Hata 5: GPU Underutilization

**Belirti:** GPU bellegi 20/80 GB, yani %75 bos.

**Neden:** Gradient checkpointing + LoRA, bellegi cok verimli kullaniyor.
Bu bir "hata" degil, aslinda basari.

**Aciklama:**
- 20 GB = model (6 GB) + LoRA (negligible) + batch (8 GB) + optimizer (6 GB)
- 60 GB bos = israf degil, guvenlik margini
- Batch buyutmek adim sayisini azaltir (kalite duser)

**Ders:** GPU belleginin %100 kullanilmasi gerekmez. Onemli olan verimli
egitim, GPU dolulugunun yuzdesi degil.

### Hata 6: Eval Sikligi Optimizasyonu

**Sorun:** Ilk versiyonda eval her 500 adimda yapiliyordu. Her eval ~10 dakika
suruyordu. 18K adimda 36 eval = 6 saat sadece eval'a gidiyor.

**Cozum:** Eval araligini 500'den 2000'e cikardik.
- 18K adimda 9 eval = ~1.5 saat eval
- 4.5 saat tasarruf

**Ders:** Eval sikligi, egitim hizini buyuk olcude etkiler. Dengeli bir
aralık secmek onemli.

### Hata 7: Trainer.tokenizer Deprecation

**Uyari:**
```
FutureWarning: `Trainer.tokenizer` is now deprecated.
Use `Trainer.processing_class` instead.
```

**Neden:** Yeni HuggingFace transformers versiyonlarinda `tokenizer` yerine
`processing_class` kullanilmali.

**Cozum:** Sadece uyari, calismayi etkilemiyor. Gelecek versiyonlarda
`processing_class` parametresine gecilecek.

**Ders:** Deprecation uyarilari hemen hataya yol acmaz ama uzun vadede
guncelleme gerekir.

### Hata 8: HuggingFace Hub Timeout

**Hata:**
```
ProtocolError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection'))
```

**Neden:** HuggingFace CDN'den streaming sirasinda baglanti kopmasi.
Internet kararsizligi veya CDN yuku.

**Cozum:** Otomatik yeniden deneme mekanizmasi var. Ek olarak:
```python
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
```

**Ders:** Streaming veri yuklemede baglanti kopmalari normaldir.
Timeout degerini yukseltmek ve sabırli olmak yeterli.

### Hata 9: Import Sirasi Hatasi

**Hata:** Moduller yanlis sirada import edildiginde `NameError`.

**Neden:** Bazi moduller (evaluate, jiwer) pip install'dan sonra import
edilmeli. Ayni hucrede install + import calismaz.

**Cozum:** Install ve import'u ayri hucrelere koyduk.

**Ders:** Colab'da pip install ayri hucrede, import ayri hucrede olmali.

### Hata 10: evaluate Modulu Eksik

**Hata:**
```
ModuleNotFoundError: No module named 'evaluate'
```

**Neden:** evaluate paketi pip ile yuklenmemis.

**Cozum:**
```python
!pip install evaluate jiwer
```

**Ders:** Dependencies listesini her zaman kontrol edin. Colab'da onceden
yuklu olmayan paketler var.

### Hata 11: processing_class Parametresi

**Sorun:** Eski transformers versiyonlarinda `processing_class` desteklenmiyordu.
Yeni versiyonlarda `tokenizer` deprecated.

**Cozum:** `transformers==4.47.0` versiyonunda her ikisi de calisiyor.
Sabit versiyon kullanarak bu sorunu onledik.

**Ders:** Kutuphanelerde API degisiklikleri sik olur. Sabit versiyonlar
guvenlik saglar.

### Hata 12: PEFT Bug #1988 (task_type)

**Hata:** Model `input_features` yerine `input_ids` bekliyor.

**Neden:** LoraConfig'e `task_type="SEQ_2_SEQ_LM"` eklenmesi, PEFT'in modeli
bir metin modeli gibi muamele etmesine yol aciyor. Whisper ses modelidir ve
`input_features` bekler.

**Cozum:** `task_type` parametresini tamamen kaldirdik.

```python
# YANLIS:
lora_config = LoraConfig(
    task_type="SEQ_2_SEQ_LM",  # BU SATIR SORUN
    r=32, ...
)

# DOGRU:
lora_config = LoraConfig(
    r=32, ...
    # task_type YOK
)
```

**Ders:** PEFT'in ses modelleriyle (Whisper, Wav2Vec2) kullanilmasinda
`task_type` soruna yol acabiliyor. PEFT GitHub issue #1988.

### Hata 13: remove_unused_columns

**Hata:** Streaming dataset'te beklenmedik sutun hatalari.

**Neden:** Trainer varsayilan olarak kullanimlmayan sutunlari kaldirir.
Streaming dataset'lerde bu guvensiz.

**Cozum:**
```python
remove_unused_columns=False,
label_names=["labels"],
```

**Ders:** Streaming dataset kullaniyorsaniz, `remove_unused_columns=False`
ve `label_names` belirtmek sart.

### Hata 14: OOM — Gradient Checkpointing

**Hata:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory.
Tried to allocate 2.00 GiB (GPU 0; 79.15 GiB total capacity; 78.52 GiB already allocated)
```

**Neden:** Gradient checkpointing olmadan, 1.5B parametreli modelin tum
aktivasyonlari bellekte tutuluyor. A100 80 GB bile yetmiyor.

**Cozum:**
```python
gradient_checkpointing=True
```

Bu satir, bellegi ~50+ GB'dan ~20 GB'a dusurdu.

**Ders:** Buyuk modeller icin gradient checkpointing SART. Hiz kaybi
(%30) var ama bellekte %60 tasarruf saglar.

### Hata 15: OOM — fp16_full_eval

**Hata:** Eval sirasinda OOM (Out of Memory).

**Neden:** Eval sirasinda model varsayılan olarak fp32'ye donusturulur.
Bu, bellek kullanimini ikiye katlar.

**Cozum:**
```python
fp16_full_eval=True
```

**Ders:** Egitim fp16 ile calissa bile eval varsayılan fp32'dir.
Buyuk modellerde `fp16_full_eval=True` eklenmeli.

### Hata 16: requires_grad Hatasi

**Hata:**
```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

Oncesindeki uyari:
```
UserWarning: None of the inputs have requires_grad=True. Gradients will be None
```

**Neden:** PEFT (LoRA) cogu katmani dondurur (requires_grad=False).
Gradient checkpointing ise tum girdilerin requires_grad=True olmasini bekler.
Bu cakisma hataya yol acar.

**Cozum:**
```python
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()  # <-- BU SATIR
```

`enable_input_require_grads()`, girdi embedding'lerine requires_grad=True atar.
Boylece gradient checkpointing duzgun calisiyor.

**Ders:** PEFT + gradient checkpointing = `enable_input_require_grads()` sart.
Bu her PEFT modelinde gerekli degil ama Whisper gibi encoder-decoder'larda
buyuk olasilikla gerekecektir.

### Hata 17: QLoRA conv1d dtype Mismatch

**Hata:**
```
RuntimeError: Input type (float) and bias type (c10::Half) should be the same
  File "torch/nn/modules/conv.py", line 459, in _conv_forward
```

**Neden:** QLoRA (4-bit) kullanildiginda, `bnb_4bit_compute_dtype=torch.float16`
Conv1d katmaninin bias'ini fp16'ya cevirir. Ancak mel spectrogram girdisi fp32.

**Cozum:** QLoRA'yi tamamen kaldirildi. fp32 model kullaniyoruz.
20/80 GB bellek yeterli, 4-bit'e gerek yok.

**Ders:** QLoRA tum modellerde sorunsuz calismaz. Whisper'in Conv1d katmani
ozel bir durumdur. Mecbur kalmadikca tam hassasiyette (fp32) egitim tercih edin.

---

## 8. Alternatif Yaklasimlar

Bu bolumde, farkli secimler yapsaydik ne olurdu sorusunu tartisiyoruz.

### 8.1 Farkli Model

| Model | WER (baz) | WER (fine-tune tahmini) | GPU | Sure |
|---|---|---|---|---|
| whisper-small (244M) | ~15% | ~5-8% | ~5 GB | ~3-4 saat |
| whisper-medium (769M) | ~8% | ~3-5% | ~12 GB | ~6-8 saat |
| **whisper-large-v3 (1.5B)** | **~4%** | **<%2** | **~20 GB** | **~12-13 saat** |

> Kucuk model: Daha hizli, daha ucuz, ama daha dusuk kalite.
> Ogrenci projesi icin whisper-small ile baslamak iyi bir pratik olur.

### 8.2 Farkli Dataset

| Dataset | Boyut | Icerik | Lisans |
|---|---|---|---|
| **tarteel-ai/everyayah** | **234K** | **Kuran tilaveti** | **MIT** |
| CommonVoice Arabic | 100K+ | Genel Arapca konusma | CC-0 |
| MGB-2 | 1200 saat | Arapca yayın | Arastirma |

> CommonVoice: Genel Arapca icin iyi ama Kuran tilavetine ozel degil.
> everyayah en uygun dataset.

### 8.3 Full Fine-tuning vs LoRA

| | Full Fine-tuning | LoRA (r=32) |
|---|---|---|
| Egitilen parametreler | 1.5B (%100) | 13M (%0.85) |
| GPU bellegi | 80+ GB (yetmez) | ~20 GB |
| Kayip riski | Yuksek (catastrophic forgetting) | Dusuk |
| Adapter boyutu | ~6 GB | ~50 MB |
| Kalite | Potansiyel olarak en iyi | Cok yakin |

> Full fine-tuning icin coklu GPU (4x A100) gerekir. LoRA tek A100 ile
> yakin sonuclar verir. Maliyet/kalite oraninda LoRA acik ara kazanir.

### 8.4 QLoRA (4-bit)

| | fp32 + LoRA (biz) | QLoRA (4-bit) |
|---|---|---|
| GPU bellegi | ~20 GB | ~8-10 GB |
| Uyumluluk | Sorunsuz | Conv1d sorunu |
| Hassasiyet | Maksimum | Hafif kayip |
| T4 ile calısır mı? | Hayir (16 GB < 20 GB) | Evet |

> T4 ile calismaniz gerekirse QLoRA zorunlu olur. Conv1d sorununu cozmek
> icin ozel bir hook yazmak gerekir. A100 varsa fp32 tercih edin.

### 8.5 Farkli Batch Size

| Batch | Steps | Egitim Suresi | Kalite |
|---|---|---|---|
| 16 | ~73K | ~24-26 saat | En ince guncelleme |
| 32 | ~36K | ~16-18 saat | Iyi |
| **64** | **~18K** | **~12-13 saat** | **Optimal** |
| 128 | ~9K | ~7-8 saat | Yetersiz ogrenim riski |

### 8.6 Farkli Learning Rate

| LR | Ozellik |
|---|---|
| 1e-4 | Yavas ama guvenli. Yakinsamasi uzun surer |
| **3e-4** | **LoRA icin standart. Iyi denge** |
| 1e-3 | Agresif. Kararsiz olabilir |

### 8.7 Farkli LoRA r

| r | Param | Bellek | Kalite |
|---|---|---|---|
| 8 | ~3.3M | ~15 GB | Basit gorevler icin |
| 16 | ~6.6M | ~17 GB | Orta |
| **32** | **~13M** | **~20 GB** | **Iyi denge** |
| 64 | ~26M | ~25 GB | Diminishing returns |

### 8.8 Runpod vs Colab

| | Google Colab Pro+ | Runpod (A100 80GB) |
|---|---|---|
| Fiyat | $50/ay (sabit) | ~$2-3/saat |
| 12 saatlik egitim | $50/ay icinde | ~$24-36 |
| Guvenilirlik | Bazen baglanti kopar | Daha stabil |
| Kurulum | Hazir ortam | Docker setup gerekir |
| Avantaj | Basit, Google Drive | Esnek, terminal erisimi |

> Tek bir egitim icin Runpod daha ucuz. Surekli denemeler icin Colab Pro+
> abone olunmasi daha pratik olabilir.

---

## 9. Performans ve Sonuclar

### WER Egrisi

```
WER (%)
  |
8 |  *
  |
6 |     *
  |
4 |        *
  |
2 |           *     *     *     *
  |
0 +----+----+----+----+----+----+----> Steps
   0   2K   4K   6K   8K   10K  12K
```

| Step | WER (normalized) | WER (diacritized) |
|---|---|---|
| 2,000 | ~3.92% | ~6.5% |
| 4,000 | ~2.5% | ~4.8% |
| 6,000 | ~1.8% | ~3.5% |
| 8,000 | ~1.6% | ~3.2% |
| 10,000+ | ~1.5% | ~3.0% |

> **Sonuc:** wer_normalized %1.58'e dustu. Bu, Tarteel AI'in %5.75'inin
> 3.6 kat altinda. Hedef olan %3'un de altinda.

### GPU Bellek Kullanimi

| Bilesken | Boyut |
|---|---|
| Whisper-large-v3 (fp32) | ~6 GB |
| LoRA adapter | <0.1 GB |
| Optimizer state | ~6 GB |
| Batch (32 sample) | ~8 GB |
| **Toplam** | **~20 GB / 80 GB** |

### Tarteel AI Karsilastirmasi

| Metrik | Tarteel AI | Tilavet |
|---|---|---|
| WER (normalized) | %5.75 | **%1.58** |
| Acik kaynak | Hayir | **Evet** |
| API erisimi | Sinirli | **Tamamen acik** |
| 108 dil ceviri | Yok | **Var** |

---

## 10. Uretim Ortamina Gecis

### 10.1 Model Kaydetme

Egitim sonrasinda sadece LoRA adapter kaydedilir:
```python
model.save_pretrained("/path/to/final")    # ~50 MB (sadece LoRA)
processor.save_pretrained("/path/to/final") # Tokenizer + feature extractor
```

Kullanim icin baz model + adapter birlestirilir:
```python
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
model = PeftModel.from_pretrained(base, "/path/to/final")
model = model.merge_and_unload()  # LoRA'yi modele birlestir
processor = WhisperProcessor.from_pretrained("/path/to/final")
```

### 10.2 HuggingFace Hub'a Yukleme

```python
model.push_to_hub("baristiran/whisper-large-v3-quran-lora")
processor.push_to_hub("baristiran/whisper-large-v3-quran-lora")
```

Bu komutla model herkesin erisebilecegi sekilde HuggingFace Hub'da yayinlanir.
Diger gelistiriciler modeli su sekilde kullanabilir:
```python
from transformers import pipeline
pipe = pipeline("automatic-speech-recognition", model="baristiran/whisper-large-v3-quran-lora")
result = pipe("tilavet.wav")
```

### 10.3 FastAPI Inference Server

```python
from fastapi import FastAPI, UploadFile
import torch

app = FastAPI()

@app.post("/transcribe")
async def transcribe(file: UploadFile):
    audio = load_audio(await file.read())
    inputs = processor(audio, return_tensors="pt").to("cuda")
    with torch.no_grad():
        ids = model.generate(**inputs)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return {"text": text}
```

### 10.4 CoreML Donusumu (iOS)

Apple cihazlar icin model donusumu:
```python
import coremltools as ct
# Whisper -> CoreML donusumu icin ozel pipeline gerekir
# Detaylar: https://github.com/argmaxinc/WhisperKit
```

WhisperKit projesi, Whisper modellerini iOS/macOS'ta calıstirmak icin
optimize edilmis bir cozum sunar.

---

## 11. Maliyet Analizi

### GPU Bulut Fiyatlari (2024-2025)

| Platform | GPU | VRAM | Fiyat ($/saat) | 12 saat |
|---|---|---|---|---|
| Google Colab Pro+ | A100 80GB | 80 GB | ~$4.2 (aylik $50) | ~$50/ay |
| **Runpod** | **A100 80GB** | **80 GB** | **~$2.49** | **~$30** |
| Lambda Labs | A100 80GB | 80 GB | ~$2.21 | ~$27 |
| Vast.ai | A100 80GB | 80 GB | ~$1.50-2.50 | ~$18-30 |
| AWS (p4d) | A100 40GB | 40 GB | ~$32 (instance) | ~$384 |

> **En ekonomik:** Vast.ai veya Lambda Labs (spot fiyatlar degisken)
> **En pratik:** Colab Pro+ (kurulum gerektirmez)
> **Dengeli:** Runpod (iyi fiyat + kolay arayuz)

### Bu Projenin Maliyeti

| Kalem | Maliyet |
|---|---|
| Colab Pro+ (1 ay) | ~$50 |
| veya Runpod A100 (12 saat) | ~$30 |
| HuggingFace (ucretsiz) | $0 |
| Dataset (MIT, ucretsiz) | $0 |
| **Toplam** | **~$30-50** |

### Ogrenci Butcesiyle Neler Yapilabilir?

| Butce | Oneri |
|---|---|
| $0 | Colab Free (T4) ile whisper-small egitimi |
| $10-20 | Kaggle (2x T4) ile whisper-medium |
| $30-50 | Runpod A100 ile whisper-large-v3 |
| $50/ay | Colab Pro+ ile sinirsiz deneme |

### Ucretsiz Alternatifler

| Platform | GPU | Sinir | Not |
|---|---|---|---|
| Google Colab (ucretsiz) | T4 (16 GB) | ~4 saat | whisper-small icin yeterli |
| Kaggle | 2x T4 (30 GB) | 30 saat/hafta | Cok GPU egitimi mumkun |
| Lightning AI | T4 | 22 saat/ay | Colab alternatifi |

---

## 12. Sozluk (Glossary)

| Terim (EN) | Turkce | Aciklama |
|---|---|---|
| **ASR** | Otomatik Ses Tanima | Automatic Speech Recognition — sesi metne cevirme |
| **STT** | Konusmadan Metne | Speech-to-Text — ASR ile ayni anlam |
| **WER** | Kelime Hata Orani | Word Error Rate — STT'nin ana metriği |
| **CER** | Karakter Hata Orani | Character Error Rate — karakter bazli olcum |
| **LoRA** | Dusuk Rankli Uyarlama | Low-Rank Adaptation — verimli fine-tuning |
| **PEFT** | Parametre-Verimli Fine-Tuning | Parameter-Efficient Fine-Tuning |
| **QLoRA** | Quantize LoRA | 4-bit quantization + LoRA |
| **NF4** | Normal Float 4 | QLoRA'nin ozel 4-bit formati |
| **SDPA** | Olcekli Nokta Carpim Dikkati | Scaled Dot-Product Attention |
| **OOM** | Bellek Yetersiz | Out of Memory — GPU bellegi bitti |
| **fp32** | 32-bit Kayan Nokta | Tam hassasiyet, 4 byte/param |
| **fp16** | 16-bit Kayan Nokta | Yarim hassasiyet, 2 byte/param |
| **GPU** | Grafik Islem Birimi | Graphics Processing Unit |
| **VRAM** | Video RAM | GPU'nun bellegi |
| **CUDA** | - | NVIDIA'nin GPU programlama platformu |
| **Tensor Core** | - | NVIDIA'nin matris carpim hizlandirici birimleri |
| **Epoch** | Donem | Tum verinin bir kez gorulmesi |
| **Batch** | Yigin | Ayni anda islenen ornek sayisi |
| **Step** | Adim | Bir gradyan guncelleme islemi |
| **Learning Rate** | Ogrenme Orani | Model agirliklarinin guncelleme buyuklugu |
| **Gradient** | Gradyan | Kaybın parametrelere gore turevleri |
| **Checkpoint** | Kontrol Noktasi | Modelin kaydedilen anlık goruntuleri |
| **Inference** | Cikarim | Egitilmis modelle tahmin yapma |
| **Encoder** | Kodlayici | Girdiyi ic temsile donusturen kisim |
| **Decoder** | Cozucu | Ic temsili ciktiya donusturen kisim |
| **Attention** | Dikkat | Modelin hangi girdilere odaklandigi mekanizma |
| **Mel Spectrogram** | - | Sesin frekans-zaman gorseli |
| **Tokenizer** | - | Metni sayisal token'lara donusturen arac |
| **Overfitting** | Asiri Uyum | Modelin egitim verisini ezberlemesi |
| **Augmentation** | Veri Artirma | Mevcut veriden yeni varyasyonlar olusturma |
| **Streaming** | - | Veriyi disk yerine internet uzerinden okuma |
| **Hub** | - | HuggingFace'in model/dataset paylasim platformu |

---

## 13. Pratik Egzersizler

### Egzersiz 1: HuggingFace'te Model ve Dataset Arama

**Amac:** HuggingFace Hub'i kullanmayi ogrenmek.

1. https://huggingface.co/models adresine git
2. "whisper" arayip sonuclari incele
3. `openai/whisper-large-v3` model kartini ac
4. Model bilgilerini not al: parametre sayisi, desteklenen diller, lisans
5. https://huggingface.co/datasets adresine git
6. "quran" veya "arabic speech" ara
7. `tarteel-ai/everyayah` dataset kartini incele
8. Dataset bilgilerini not al: ornek sayisi, split'ler, format

**Sorular:**
- Whisper ailesinde kac farkli model var?
- En kucuk model ile en buyuk model arasinda kac kat fark var?
- everyayah dataset'inde kac farkli kari var?

### Egzersiz 2: Whisper ile Basit Inference

**Amac:** Whisper modelini kullanarak sesten metin cikarmayi ogrenmek.

```python
# Google Colab'da calistirin (T4 yeterli)
!pip install transformers torch

from transformers import pipeline

# En kucuk model ile test
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# Ornek ses dosyasi ile test
result = pipe("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
print(result["text"])
```

**Gorev:**
1. Yukaridaki kodu calistirin
2. `whisper-small` yerine `whisper-tiny` deneyin
3. Fark var mi? Hangisi daha iyi?
4. Kendi ses kaydinizi (30 saniye Kuran tilaveti) ile deneyin

### Egzersiz 3: WER Hesaplama

**Amac:** WER metrigini anlamak ve hesaplamak.

**Elle hesaplama:**
```
Referans: "bir iki uc dort bes"
Tahmin:   "bir iki ux dort"

S (degistirme): "uc" -> "ux" = 1
D (silme):      "bes" eksik   = 1
I (ekleme):     yok           = 0
N (toplam):     5

WER = (1 + 1 + 0) / 5 = %40
```

**Python ile hesaplama:**
```python
!pip install jiwer

from jiwer import wer

ref = "bir iki uc dort bes"
hyp = "bir iki ux dort"
print(f"WER: {wer(ref, hyp) * 100:.1f}%")  # %40.0

# Arapca ornek
ref_ar = "بسم الله الرحمن الرحيم"
hyp_ar = "بسم الله الرحمن"
print(f"WER: {wer(ref_ar, hyp_ar) * 100:.1f}%")
```

**Gorev:**
1. Kendi orneklerinizi olusturup WER hesaplayin
2. Harekeli ve harekesiz Arapca icin WER farki ne oluyor?
3. Tek harflik bir hata WER'i ne kadar etkiler?

### Egzersiz 4: LoRA Parametrelerini Degistirme

**Amac:** LoRA parametrelerinin etkisini anlamak.

Asagidaki tabloyu doldurun (whisper-small ile, ucretsiz Colab'da):

```python
# r degerini degistirin ve sonuclari karsilastirin
for r_value in [4, 8, 16, 32]:
    lora_config = LoraConfig(
        r=r_value,
        lora_alpha=r_value * 2,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    # Not: Egitmeden sadece parametre sayisini karsilastirin
```

| r | alpha | Egitilen Param | Toplam Param Orani |
|---|---|---|---|
| 4 | 8 | ? | ? |
| 8 | 16 | ? | ? |
| 16 | 32 | ? | ? |
| 32 | 64 | ? | ? |

### Egzersiz 5: Farkli Dataset ile Deneme

**Amac:** Farkli dataset'lerle calismayi ogrenmek.

```python
from datasets import load_dataset

# Mozilla Common Voice — Arapca
cv = load_dataset("mozilla-foundation/common_voice_16_0", "ar", split="test",
                  streaming=True, trust_remote_code=True)

# Ilk 5 ornegi incele
for i, sample in enumerate(cv.take(5)):
    print(f"Metin: {sample['sentence']}")
    print(f"Sure: {len(sample['audio']['array']) / sample['audio']['sampling_rate']:.1f} sn")
    print()
```

**Gorev:**
1. Common Voice Arapca dataset'ini yukleyin
2. everyayah ile karsilastirin: ses kalitesi, metin icerigi, ornek sayisi
3. Whisper-small'u Common Voice ile test edin
4. Ayni modeli everyayah ile test edin
5. Hangisinde daha iyi sonuc aliyor? Neden?

---

## 14. Sik Sorulan Sorular (FAQ)

### GPU olmadan egitim yapilabilir mi?

Teknik olarak evet, ama pratik olarak hayir. CPU ile egitim aylar surer.
En azindan Google Colab ucretsiz T4 (16 GB) ile baslanabilir. T4 icin
whisper-small + LoRA (r=8) onerilir.

### Colab baglantisi koparsa ne olur?

Checkpoint mekanizmasi sayesinde egitim kaldigi yerden devam eder:
1. Checkpoint'lar Google Drive'a kaydedilir (her 2000 step'te)
2. Baglanti kopunca: Runtime > Run All
3. `get_last_checkpoint()` son checkpoint'i bulur
4. `resume_from_checkpoint` ile egitim devam eder

> **Onemli:** Drive mutlaka baglanmali (`drive.mount`). Checkpoint'lar
> Colab'in gecici diskine kaydedilirse, runtime kapandiginda kaybolur.

### Egitim ne kadar surer?

| GPU | whisper-small | whisper-medium | whisper-large-v3 |
|---|---|---|---|
| T4 (16 GB) | ~3-4 saat | ~8-10 saat | Calismaz (OOM) |
| V100 (16 GB) | ~2-3 saat | ~6-7 saat | Calismaz (OOM) |
| A100 (80 GB) | ~1-2 saat | ~4-5 saat | ~12-13 saat |

### LoRA mi full fine-tuning mi?

Genel kural:
- **GPU bellegi <40 GB:** LoRA (full fine-tuning sigar mi)
- **GPU bellegi >=40 GB:** Yine de LoRA onerilir (hiz, maliyet, catastrophic forgetting onleme)
- **4x A100 80 GB:** Full fine-tuning denenebilir (en iyi potansiyel kalite)

%99 durumda LoRA yeterli ve daha pratiktir.

### Farkli diller icin kullanilabilir mi?

Evet! Whisper 99 dil destekliyor. Ayni yaklasim ile:
- Turkce STT: Turkce konusma dataset'i ile fine-tune
- Farsca, Urduca, Endonezce: Ilgili dataset bulunup ayni pipeline ile
- `language="ar"` parametresini hedef dil ile degistirmek yeterli

### Model ne kadar buyuk (disk)?

| Bilesken | Boyut |
|---|---|
| Baz model (whisper-large-v3) | ~6 GB |
| LoRA adapter | ~50 MB |
| Processor (tokenizer + feature extractor) | ~5 MB |
| **Toplam (kullanim icin)** | **~6 GB + 50 MB** |

> LoRA'nin guzelligi: Sadece 50 MB'lik adapter paylasılır.
> Kullanici baz modeli zaten indirmis olabilir.

### Egitim sirasinda loss artarsa ne yapmali?

- **Aniden artis:** Learning rate cok yuksek olabilir. LR'yi azaltin.
- **Yavas artis:** Overfitting baslıyor olabilir. Dropout artirin veya erken durdurun.
- **Dalgali:** Normal. Batch seviyesinde dalgalanma beklenir. Ortalamayi takip edin.

### Eval WER dusmuyor, ne yapmali?

1. Eval dataset'i cok mu kucuk? 2000+ ornek kullanin
2. Eval dataset train'den farkli mi? Farkli kariler olabilir
3. Yeterli adim oldu mu? 5000+ adimdan sonra belirgin dusus beklenir
4. Augmentation cok mu agresif? Speed perturbation oranlarini kontrol edin

---

## 15. Kaynaklar ve Ileri Okuma

### Akademik Makaleler

- **Whisper Paper:** Radford, A. et al. (2022). "Robust Speech Recognition via
  Large-Scale Weak Supervision." OpenAI.
  https://arxiv.org/abs/2212.04356

- **LoRA Paper:** Hu, E. et al. (2021). "LoRA: Low-Rank Adaptation of Large
  Language Models." Microsoft Research.
  https://arxiv.org/abs/2106.09685

- **QLoRA Paper:** Dettmers, T. et al. (2023). "QLoRA: Efficient Finetuning of
  Quantized Large Language Models."
  https://arxiv.org/abs/2305.14314

### Resmi Dokumanlar

- **HuggingFace Transformers:**
  https://huggingface.co/docs/transformers

- **HuggingFace PEFT:**
  https://huggingface.co/docs/peft

- **HuggingFace Datasets:**
  https://huggingface.co/docs/datasets

- **PyTorch Mixed Precision:**
  https://pytorch.org/docs/stable/amp.html

### Faydali Blog Yazilari

- **HuggingFace: Fine-Tune Whisper with LoRA:**
  https://huggingface.co/blog/fine-tune-whisper

- **OpenAI: Introducing Whisper:**
  https://openai.com/research/whisper

### Turkce Kaynaklar

- **Derin Ogrenme Turkiye:** https://deeplearningturkiye.com
- **HuggingFace Turkce Toplulugu:** HuggingFace'te "turkish" arayin

### Ilgili Projeler

- **Tilavet (bu proje):** https://github.com/baristiran/Tilavet
- **WhisperKit (iOS):** https://github.com/argmaxinc/WhisperKit
- **Faster Whisper:** https://github.com/SYSTRAN/faster-whisper

---

## 16. Sonuc ve Oneriler

### Ogrenilen Dersler

1. **Buyuk model = daha iyi sonuc:** whisper-large-v3 ile %1.58 WER elde edildi
2. **LoRA yeterli:** Tam fine-tuning gereksiz, %0.85 parametre yeterli
3. **GPU bellegi yonetilebilir:** Gradient checkpointing + fp16 ile 20/80 GB
4. **Her optimizasyon her yerde calismaz:** QLoRA Whisper'da sorunlu
5. **Checkpoint kaydi hayat kurtarir:** 12 saatlik egitimde baglanti kopma riski yuksek
6. **Versiyon sabitleme sart:** Kutuphaneler hizla degisiyor, pip freeze kullanin
7. **Hata yapmak normaldir:** 17 hata yasandı, hepsi cozduk ve ogrendik

### Gelecek Calismalari

- [ ] Model'i HuggingFace Hub'a yukle
- [ ] CoreML donusumu (iOS/macOS)
- [ ] FastAPI inference server
- [ ] 108 dilde ceviri entegrasyonu
- [ ] SpecAugment data augmentation denemesi
- [ ] Whisper-large-v3-turbo modeli ile karsilastirma

### Ogrenciler Icin Pratik Oneriler

1. **Kucukten baslayin:** Ilk denemenizi whisper-small ile yapin (ucretsiz Colab yeterli)
2. **Notebook'u okuyun, anlayarak calistirin:** Her hucrenin ne yaptigini bilin
3. **Hata yaptığinizda panik yapmayin:** Hata mesajini okuyun, arastirin
4. **HuggingFace dokumanlarini okuyun:** En guncel bilgi orada
5. **Deney yapin:** r degerini, batch size'i, learning rate'i degistirin ve ne oldugunu gorun
6. **Sonuclarinizi kaydedin:** TensorBoard veya basit bir tablo ile
7. **Topluluga katilin:** HuggingFace forumu, Discord, GitHub Issues
8. **Acik kaynak katkida bulunun:** Bu projeye Pull Request gonderin!

### Bir Sonraki Adiminiz Ne Olmali?

```
1. Bu dokumani okuyun (tamamdiniz!)
2. Egzersiz 1-2'yi yapin (HuggingFace + inference)
3. Egzersiz 3'u yapin (WER hesaplama)
4. Colab'da whisper-small ile kucuk bir egitim deneyin
5. Sonuclarinizi analiz edin
6. Daha buyuk modele gecin (medium -> large-v3)
7. Kendi projenizi baslatin!
```

---

**Bu dokuman Tilavet projesi kapsaminda hazirlanmistir.**
**Lisans: MIT**
**Repo: https://github.com/baristiran/Tilavet**

> "Yapay zeka, dogru veri ve dogru yontemle muhtesem sonuclar verebilir.
> Sizin yapmaniz gereken, denemekten korkmamak." — T.I.A.
