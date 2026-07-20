# TSPLIB Vision–LLM Benchmark

Bu proje, görsel-dil modellerinin bir Traveling Salesman Problem (TSP) örneğini yalnızca düğüm görselinden çözme, geçersiz rotaları onarma ve geçerli rotaları iyileştirme yeteneklerini karşılaştırır.

Deneylerde TSPLIB `eil51` problemi kullanılmıştır:

- Düğüm sayısı: `51`
- Mesafe tipi: `EUC_2D`
- Bilinen optimum tur uzunluğu: `426`

## Araştırma akışı

Proje dört aşamalı bir deney düzeni kullanır:

1. **Vision zero-shot:** Model, düğüm görselinden doğrudan bir tur üretir.
2. **Critic–scorer repair:** Critic geçersiz rotayı onarır; scorer mevcut ve aday rota arasında seçim yapar.
3. **Critic–scorer optimization:** Geçerli rotanın mesafesi multi-agent yapı ile düşürülmeye çalışılır.
4. **Hibrit 2-opt:** Multi-agent sonucuna deterministik 2-opt uygulanır.

Python doğrulama katmanı her aday rota için şu koşulları denetler:

- Rota düğüm `1` ile başlayıp düğüm `1` ile bitmelidir.
- `1–51` arasındaki her düğüm tam bir kez ziyaret edilmelidir.
- Kapalı tur toplam `52` tamsayı içermelidir.
- Geçersiz veya daha kötü adaylar kabul edilmemelidir.

## Sonuçlar

| Yöntem | Geçerli | Mesafe | Optimuma göre gap |
|---|:---:|---:|---:|
| OpenRouter Nemotron zero-shot | Evet | 1308 | %207,04 |
| Groq Qwen zero-shot | Hayır | — | — |
| Groq Qwen critic–scorer repair | Evet | 620 | %45,54 |
| Groq Qwen critic–scorer optimization | Evet | 620 | %45,54 |
| Groq rotası + deterministik 2-opt | Evet | **439** | **%3,05** |
| TSPLIB optimum | Evet | 426 | %0,00 |

Groq zero-shot çıktısında `6`, `42` ve `48` düğümleri eksikti. Critic–scorer repair aşaması bu rotayı ilk iterasyonda geçerli hâle getirdi ve `620` mesafeli bir tur üretti. Sonraki critic–scorer optimizasyon aşamasında mesafe düşürülemedi. Deterministik 2-opt ise rotayı `620` değerinden `439` değerine indirdi.

Bu nedenle `439` sonucu yalnızca multi-agent başarısı olarak değil, **LLM tabanlı rota onarımı + klasik yerel arama** şeklindeki hibrit yöntemin sonucu olarak değerlendirilmelidir.

## Proje yapısı

```text
tsplib-vision-llm-benchmark/
├── data/
│   └── tsplib/
│       ├── eil51.tsp
│       └── eil51.opt.tour
├── src/
│   ├── analysis/
│   │   └── build_comparison.py
│   ├── core/
│   │   ├── inspect_tsplib.py
│   │   └── tsp_utils.py
│   ├── optimization/
│   │   └── run_two_opt.py
│   ├── providers/
│   │   ├── groq/
│   │   │   ├── run_zero_shot.py
│   │   │   ├── run_multi_agent.py
│   │   │   └── run_optimize.py
│   │   └── openrouter/
│   │       ├── run_zero_shot.py
│   │       └── run_multi_agent.py
│   └── visualization/
│       ├── plot_eil51.py
│       └── plot_route_results.py
├── tests/
│   └── test_tsp_utils.py
├── output/
│   ├── figures/
│   ├── results/
│   ├── checkpoints/
│   └── archive/
├── .env.example
├── pytest.ini
├── requirements.txt
└── README.md
```

## Kurulum

Python `3.11` önerilir.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ortam değişkenleri için:

```powershell
Copy-Item .env.example .env
```

`.env` içeriği:

```env
OPENROUTER_API_KEY=
OPENROUTER_MODEL=nvidia/nemotron-nano-12b-v2-vl:free

GROQ_API_KEY=
GROQ_MODEL=qwen/qwen3.6-27b
```

Gerçek API anahtarları yalnızca `.env` dosyasında tutulmalıdır. `.env` Git tarafından yok sayılır.

## Veri ve test kontrolü

```powershell
python -m src.core.inspect_tsplib
python -m pytest -q
```

Beklenen test sonucu:

```text
9 passed
```

## Görsel oluşturma

```powershell
python -m src.visualization.plot_eil51
```

Üretilen temel görsel:

```text
output/figures/eil51_nodes.png
```

## Deney komutları

### OpenRouter zero-shot

```powershell
python -m src.providers.openrouter.run_zero_shot
```

### OpenRouter critic–scorer

```powershell
python -m src.providers.openrouter.run_multi_agent `
  --iterations 3 `
  --delay 10
```

### Groq zero-shot

```powershell
python -m src.providers.groq.run_zero_shot
```

### Groq critic–scorer repair

```powershell
python -m src.providers.groq.run_multi_agent `
  --iterations 3 `
  --delay 10
```

### Groq critic–scorer optimization

```powershell
python -m src.providers.groq.run_optimize `
  --iterations 5 `
  --delay 10 `
  --patience 2
```

Bu aşama iki ardışık iterasyonda iyileşme olmazsa erken durur.

### Deterministik 2-opt

```powershell
python -m src.optimization.run_two_opt
```

### Karşılaştırma özeti

```powershell
python -m src.analysis.build_comparison
```

Üretilen özetler:

```text
output/results/summary/eil51_comparison.json
output/results/summary/eil51_comparison.csv
output/figures/eil51_method_comparison.png
```

## Deney kayıtları

JSON kayıtlarında deney türüne göre şu bilgiler tutulur:

- kullanılan model,
- rota ve geçerlilik durumu,
- mesafe ve optimalite boşluğu,
- critic ve scorer kararları,
- API çağrı sayıları,
- token kullanım bilgileri,
- başlangıç ve güncelleme zaman damgaları,
- hata ve checkpoint bilgileri.

Şu an critic, scorer ve iterasyon bazında kesin çalışma süresi ayrıca ölçülmemektedir. Zaman damgaları toplam süre için yaklaşık bilgi verir; `--delay`, yeniden deneme ve `--resume` araları bu farkı etkileyebilir.

## Temel bulgular

- Görsel zero-shot üretim, `eil51` gibi 51 düğümlü bir problemde rota geçerliliğini garanti etmemiştir.
- Critic–scorer yapı, eksik düğümlü rotayı başarıyla onarmıştır.
- Aynı multi-agent yapı, geçerli rota elde edildikten sonra ek mesafe iyileştirmesi sağlayamamıştır.
- Deterministik 2-opt, LLM tarafından onarılan rotayı optimuma çok yakın bir seviyeye taşımıştır.
- En güçlü sonuç, LLM ile yapılandırılmış rota üretimi/onarma ve klasik optimizasyonun birlikte kullanıldığı hibrit yaklaşımdır.

## Sınırlılıklar

- Deney yalnızca `eil51` üzerinde yürütülmüştür.
- Ücretsiz API sağlayıcılarının model erişimi ve rate limitleri değişebilir.
- Her model için sınırlı sayıda tekrar yapılmıştır.
- LLM çıktıları aynı promptla dahi değişkenlik gösterebilir.
- `439` sonucu doğrudan LLM çıktısı değil, LLM rotasına uygulanan 2-opt sonucudur.

## Güvenlik

Commit öncesinde gizli anahtar kontrolü yapılmalıdır:

```powershell
git grep -n -E "gsk_|sk-or-|AIza" -- .
git check-ignore -v .env
```

İlk komut hiçbir gerçek API anahtarı döndürmemelidir. İkinci komut `.env` dosyasının `.gitignore` tarafından yok sayıldığını göstermelidir.
