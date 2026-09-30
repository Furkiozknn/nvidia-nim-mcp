# Denetim: nvidia-nim-mcp (30 Eylül 2026)

Yenilemeden önce `master` (0.1.0, `da9618a`) üzerinde, bu makinede (Windows 11, 8 GB RAM, GPU yok, Python 3.11/3.12/3.14 uv ile, uv 0.12.5, Git Bash) ölçüldü. Ölçülmeyen bir şey yazılmadı. Ham çıktılar depo dışında: `kanit/nvidia-nim-mcp/{once,sonra}/` (`olc.py` komut listesini koşturur, `sonda.py` gerçek stdio el sıkışması yapar).

**Canlı NIM çağrısı yapılmadı.** Bu ortamda `NVIDIA_API_KEY` (ve diğer sağlayıcı anahtarları) tanımlı değil; anahtar istenmedi, girilmedi. NVIDIA'nın modelleri gerçekten yanıt veriyor mu, kota ne kadar, hangi model şimdi ölü: **ölçülmedi**. Anahtarsız yollar gerçek koşuldu; "anahtar var ama reddediliyor" yolu **mock** ile gösterildi (aşağıda açıkça işaretli).

## Temiz ortamda kurulum ve ilk sonuç

Her satır boş bir uv önbelleğiyle (`UV_CACHE_DIR` boş klasör), sağlayıcı anahtarı olmayan ortamda koşuldu.

| Yol | Önce (`master`) | Sonra (`yenileme/arayuz`) |
|---|---|---|
| `uv sync --group dev` (boş önbellek) | 13,8 s | 13,5 s |
| `uvx --from git+https://github.com/Furkiozknn/nvidia-nim-mcp nvidia-nim-mcp --help` (boş önbellek, stdin kapalı) | 36,6 s, **çıktı yok** (`--help` yok sayılıp sunucu başlıyor, stdin EOF'unda çıkıyor) | 27,1 s, yardım metni |
| aynısı, sıcak önbellek | 18,2 s, çıktı yok | 7,8 s |
| `uv run nvidia-nim-mcp --version` | 12,5 s, çıktı yok | 4,2 s, `nvidia-nim-mcp 0.1.0` |
| **sunucu `initialize` yanıtı** (gerçek stdio, 3 koşu) | **12,9 / 14,5 / 11,8 s** (ilk denemede 12,3 s) | **3,4 / 3,5 / 3,7 s** |
| `python -m venv` + `pip install -e .` + `pip install pytest pytest-asyncio` (CONTRIBUTING'deki pip yolu) | — | 7,7 + 79,1 + 3,9 s; `pytest` 102 geçti (26 s) |

"Tek komut, bir dakikada ilk sonuç": `uvx` ile soğuk kurulum 27-37 s, ardından sunucu 3,5 s'de yanıt veriyor; tutuyor. Ana neden `import litellm`: tek başına ~10 s (`python -X importtime`: 9,8-10,3 s), sunucunun 13 s'lik açılışının çoğu. MCP istemcileri `initialize` için sınırlı süre bekler; 12-14 s, yavaş bir makinede ya da soğuk diskte zaman aşımına yakındı. `nvidia-nim-mcp --version` hâlâ ~4 s (modülü içe aktarmak `mcp.server`'ı çekiyor, ~4 s); bunu düşürmek ayrı bir başlatıcı modülü gerektirirdi, yapılmadı.

## README komutları

| Komut | Sonuç |
|---|---|
| `uv sync` / `uv sync --group dev` | çalıştı (13,8 s). **`uv.lock` bayattı**: proje adı hâlâ `nvidia-nim`, `uv lock --check` ve `uv sync --frozen/--locked` "Missing workspace member" ile düşüyordu; CI düz `uv sync` ile her seferinde yeniden çözüyordu. `uv lock` ile yenilendi, CI'ya `uv lock --check` eklendi |
| `uv run pytest`, `pytest tests/test_api_key_guard.py`, `pytest -q` | çalıştı: 80 geçti, 17,5 s (README'deki "80 tests" koşudan tutuyor) |
| `claude mcp add --transport stdio nvidia-nim -- uv run --project /path/to/this/repo nvidia_image.py` | **çalıştırılmadı** (kullanıcı yapılandırmasını değiştirir). Aynı çağrı `uv run --project . nvidia_image.py` olarak koşuldu: sunucu başlıyor. `.env` yalnızca klondaki `nvidia_image.py`'nin yanında bulunuyor (`load_dotenv()` çağıran dosyanın yolundan yukarı arar); `uvx` ile kurulumda görünmez, README bunu söylemiyordu |
| `uv sync --extra local-embeddings` | **çalıştırılmadı** (torch ~1 GB, 8 GB RAM). Yolun kendisi anahtarsız ve extra'sız koşuldu: `create_embedding` "no local fallback available (run `uv sync --extra local-embeddings`...)" diyor |
| `.env` / `NVIDIA_NIM_OUTPUT_DIR` / `output/` | `generate_image` anahtarsız gerçekten koşuldu (Pollinations.ai, **NIM değil**, üçüncü taraf): `output/pollinations_….jpg` 9,6 KB yazıldı, 1,0-3,5 s |
| README örnek kullanım (düz dil istekleri) | model seçimi istemciye ait; sunucu tarafında yedi aracın hepsi `tools/list`'te ve açıklamalı |
| "mcp-vet verdict LOW" | doğru (aşağıda) |
| "80 tests" (`project-meta.json` özeti dahil) | koşudan çıkıyor; yenilemeden sonra 102 |

Sunucu `server.json` (MCP Registry taslağı) açıklamasında var olmayan bir "reranking" aracı geçiyordu; kaldırıldı (`health probe` yazıldı). Yayın/başvuru yapılmadı.

## `mcp-vet` ile araç ve kaynak denetimi

`mcp-vet audit --offline --path .` (0.6.0, yenileme dalı): **LOW**, çıkış 1; önce de sonra da aynı. Ekosistem denetimi konusunda (#19) bu depoya ait açık bulgu yok (tek bulgu profil deposunun kendi `ci.workflows` ayrışması; meta-source ayrışması olduğu için dokunulmadı).

Bulgular ve değerlendirme:
- `filesystem.read` + ağ: `describe_image` kullanıcının verdiği dosyayı base64'leyip NVIDIA'ya ya da yapılandırılmış yedek sağlayıcıya yüklüyor. Gerçek bir yetenek, kodda uzantı listesi (jpg/jpeg/png/webp) ve 10 MB sınırı var. **Araç açıklaması bunu söylemiyordu** (yalnızca "Analyze/describe a local image"); açıklamaya "dosya NVIDIA'ya ya da yapılandırılmış yedek sağlayıcıya yüklenir; yalnızca jpg/jpeg/png/webp, 10 MB" eklendi. Aynı şekilde `generate_image` açıklamasına istemin NVIDIA'ya ya da Pollinations.ai'ye gittiği eklendi.
- `environment.read` + ağ: beş sağlayıcı anahtarı, her biri kendi sağlayıcısına gidiyor; `mcp-vet` bunu LOW ve "API istemcisinin kendi anahtarı" diye sınıflıyor.
- MEDIUM "runtime'da paket kuruyor": `tests/test_output_dir.py` içindeki bir yorum satırı (gönderilen sunucu dışında).
- Araç açıklamalarında enjeksiyon/"tool poisoning" bulgusu yok; yedi açıklamanın hiçbiri ajanı başka bir araca ya da bir eyleme yönlendirmiyor.

## Hata mesajları

Anahtarsız (gerçek koşu, `sonda.py`, 5 anahtar da ortamdan silinmiş):

| Araç | Önce | Sorun |
|---|---|---|
| `ask_llm`, `translate_text`, `describe_image`, `check_content_safety` | "no provider configured. Set NVIDIA_API_KEY in .env, or any of GROQ_API_KEY…" | Nereden alınacağı yok; `uvx` kurulumunda `.env` hiç okunmuyor, yani "in .env" yanlış yönlendiriyordu |
| `create_embedding` | "NVIDIA embedding failed (NVIDIA_API_KEY not set in .env - NVIDIA tier skipped) and no local fallback available (run `uv sync --extra local-embeddings`…)" | anlaşılır |
| `generate_image` | çalışıyor, ama sonuç "fallback after NVIDIA models failed" diyordu; NVIDIA hiç denenmemişti | yanıltıcı |
| `check_provider_health` | NVIDIA satırları "not configured (NVIDIA_API_KEY not set)" | iyi |
| `nvidia-nim-mcp --help` | yardım yok; sunucu başlıyor, stdin'i sessizce bekliyor | **hata**: terminalde çalıştıran kişi boş bir imleçle kalıyor |

Anahtar var ama yanlış: bütün araçlar "All … failed or timed out." diyordu; 401 ile "model emekli" ya da "kota doldu" ayırt edilemiyordu. `check_provider_health` satırı yalnızca `HTTP 401`. (Bu yol **mock** ile denendi: `kanit/nvidia-nim-mcp/sonra/yanlis-anahtar-mock.txt`, sağlayıcı yerine 401 dönen bir taslak; hiçbir istek dışarı çıkmadı, gerçek sağlayıcı davranışı ölçülmedi.)

Sonraki durum `TASARIM.md`'de.

## Testler

`uv run pytest`: 80 geçti (17,5 s), CI matrisi py3.11 ve 3.14. Tümü mock; ağ ve anahtar gerektirmiyor. Yenileme sonrası 102 (bkz. `TASARIM.md`).

## README bulguları

- İlk ekran rozet ve içindekiler tablosuydu; kurulum komutu üçüncü adımdı, `uv run --project /path/to/this/repo` biçiminde ve `uvx` yolu yoktu.
- "Zero cost" rozeti ve "no credit card": NVIDIA'nın ücretsiz katmanı için doğru olarak duruyor; doğrulanmadı (anahtar yok). Cerebras için kodda "Payment required" notu var (kod yorumu), README tablosu Cerebras'ı yine de "free-tier" gruplandırıyor; bilinen tutarsızlık, olduğu gibi bırakıldı.
- Üreticisi olmayan medya yoktu (`assets/*.svg` profil deposunun üreticisinden; dokunulmadı).
- "Every model in these chains was confirmed working with a real request" (2026-08-22 notu) 5+ hafta önceki bir ölçüm; bugün doğrulanamadı (anahtar yok).
