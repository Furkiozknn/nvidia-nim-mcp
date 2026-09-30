# Tasarım: nvidia-nim-mcp ilk kullanım ve README yenilemesi (30 Eylül 2026)

## Hedef

Profilden ya da videodan gelen biri ilk dakikada şunu yapabilmeli: aracın ne yaptığını tek cümlede anlamak, tek komutla kaydetmek, sunucunun gerçekten başladığını görmek; anahtar yoksa ya da yanlışsa ne yapacağını mesajdan okumak. Araçların davranışı, model zincirleri, çıkış sözleşmesi (araçlar hâlâ hata durumunda metin döndürür, `is_error` kullanılmaz) ve mevcut 80 test değişmedi; sürüm numarası artmadı (0.1.0).

## Önce / sonra

| Konu | Önce | Sonra |
|---|---|---|
| README ilk ekranı | banner, altı satır rozet, içindekiler, yedi araç tablosu; kurulum üçüncü adımda `uv run --project /path/…` | banner, tek cümle, tek `claude mcp add … uvx --from git+…` komutu, 17 sn gerçek çıktılı terminal demosu, "ne zaman kullanılır / kullanılmaz" tablosu, sonra eski gövde |
| Sunucu açılışı (`initialize`) | 11,8-14,5 s (`import litellm` ≈ 10 s) | 3,4-3,7 s; litellm ilk sohbet çağrısında ya da el sıkışmadan sonra arka plan iş parçacığında yükleniyor |
| `nvidia-nim-mcp --help` / `--version` | yok sayılıyor, sunucu sessizce stdin bekliyor | yardım (kayıt komutu, hangi anahtar neyi açar, `output/` nerede) / sürüm; terminalde (stdin TTY) çalıştırılırsa yardım basıp çıkar; bilinmeyen argüman: çıkış 2 + `--help` yönlendirmesi |
| Başlangıç günlüğü | yok | stderr'e tek satır: sürüm, NVIDIA anahtarı var/yok, anahtarı olan ücretsiz sağlayıcıların adı (groq, mistral…), `output/` yolu. Anahtar değeri ve değişken adı yazılmaz (CodeQL ilk sürümdeki değişken-adı listesini "clear-text logging" diye işaretlemişti) |
| Sunucu sürümü | `initialize` sürümü boş | `nvidia-nim 0.1.0` |
| "Anahtar yok" mesajı | "Set NVIDIA_API_KEY in .env, or any of…" | aynı + nereden alınır (build.nvidia.com) + "bu MCP sunucusunun ortamına (kayıt `env`'i) ya da `nvidia_image.py`'nin yanındaki `.env`'e" |
| Anahtar var ama reddedildi (401/403) | "All … failed or timed out." | aynı satır + "A provider rejected its API key … `check_provider_health`'i çalıştır"; `generate_image` satırları `HTTP 401 (key rejected…)`; `check_provider_health` satırları da. **Diğer bütün hata mesajları bayt bayt aynı** (mevcut testler gevşetilmedi) |
| `generate_image` anahtarsız sonucu | "fallback after NVIDIA models failed" (NVIDIA denenmedi) | "keyless tier, NVIDIA_API_KEY not set"; anahtar varken NVIDIA düşerse eski ifade |
| Araç açıklamaları | `describe_image` dosyanın yüklendiğini, `generate_image` istemin dışarı gittiğini söylemiyordu | ikisi de söylüyor (ajan bunu görüyor) |
| `uv.lock` | proje adı `nvidia-nim`, `--locked` düşüyor | yenilendi, CI'da `uv lock --check` |
| `server.json` | var olmayan "reranking" aracı | "health probe" |
| Sunucuyu denemek | istemci gerekiyordu | `scripts/sonda.py`: sunucuyu başlatır, `initialize` + `tools/list` yapar, isteğe bağlı araç çağırır; çıkış 0/1/2/3 |
| Test | 80 | 102 (+22: `tests/test_first_use.py`) |

## CLI akışı

```
kaydet      claude mcp add --transport stdio --env NVIDIA_API_KEY=YOUR_KEY nvidia-nim -- \
              uvx --from git+https://github.com/Furkiozknn/nvidia-nim-mcp nvidia-nim-mcp
            (anahtarsız: --env satırı çıkar; generate_image yine çalışır)
kontrol     nvidia-nim-mcp --help | --version
            python scripts/sonda.py --no-keys -- nvidia-nim-mcp        initialize ok, 7 araç, çıkış 0
            python scripts/sonda.py --no-keys --call ask_llm '{"question":"hi"}' -- nvidia-nim-mcp
teşhis      Claude içinde check_provider_health  -> hangi model/anahtar ölü, "key rejected" dahil
yanlış      anahtar yok  -> hangi anahtarlar işe yarar, nereden alınır, nereye konur
            anahtar reddedildi -> "HTTP 401 (key rejected…)" + düzeltme satırı
```

## Tasarım kararları

- **Tembel `litellm`.** `litellm = _LazyLitellm()` modül gibi davranır (okuma ve yazma gerçek modüle gider), yani `litellm.acompletion` çağrıları ve testlerdeki `monkeypatch.setattr(nvidia_image.litellm, "acompletion", …)` değişmeden çalışır; 80 eski test sıfır değişiklikle geçti. `main()` yüklemeyi arka plan iş parçacığında başlatır: el sıkışma hemen yanıtlanır, ilk sohbet çağrısı çoğunlukla beklemez.
- **Yalnızca 401/403 ayrı anılır.** Zincirdeki bir sağlayıcı 401/403 verirse (NVIDIA HTTP yolu ya da litellm `AuthenticationError`/`PermissionDeniedError`) araç mesaja tek satır ekler; başka her hata ("model emekli", 429, 503, zaman aşımı) eski mesajı aynen döndürür. Sınır: litellm yedek zincirinde tüm sağlayıcılar düşerse yüzeye çıkan istisna zincirin bir ucundan gelir; reddedilen anahtar orta bir halkadaysa ipucu görünmeyebilir (`check_provider_health` her durumda satır satır gösterir).
- **`--help` bir argparse değil, sabit metin.** Tek bayrak çifti var (`--help`, `--version`); argparse eklemek 30 satır ve bir bağımlılık yüzeyi olurdu. TTY'de çalıştırıldığında sunucu başlatılmaz: bir MCP istemcisi olmayan bir terminalde başlatmanın anlamı yok.
- **`is_error` yok.** Araçlar başarısızlığı normal metin sonucu olarak döndürmeye devam ediyor; istemcilerin bunu nasıl işlediği değiştirilirse bu bir davranış değişikliği olurdu (çıkış sözleşmesi bozulmaz kuralı).
- **`sonda.py` depoda.** Yalnızca demo için değil: README "başladı mı" sorusuna bununla cevap veriyor, testlerden biri onu gerçek bir sunucu başlangıcında koşturuyor (tek başlangıç ≈ 4 s).

## Görsel dil (video sisteminden alınanlar)

Demo, `mcp-vet` yenilemesinde kurulan FRK-OS terminal sahnesini olduğu gibi kullanır (`scripts/demo-uret.py` o betikten uyarlandı): `sosyal/uret/tema.mjs` `klasik.akis` renkleri (`zemin #0e0d0b`, `panel #14120e`, `yazi #f1ece2`, vurgu `#ffc21a`), yardımcı vurgular `#ff4d6d`, `#ff7a1a`, `#19d3e6`, `doku: "izgara"` (48 px, `rgba(241,236,226,.045)`), JetBrains Mono (`assets/yazi/`, SIL OFL 1.1, OFL metni yanında), 30 ms/harf yazma, satır satır çıktı. Renk yalnızca boyamadır, hiçbir karakteri değiştirmez: `initialize ok:` / `tools/list:` camgöbeği, `no provider configured.` turuncu, `sonda:` hata öneki mercan. Bilerek alınmayanlar: League Gothic başlık (README'de görsel başlık yok), geçişler (bir terminal demosunda çıktı okunmalı). `prefers-reduced-motion`'da imleç yanıp sönmesi kapalı.

Kontrast (panel `#14120e` üstünde, WCAG göreli parlaklıktan; aynı palet `mcp-vet` yenilemesinde hesaplanmıştı): krem 15,9:1, sönük metin `#b6ae9d` 8,5:1, sarı 11,6:1, mercan 5,8:1, turuncu 7,2:1, camgöbeği 10,3:1; hepsi ≥ 4,5:1.

## Kararlar ve sınırlar

- **Banner değişmedi.** `assets/banner.svg` profil deposunun üreticisinden gelir; elle değiştirmek bir sonraki üretimde silinir.
- **README'de üreticisi olmayan medya yoktu**, çıkarılan bir şey yok. Yeni GIF (`docs/demo/demo.gif`, 1,1 MB) `scripts/demo-uret.py` + `demo-kayit.js` ile yeniden üretilebilir; `komutlar.txt` aynı kaydın düz metni. Dikey 1080x1920 sessiz kayıt depoya girmedi (`*.mp4` `.gitignore`'da); günlük video hattı için `sosyal/medya/projeler/nvidia-nim-mcp/terminal.mp4`.
- **Canlı NIM ölçümü yok** (anahtar ortamda tanımlı değil). "Model zincirleri gerçekten çalışıyor mu" sorusuna bu yenileme cevap vermiyor; `check_provider_health` bunun için var.
- `claude mcp add …` ve `uv sync --extra local-embeddings` çalıştırılmadı (ilki kullanıcı yapılandırmasını değiştirir, ikincisi ~1 GB torch). Sürüm, etiket, PyPI, dizin/awesome-list başvurusu, Pages, GitHub description/homepage **yapılmadı** (onay kapısı).
