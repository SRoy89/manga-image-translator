# Hướng dẫn sử dụng Auto Manga Pipeline

`auto_manga` là lớp tự động hóa nằm bên ngoài core `manga-image-translator`. Pipeline thực hiện:

```text
MangaDex URL/UUID hoặc JSON manifest mẫu
→ gọi source adapter để đọc manga và danh sách chapter
→ chọn chapter cần xử lý
→ tải ảnh theo đúng thứ tự
→ kiểm tra ảnh đầu vào
→ gọi manga-image-translator
→ dịch sang tiếng Việt
→ kiểm tra output
→ lưu trạng thái vào SQLite để có thể resume
```

Chỉ sử dụng pipeline với nguồn mà bạn có quyền truy cập và được phép tải. Adapter hiện tại không hỗ trợ và không cố vượt qua paywall, CAPTCHA, đăng nhập, anti-bot hoặc các cơ chế hạn chế truy cập.

## 1. Yêu cầu môi trường

Core của repository hiện hỗ trợ Python 3.10 và 3.11. Nên dùng Python 3.11.

Tại thư mục gốc của repository:

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Trên Windows PowerShell:

```powershell
py -3.11 -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Kiểm tra CLI:

```bash
python main.py --help
python -m manga_translator local --help
```

## 2. Cấu hình

Config mặc định nằm tại:

```text
auto_manga/config.yaml
```

Nội dung mặc định:

```yaml
storage:
  raw: "./data/raw"
  translated: "./data/translated"

download:
  timeout: 20
  retries: 3
  delay: 1.0
  concurrency: 3

sources:
  mangadex:
    translated_languages:
      - en
      - ja
      - ko
      - zh
      - zh-hk
    data_saver: false

translation:
  translator: "deepseek"
  target_language: "VIN"

database:
  path: "./data/state.db"
```

Ý nghĩa:

- `storage.raw`: nơi lưu ảnh gốc đã tải.
- `storage.translated`: nơi lưu ảnh đã dịch.
- `download.timeout`: timeout cho mỗi HTTP request, tính bằng giây.
- `download.retries`: số lần thử lại sau lần request đầu tiên.
- `download.delay`: khoảng cách tối thiểu giữa các request, tính bằng giây.
- `download.concurrency`: số ảnh tối đa được xử lý đồng thời.
- `sources.mangadex.translated_languages`: các ngôn ngữ được lấy khi duyệt manga MangaDex.
- `sources.mangadex.data_saver`: `false` dùng ảnh gốc; `true` dùng ảnh nén data-saver.
- `translation.translator`: translator của core, ví dụ `deepseek`, `chatgpt` hoặc `gemini`.
- `translation.target_language`: mã ngôn ngữ đích. Tiếng Việt là `VIN`.
- `database.path`: file SQLite dùng để lưu trạng thái.

`storage.raw` và `storage.translated` bắt buộc phải khác nhau. Các đường dẫn tương đối được resolve theo thư mục hiện tại khi chạy command.

Muốn dùng config riêng:

```bash
python main.py manga "$MANGA_URL" --chapters 1-5 --config ./my-config.yaml
python main.py resume --config ./my-config.yaml
```

Khi dùng config riêng, các lần chạy mới và `resume` phải trỏ tới cùng một `database.path`.

## 3. API key

Không đặt API key trong YAML, source code hoặc command line. Core đọc key từ environment.

DeepSeek:

```bash
export DEEPSEEK_API_KEY="your-key"
```

OpenAI/ChatGPT:

```bash
export OPENAI_API_KEY="your-key"
```

Gemini:

```bash
export GEMINI_API_KEY="your-key"
```

PowerShell:

```powershell
$env:DEEPSEEK_API_KEY = "your-key"
```

Chỉ cần khai báo key của translator đang dùng. Pipeline không ghi key vào SQLite và không truyền key qua argument của subprocess.

## 4. Nguồn được hỗ trợ

### MangaDexSource

`MangaDexSource` dùng các endpoint public của API chính thức và MangaDex@Home. Không cần API key MangaDex, không đăng nhập và không vượt qua chapter bị hạn chế hoặc nằm ở website ngoài.

Các dạng input được hỗ trợ:

```text
https://mangadex.org/title/<manga-uuid>
https://mangadex.org/title/<manga-uuid>/<slug>
https://mangadex.org/chapter/<chapter-uuid>
<raw-uuid>
```

Ví dụ:

```bash
python main.py manga \
  "https://mangadex.org/title/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee/example" \
  --chapters 1-5

python main.py chapter \
  "https://mangadex.org/chapter/11111111-2222-3333-4444-555555555555"

python main.py manga "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee" --latest
python main.py chapter "11111111-2222-3333-4444-555555555555"
```

UUID phải đúng format có dấu gạch ngang. Với raw UUID, command `manga` hoặc `chapter` cung cấp loại resource; adapter không gọi API chỉ để đoán.

Khi chạy `manga`, `translated_languages` lọc feed. Nếu cùng volume, chapter number và ngôn ngữ có nhiều release, adapter chọn theo thứ tự cố định:

1. Entity version cao hơn.
2. `readableAt`/ngày public mới hơn.
3. UUID nhỏ hơn theo thứ tự từ điển nếu vẫn hòa.

Chapter external hoặc không có page public bị bỏ qua. Chapter number null được giữ như một special với ID ổn định. Các ngôn ngữ/volume khác nhau dùng storage key riêng nên không đè folder nhau. Command `chapter` xử lý chính chapter được chỉ định, không áp dụng bộ lọc ngôn ngữ của manga feed.

Adapter gửi User-Agent nhận diện project, giới hạn request API ở mức thận trọng, giới hạn At-Home metadata theo quota công bố và tôn trọng `Retry-After` khi nhận HTTP 429. SSL verification luôn bật.

Chỉ xử lý nội dung bạn có quyền tải và dịch. Việc MangaDex trả page public không tự động cấp quyền tái phân phối; người dùng vẫn phải tuân thủ attribution và điều khoản MangaDex/scanlation group.

### ExampleSource

`ExampleSource` là adapter tham chiếu đọc JSON manifest public qua HTTP hoặc HTTPS, không phải crawler HTML tổng quát.

### Manga manifest

Ví dụ `manifest.json`:

```json
{
  "id": "public-manga-id",
  "title": "Public Domain Manga",
  "chapters": [
    {
      "id": "chapter-1",
      "number": "1",
      "title": "Chapter 1",
      "pages": [
        "https://public.example/manga/chapter-1/001.jpg",
        "https://public.example/manga/chapter-1/002.jpg"
      ]
    },
    {
      "id": "chapter-2",
      "number": "2",
      "title": "Chapter 2",
      "pages": [
        {
          "index": 1,
          "image_url": "https://public.example/manga/chapter-2/001.jpg"
        },
        {
          "index": 2,
          "image_url": "https://public.example/manga/chapter-2/002.jpg"
        }
      ]
    }
  ]
}
```

Yêu cầu đối với manifest:

- Manga phải có `id` và `title`.
- Mỗi chapter phải có `id` và `number`.
- Chapter ID không được trùng nhau.
- `pages` phải có ít nhất một ảnh.
- Page URL chỉ được dùng HTTP hoặc HTTPS.
- Page index phải là số nguyên, không âm và không được trùng.
- Page có thể là URL string hoặc object gồm `index` và `image_url`.

Nếu chapter không có `url`, adapter tự tạo URL dạng:

```text
https://public.example/manifest.json#chapter=chapter-1
```

### Direct chapter manifest

Command `chapter` cũng chấp nhận JSON document riêng có các field `manga`, `chapter` và `pages`:

```json
{
  "manga": {
    "id": "public-manga-id",
    "title": "Public Domain Manga",
    "source_url": "https://public.example/manifest.json"
  },
  "chapter": {
    "id": "chapter-3",
    "number": "3",
    "title": "Chapter 3"
  },
  "pages": [
    "https://public.example/manga/chapter-3/001.jpg",
    "https://public.example/manga/chapter-3/002.jpg"
  ]
}
```

## 5. Các command sử dụng

### Dịch một khoảng chapter

```bash
python main.py manga "https://public.example/manifest.json" --chapters 1-5
python main.py manga "$MANGADEX_TITLE_URL" --chapters 1-5
```

Khoảng chapter dựa trên field `number`, bao gồm cả hai đầu. Chapter thập phân cũng được hỗ trợ, ví dụ:

```bash
python main.py manga "$MANGA_URL" --chapters 10-12.5
```

### Dịch một chapter theo số

```bash
python main.py manga "$MANGA_URL" --chapters 12
```

### Dịch chapter mới nhất

```bash
python main.py manga "$MANGA_URL" --latest
```

Nếu chapter number là số, chapter có giá trị lớn nhất được chọn. Nếu không có chapter number dạng số, adapter dùng chapter cuối trong manifest.

### Dịch tất cả chapter

Không truyền `--chapters` hoặc `--latest`:

```bash
python main.py manga "$MANGA_URL"
```

### Dịch trực tiếp một chapter

Dùng MangaDex chapter URL hoặc raw UUID:

```bash
python main.py chapter "$MANGADEX_CHAPTER_URL"
python main.py chapter "$MANGADEX_CHAPTER_UUID"
```

Dùng direct chapter manifest của ExampleSource:

```bash
python main.py chapter "https://public.example/chapter-3.json"
```

Hoặc dùng fragment từ manga manifest:

```bash
python main.py chapter \
  "https://public.example/manifest.json#chapter=chapter-3"
```

Nên đặt URL có ký tự `#`, `&` hoặc `?` trong dấu nháy.

### Chọn source adapter rõ ràng

```bash
python main.py manga "$MANGADEX_TITLE_URL" --source mangadex --chapters 1-5
python main.py manga "$MANGA_URL" --source example --chapters 1-5
```

Các source hiện có là `mangadex` và `example`. Thông thường không cần truyền `--source` vì registry tự chọn.

### Resume

```bash
python main.py resume
```

Nếu dùng config riêng:

```bash
python main.py resume --config ./my-config.yaml
```

### Debug logging

`--verbose` là option cấp cao nhất nên phải đứng trước subcommand:

```bash
python main.py --verbose manga "$MANGA_URL" --chapters 1-5
python main.py --verbose resume
```

## 6. Cấu trúc file output

Ví dụ manga có title `Public Domain Manga`:

```text
data/
├── raw/
│   └── Public Domain Manga/
│       ├── chapter-001/
│       │   ├── 001.jpg
│       │   ├── 002.jpg
│       │   └── 003.jpg
│       └── chapter-002/
│           ├── 001.jpg
│           └── 002.jpg
├── translated/
│   └── Public Domain Manga/
│       ├── chapter-001/
│       │   ├── 001.jpg
│       │   ├── 002.jpg
│       │   └── 003.jpg
│       └── chapter-002/
│           ├── 001.jpg
│           └── 002.jpg
└── state.db
```

Manga title và chapter number được sanitize trước khi tạo folder. Unicode được giữ lại nếu an toàn. Các ký tự nguy hiểm, path traversal và tên file đặc biệt của Windows được xử lý.

## 7. Download và validation

Downloader thực hiện:

- Sắp xếp page theo `Page.index`.
- Đặt tên tuần tự `001.jpg`, `002.jpg`, ...
- Tải vào file `.part`, sau đó replace file đích theo cách atomic.
- Không tải lại file đã tồn tại và là ảnh hợp lệ.
- Thử lại timeout, connection reset, HTTP 408, HTTP 429 và HTTP 5xx.
- Không retry các lỗi HTTP cố định như 403 hoặc 404.
- Nếu nhiều page dùng cùng một URL, ảnh chỉ được tải một lần rồi copy sang vị trí còn lại.
- Phát hiện và log ảnh có nội dung trùng.
- Thay thế ảnh zero-byte hoặc ảnh hỏng.
- Xóa page đánh số bị thừa so với manifest hiện tại.

Chapter chỉ chuyển sang `downloaded` khi:

- Folder raw tồn tại.
- Có ít nhất một ảnh.
- Đúng số lượng page dự kiến.
- Tên file tạo thành sequence liên tục.
- Tất cả file là ảnh đọc được và lớn hơn 0 byte.

Nếu một ảnh fail sau toàn bộ retry, chapter được đánh dấu `failed`; các chapter sau vẫn tiếp tục.

## 8. Translation

Wrapper gọi CLI hiện có của repository bằng argument list, không dùng shell:

```text
python -m manga_translator local \
  -i RAW_CHAPTER_FOLDER \
  -o TRANSLATED_CHAPTER_FOLDER \
  --config-file TEMP_CONFIG.json
```

Temporary config có dạng:

```json
{
  "translator": {
    "translator": "deepseek",
    "target_lang": "VIN"
  }
}
```

Trước khi chạy translator:

- Output hợp lệ đã có được giữ lại.
- Output zero-byte hoặc hỏng bị xóa để core tạo lại.
- Output page thừa bị xóa.
- Partial output hợp lệ được giữ để core chỉ xử lý page còn thiếu.

Chapter chỉ chuyển sang `translated` khi output có đúng số ảnh, đúng sequence và mọi ảnh đều hợp lệ.

## 9. SQLite state và resume

File `state.db` có các trạng thái chapter:

```text
pending
downloading
downloaded
translating
translated
failed
```

Ý nghĩa:

- `pending`: đã biết chapter nhưng chưa tải xong.
- `downloading`: đang tải ảnh.
- `downloaded`: raw images đã được kiểm tra thành công.
- `translating`: đang chạy manga-image-translator.
- `translated`: translated output đã được kiểm tra thành công.
- `failed`: chapter gặp lỗi; nội dung lỗi được ghi vào database.

Khi chạy `resume`:

- `downloading` được đưa về `pending` rồi filesystem được kiểm tra lại.
- Nếu toàn bộ raw images đã tải xong trước khi chương trình dừng, chúng được tái sử dụng.
- Nếu chỉ tải được một phần, file hợp lệ được giữ và chỉ page thiếu/hỏng được tải lại.
- `translating` được đưa về `downloaded`.
- Partial translated output không được tin tuyệt đối; page hợp lệ được giữ, page thiếu/hỏng được xử lý lại.
- `translated` cũng được kiểm tra lại. Nếu output mất hoặc hỏng, chapter được đưa về `downloaded` hoặc `pending` để sửa.
- `failed` được thử lại.
- Chapter `translated` có raw/output hợp lệ không chạy lại.

Không chạy đồng thời nhiều process `manga`, `chapter` hoặc `resume` dùng chung một `state.db`.

## 10. Logging và exit code

Ví dụ output:

```text
[INFO] Manga: Public Domain Manga
[INFO] Found 20 chapters
[INFO] [1/5] Chapter 1
[INFO] [DOWNLOAD] 12 pages
[INFO] [DOWNLOAD] completed
[INFO] [TRANSLATE] deepseek -> VIN
[INFO] [TRANSLATE] completed
[INFO] Summary: selected=5 translated=5 skipped=0 failed=0
```

Exit code của `auto_manga`:

- `0`: toàn bộ chapter được xử lý hoặc skip thành công.
- `1`: có ít nhất một chapter fail.
- `2`: lỗi config, URL/source hoặc tham số chapter.

Core `manga-image-translator` có một số đường lỗi vẫn trả exit code `0`. Vì vậy wrapper không chỉ kiểm tra exit code mà còn kiểm tra đầy đủ translated output.

## 11. Xử lý lỗi thường gặp

### `No module named colorama`, `cv2`, `torch`, ...

Dependency của core chưa được cài hoặc đang dùng sai virtual environment:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

Đảm bảo đang dùng Python 3.10 hoặc 3.11:

```bash
python --version
```

### `DEEPSEEK_API_KEY environment variable required`

Khai báo key trong cùng terminal trước khi chạy:

```bash
export DEEPSEEK_API_KEY="your-key"
python main.py resume
```

### `No source adapter can handle this URL`

Kiểm tra URL/UUID. MangaDex chỉ nhận HTTPS URL trên `mangadex.org` với path `title/<uuid>` hoặc `chapter/<uuid>`, hoặc raw UUID hợp lệ. `ExampleSource` nhận HTTP(S) JSON manifest; local path và website HTML thông thường không được hỗ trợ.

### `ExampleSource expected a JSON document`

URL không trả về JSON manifest đúng format. Kiểm tra response của URL và cấu trúc ở mục 4.

### `Two chapters cannot share the same storage path`

Hai chapter khác ID nhưng `number` tạo ra cùng tên folder, ví dụ `1` và `1.0`. Sửa manifest để chapter number/folder không bị trùng.

### Chapter liên tục chuyển sang `failed`

Chạy debug:

```bash
python main.py --verbose resume
```

Kiểm tra:

- URL ảnh có còn truy cập được không.
- Nguồn có trả HTTP 403/404/429 không.
- API key có đúng không.
- Disk còn dung lượng và có quyền ghi không.
- `storage.raw`, `storage.translated` và `database.path` có đúng không.
- Core có tải được model cần thiết không.

Thông tin lỗi gần nhất cũng được lưu trong column `chapters.error` của SQLite.

## 12. Kiểm tra database thủ công

Nếu có SQLite CLI:

```bash
sqlite3 data/state.db \
  "SELECT chapter_number, status, error FROM chapters ORDER BY id;"
```

Chỉ nên xem dữ liệu. Không sửa status thủ công khi pipeline đang chạy.

## 13. Chạy test

Test của subsystem không gọi network thật hoặc API trả phí:

```bash
python -m unittest discover -s test/auto_manga -p 'test*.py' -v
```

Import và syntax checks:

```bash
python -m compileall -q auto_manga main.py test/auto_manga
python -c "import auto_manga.main; import auto_manga.pipeline.orchestrator"
```

## 14. Những phần chưa hỗ trợ

- Crawler HTML cho website manga thực tế.
- MangaDex authentication hoặc chapter không có page public trên MangaDex@Home.
- Authentication hoặc nội dung yêu cầu đăng nhập.
- Paywall, CAPTCHA hoặc anti-bot bypass.
- Watch mode tự phát hiện chapter mới.
- Chạy nhiều pipeline process đồng thời trên cùng database.
- Phát hiện source thay đổi nội dung page nhưng vẫn giữ nguyên URL và số lượng page.
- Xác minh chất lượng/ngữ nghĩa bản dịch; validation hiện kiểm tra file ảnh và số lượng output.

Muốn hỗ trợ một nguồn hợp pháp mới, implement `MangaSource` trong `auto_manga/crawler/sources/` và đăng ký adapter trong `SourceRegistry`. Không đặt parsing source-specific vào orchestrator hoặc downloader.
