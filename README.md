# Assignment 1151：Parts 1-4

Rust 共用工具庫與同一個實驗程式，涵蓋 Bézier、non-zero winding 填色、supersampling，以及 Part 4 的向量化實驗（希爾伯特掃描與傅立葉輪廓、DBSCAN 分群、還原成直線／圓弧／Bézier、挖洞、彩色），另附旋轉實驗（Transformations，另成一份不繳交的 PDF）。檢查點見 [docs/checkpoints.md](docs/checkpoints.md)，驗證記錄見 [docs/audit.md](docs/audit.md)。Part 5、整份作業打包與上傳另行處理。

## 建置與執行

本次在 Windows x86_64、Rust / Cargo 1.92.0、MSVC 工具鏈執行。建議相同或更新 Rust stable；Windows 需 Rust MSVC 所用的 linker。其他平台尚未實測。

在專案根目錄執行（不加子命令時等同 `all`，舊的寫法照常可用）：

```powershell
cargo build --offline --release
python scripts/decode_paintings.py
cargo run --offline --release -- all --out output --size 384 --wedges 96 --paintings assets/paintings
```

### 子命令

| 指令 | 做什麼 |
|---|---|
| `all [--paintings 資料夾]` | Parts 1-4 全部實驗；有 `--paintings` 時對資料夾內每個 BMP 跑名畫展示。`run.json` 標為 complete，可供稽核。 |
| `part1` … `part4` | 只跑一個部分；`run.json` 標為 partial，稽核會拒絕（避免拿部分結果當完整成果）。 |
| `paint 圖.bmp… [--factor N] [--methods k24_stack,lab1266,…]` | 名畫展示（調色盤、CMY、Lab 向量化），可只選部分做法。 |
| `hifi --paintings 資料夾 [--factor N]` | 高精度比較實驗（像素邊界與平滑輪廓；DFT、精修、僅梯度下降；面積項掃描）。 |
| `hifi 圖.bmp… [--factor N] [--levels 32,16,16] [--fit dft\|refined\|descent]` | 以傅立葉級數擬合單張圖，只輸出係數：`export_<圖名>_<fit>_coefficients.csv`（每條輪廓的每個 c_k）與 `export_<圖名>_<fit>.json`（尺寸、各階數值、ΔE／SSIM）。預設 refined。 |
| `render 係數.csv… [--scale N] [--svg] [--bmp]` | 讀取 `hifi` 輸出的係數檔，畫成可直接用瀏覽器開啟的 Lab SVG 與／或 BMP；`--scale N` 直接由級數放大 N 倍（1..16），不經點陣重取樣。預設兩種都輸出。 |

任何 JPEG 放進資料夾後，以 `python scripts/decode_paintings.py 資料夾` 轉成 BMP（Pillow 只做解碼）。`--factor N` 以 N×N 區塊平均縮小；不給時自動選最小的倍數，使長邊不超過 1000 px（`paint`、`all`）或 2000 px（`hifi`），對本作業的兩幅畫正好是原本的設定（Gleizes《女人與孩子》2 與 1、《創造亞當》4 與 2）。`paint` 與 `hifi` 不改動 `run.json`。

名畫展示每幅畫跑九種做法：Lab k-means 調色盤 K=8、16、24（剪貼式）、24 色堆疊式（避免色塊間露出底色，含清理版）、CMY 8+8+8，以及 Lab 8+8+8 與 12+6+6（含清理版）。Lab 版另輸出 `*_browser.svg`：三版分放紅綠藍通道，再以 SVG 濾鏡（feColorMatrix、立方、feColorMatrix、sRGB gamma）重組，可直接用瀏覽器開啟；以 `python -m http.server` 服務專案根目錄後開啟 `scripts/svg_check.html?stem=paint_adam_lab1266`，可逐像素比較瀏覽器與本程式的結果。平常的整合測試不跑名畫。

高精度比較實驗約需 15 分鐘（20 執行緒），稽核會自動檢查 hifi_* 檔。兩幅畫以較高解析度、Lab 32+16+16 階、每條輪廓約 n/3 個諧波（上限 1024），直接由傅立葉級數取樣填回，不轉成圖元。比較兩種輪廓：像素邊界與「雙邊濾波平滑後以 marching squares 取次像素等值線」；並比較三種係數：DFT、DFT 再以 chamfer 梯度下降精修、僅從圓開始梯度下降。面積項（Chan–Vese）掃描記錄於 `hifi_region_sweep.csv`。

### 圖片與 GitHub Actions

只有公有領域的《創造亞當》（`assets/paintings/adam.jpg`）納入 Git，來源見 [assets/paintings/SOURCES.md](assets/paintings/SOURCES.md)；其他圖片只在本機使用。`.github/workflows/experiments.yml` 在每次 push 時於 Ubuntu 建置、測試、跑全部實驗與名畫展示、執行獨立稽核，並把 `output/` 上傳為 artifact；手動觸發時可勾選加跑高精度實驗。報告 PDF 不在 Actions 中產生。

沒有外部 Rust crate，也不需要 Python 或 Node.js。所有場景由同一個執行檔產生，原始控制點、覆蓋率、摘要、圖表與文字回答存入 `output/`。成功時 `run.json` 的 status 為 complete；途中失敗則保留 in-progress，不將舊成果宣稱為完成。再次使用相同輸出目錄會覆寫同名實驗檔；需要保留不同測量時請換 `--out`。

```powershell
target/release/multimedia.exe --help
cargo test --offline
cargo fmt --check
cargo clippy --offline --all-targets -- -D warnings
```

CLI 畫布允許 128..2048，輪盤允許 40..512 個楔形；預設 384、96。畫布下限供有效像素列選段，上限限制高倍率實驗的記憶體負擔。一般 `Image<T>` 不受此實驗尺寸範圍限制。

## 程式庫

| 模組 | 提供能力 |
| --- | --- |
| math | 固定尺寸 `Matrix<R,C>`、`Vector<N> = Matrix<N,1>`，乘法、轉置、單位矩陣、內積與長度 |
| geometry | `Point2`、矩陣公式 `CubicBezier`、獨立中點分割、密集取樣與點到線段距離 |
| path | 直線／cubic 子路徑、開放／封閉狀態、反向、變換及 flatten |
| raster | 每列排序 signed crossings，non-zero winding 填色、射線繞數查詢、supersampling |
| image | `Image<T>` 列／像素存取、fallible map、`Image<f64>` 區塊平均與雙線性旋轉重取樣、浮點 Color 混色 |
| bmp | 自製未壓縮 24-bit BMP，bottom-up、BGR、4-byte 補齊 |
| fourier | `Complex` 與 radix-2 FFT（含反轉換） |
| hilbert | 希爾伯特曲線 `d2xy`，走遍 2^m×2^m 網格、每步移到相鄰格 |
| cluster | DBSCAN 密度分群（網格加速鄰居搜尋），群數由資料決定 |
| fit | 直線（總體最小平方）、圓（Kåsa）、三次 Bézier（Schneider 參數修正）擬合與直線／圓交點 |

```rust
use multimedia::{CubicBezier, Point2, Image, Color};

fn main() -> Result<(), multimedia::Error> {
    let curve = CubicBezier::new([
        Point2::new(0.0, 0.0), Point2::new(0.0, 2.0),
        Point2::new(2.0, 2.0), Point2::new(2.0, 0.0),
    ]);
    let _polyline = curve.flatten(5)?;
    let image = Image::new(32, 32, Color::rgb(1.0, 1.0, 1.0))?;
    image.save_bmp("white.bmp")?;
    Ok(())
}
```

f64 用於全部數學、覆蓋率與 RGB；BMP 輸出才 clamp、四捨五入至 8-bit。位置減位置得到向量、位置加位移得到位置。原點左上，正 y 向下、正旋轉視覺上順時針。像素中心為 `(x+0.5,y+0.5)`；邊採下界包含／上界排除，寫入區間左含右不含。畫布外邊仍參與 winding，只裁切像素寫入。

基礎數學運算與 `eval` 保持一般 f64 算術；曲線取樣、flatten、填色與 BMP 等資料消費端驗證非有限輸入。回傳 Result 的資料處理流程將配置失敗等錯誤保留為 `Error`；不回傳 Result 的幾何建構／變換介面採一般 Rust 配置語意。均勻細分深度最多 20，密集取樣最多 `2^20+1` 點，避免巨量配置；無效尺寸、溢位及 I/O 失敗有具體錯誤。

## 獨立稽核與報告

實驗程式先自行完成覆蓋率與 BMP。以下 Python 工具只解碼、核對及排版，需 Python 3、支援 `get_flattened_data` 的 Pillow、ReportLab。本次使用 bundled Python 3.12.14 / Pillow 12.3.0 / ReportLab 4.4.9。

```powershell
python scripts/audit.py output
python scripts/build_report.py output --lang en
python scripts/build_report.py output --lang zh
python scripts/build_report.py output --lang en --part transformations
python scripts/build_report.py output --lang zh --part transformations
python scripts/build_part5.py output --lang en
python scripts/build_part5.py output --lang zh
python -m unittest discover -s scripts -p test_validation.py
```

稽核以獨立 Bernstein 公式與投影距離驗證曲線、使用 Pillow 解碼 BMP，檢查覆蓋率、取樣列、輪盤角寬、計時中位數等。`evidence_manifest.json` 保存來源／成果 SHA-256，並核對建置時嵌入的 FNV-1a 指紋。FNV 僅用於版本識別；SHA-256 用於檔案完整性。改動 Rust 後必須重新建置、執行、稽核；PDF 產生器會拒絕已變動的來源或成果。

繳交的 Parts 1-4 報告位於 `output/pdf/assignment_1151_parts_1_4_{en,zh}.pdf`（約 30 頁），Part 5 位於 `output/pdf/assignment_1151_part5_{en,zh}.pdf`；EMI 課程繳交英文版。報告的 Part 4 先回答作業的問題：以梯度下降擬合星形的傅立葉輪廓（與像素域的希爾伯特掃描對照），再分群、找角點並還原成直線、圓弧與 Bézier（`output/recover_vector.svg`）；之後是延伸實驗——洞、彩色、名畫、高精度、相關做法與開發紀錄（需要以 `all --paintings` 與 `hifi --paintings` 產生的資料）。旋轉實驗（Transformations）是 Part 4 清單上的另一個選項，另成不繳交的 `output/pdf/assignment_1151_transformations_{en,zh}.pdf`（`--part transformations`）。預設嵌入 Windows Microsoft JhengHei；其他平台可以指定 TrueType 字型：

```powershell
python scripts/build_report.py output --font path/to/regular.ttf --bold-font path/to/bold.ttf
```

報告的第一題圖像只顯示 Rust 計算的點／線段；數據圖只繪製已有 CSV。其餘圖使用 Rust BMP。報告工具沒有代算曲線覆蓋率、填色或抗鋸齒。

## 資料解讀

本次 384 畫布、96 楔形實驗的曲線最大樣本間距為 0.089990462 px；細分深度 5 首次誤差小於 0.5 px。星形中心 winding 為 +2，反向內圈的圓環中心為 0。取樣列 131、x=57..87 位於最平緩邊的起點附近，k=8 有 11 個部分覆蓋像素。這些數字均可重新驗證。

每個 k 計時三次，以 `Instant` 記錄縮放、配置、填色與平均的中位時間，排除輸出。時間隨機器與負載改變，因此重跑產生的新數據是該次報告依據。輪盤中心對 0.5 覆蓋率的 RMS 偏差只描述灰色均勻程度，不是理想輪盤重建誤差。

`target/`、`output/`、`tmp/` 不納入 Git；正式本地任務位於 `.scratch/assignment-1151/issues/`。報告可重建，需提交時依作業規範另加入身分資訊及其他題材料。
