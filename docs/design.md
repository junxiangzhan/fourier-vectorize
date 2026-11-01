# 共用工具庫設計

狀態：設計已確認，共用工具庫與前三題實驗已實作；驗證見 [稽核記錄](audit.md)。

完整題目要求、使用者故事與驗收條件見 [規格文件](spec.md)。

## 已確定的方向

- 主要實作語言為 Rust。
- 工具庫以 Assignment_1151 的需求為範圍，不追求完整的線性代數庫。
- 支援以矩陣形式計算第一題的三次 Bézier 曲線。
- 小型數學矩陣採固定尺寸泛型 `Matrix<const R: usize, const C: usize>`，以巢狀固定長度陣列儲存元素；相容的乘法尺寸由型別表達。
- 目前不建立自訂 `MatrixLike`／`VectorLike`。只有在出現具體共用需求、確實能簡化設計時才重新評估。
- 數學物件統一使用 `f64`，不另做元素型別泛型。
- `Vector<N>` 使用 `Matrix<N, 1>` 的直欄表示，共用矩陣儲存與基本運算；向量的 dot、norm 等運算集中在對應的特殊尺寸實作。
- `Point2` 為獨立型別，可以透過組合二維向量共用座標運算。點減點得到向量，點加向量得到點；midpoint 與 lerp 回傳點。
- 影像型別命名為 `Image`，可表示一般影像，不限於覆蓋率。光柵化流程的覆蓋率資料仍以 `0..1` 表示，並可依前景與背景顏色轉成彩色影像。
- `Image<T>` 採像素型別泛型，共用尺寸、儲存與像素存取；`Image<f64>` 表示覆蓋率影像，`Image<Color>` 表示彩色影像。
- `Color` 使用三個 `f64` RGB 通道，通常位於 `0..1`；轉成 BMP 時才限制範圍、四捨五入並量化為 8-bit。
- 小型數學矩陣固定尺寸；影像尺寸於執行時指定。
- 畫布原點位於左上角，x 向右、y 向下；像素 `(x, y)` 的取樣中心為 `(x + 0.5, y + 0.5)`。正旋轉角在畫面上呈現順時針方向。

## 職責分工

| 物件／模組 | 責任 |
| --- | --- |
| Matrix、Vector、Point2 | 數學與幾何運算 |
| CubicBezier、Path | 曲線求值、細分、子路徑與反向 |
| Rasterizer | non-zero winding 填色與 supersampling |
| Image | 影像資料、像素存取、測量列讀取；數值影像支援區塊平均 |
| Color、圖片輸出 | 顏色混合、通道轉換與 BMP 存檔 |

## 數學表示

採用橫列向量表示參數冪次時，曲線點可表示為：

`q(t) = [1, t, t², t³] × B × P`

其中 B 是 4 × 4 的 Bézier 基底矩陣；P 是 4 × 2 的控制點座標矩陣，每列對應一個控制點，兩欄分別為 x 與 y。結果為 1 × 2 的座標。

第一題還需要獨立的中點細分實作。第三題的 supersampling 需要放大座標、填色及像素區塊平均；矩陣乘法可以支援座標變換，但不是像素平均的必要前提。

## BMP 格式查核

自行輸出未壓縮 24-bit `BI_RGB` BMP，不需要圖檔編碼庫；此方向已確認。

- 使用 14-byte 檔案標頭與 40-byte `BITMAPINFOHEADER`，不含調色盤時像素資料從第 54 byte 開始。
- 每個像素依 B、G、R 順序輸出，各通道為 8-bit。
- 每列像素資料補齊為 4-byte 的倍數，檔案大小須包含補齊資料。
- 高度為正數時，像素列依 bottom-up 順序輸出；高度為負數時依 top-down 順序輸出。
- 作業未指定 BMP，也未要求自製 BMP 標頭；此輸出格式源自本次設計需求。

來源：[BITMAPFILEHEADER](https://learn.microsoft.com/en-us/windows/win32/api/wingdi/ns-wingdi-bitmapfileheader)、[BITMAPINFOHEADER](https://learn.microsoft.com/en-us/windows/win32/api/wingdi/ns-wingdi-bitmapinfoheader)、[24-bit BGR](https://learn.microsoft.com/en-us/windows/win32/wmdm/-bitmapinfoheader)。

## 已確認的完整介面

以下介面已由使用者確認並完成實作。原先首次範圍為工具庫、必要測試與 BMP 示範；本次依使用者核准的檢查點，另完成前三題整合、測量與 PDF 報告。

### math：固定尺寸數學物件

- `Matrix<R, C>` 的資料為 `[[f64; C]; R]`，按列儲存；提供 `from_rows`、`zeros`、`transpose`、索引、同尺寸加減與純量乘除。
- `Matrix<R, K> * Matrix<K, C>` 回傳 `Matrix<R, C>`；乘法不相容的尺寸無法通過編譯。
- `Matrix<N, N>::identity()` 建立單位矩陣。
- `Vector<N>` 是 `Matrix<N, 1>` 的別名，不複製乘法實作。直欄矩陣另提供 `dot`、`norm_squared`、`norm`。
- `Vector<4>` 轉置得到橫列矩陣，可用於 Bézier 參數冪次。

### geometry：位置、曲線與路徑

- `Point2` 包住 `Vector<2>`，提供 `new`、`x`、`y`、`midpoint`、`lerp`、`distance`、繞指定中心旋轉與縮放。
- 運算結果遵循 `Point2 - Point2 -> Vector<2>`、`Point2 + Vector<2> -> Point2`、`Point2 - Vector<2> -> Point2`。
- `distance_to_segment(point, start, end)` 計算點到線段距離；投影限制在線段範圍內。零長度線段視為一個點。
- `CubicBezier::new([Point2; 4])` 建立三次曲線；`eval(t) -> Point2` 使用矩陣形式，內部可預先計算 `B * P`。
- `CubicBezier::split_midpoint() -> (CubicBezier, CubicBezier)` 使用中點細分，不依賴公式取樣結果。
- `CubicBezier::flatten(depth) -> Result<Polyline, Error>` 產生指定深度的折線，保留首尾端點；深度 0 為兩端點。
- `Subpath` 具有起點、`line_to`／`cubic_to` 段及開放／封閉狀態。起點與連續的段命令表達路徑，避免要求呼叫端重複提供每段起點。
- `Path` 保存多個子路徑，提供 `flatten(depth)`、`reversed()`、旋轉及縮放。
- `Polyline` 保存頂點與封閉狀態；`FlattenedPath` 保存多個折線子路徑。
- 路徑反向改變遍歷順序；三次曲線的兩個控制點互換。所有幾何位置保持不變。
- 第一題使用開放路徑；填色需要明確封閉的子路徑。

### image：一般影像與顏色

- `Image<T>` 使用執行時指定的寬高及連續的 `Vec<T>`，按列儲存像素；`T` 不需要實作矩陣或向量介面。
- `Image::new(width, height, initial_pixel) -> Result<Image<T>, Error>` 建立填滿初始值的影像，建立時需要 `T: Clone`。
- `width`、`height`、`get(x, y)`、`get_mut(x, y)`、`row(y)` 提供共用存取；查詢越界回傳 `None`。
- `map` 由像素轉換建立同尺寸的另一種影像，例如 `Image<f64> -> Image<Color>`；回傳 Result，以保留配置失敗的錯誤。
- `Image<f64>::mean_downsample(k)` 逐個不重疊的 `k × k` 區塊平均。`k` 必須為正整數，寬高必須可整除；輸出尺寸為原尺寸除以 `k`。
- `Color::rgb(r, g, b)` 與 `Color::mix(other, amount)` 提供 RGB 值與通道混合。
- 覆蓋率 `a` 轉色採 `(1 - a) * background + a * foreground`，由呼叫端選擇前景及背景。
- 作業流程直接平均數值通道，不加入 gamma 轉換或 alpha 通道。

### raster：填色與抗鋸齒

- `Rasterizer::new(width, height) -> Result<Rasterizer, Error>` 設定最終輸出尺寸。
- `render(path: &FlattenedPath, k: usize) -> Result<Image<f64>, Error>` 使用 non-zero winding 規則填色。
- `k = 1` 回傳二值覆蓋率；`k > 1` 將路徑座標與畫布尺寸放大 `k` 倍，執行同一填色流程，再逐塊平均。回傳影像始終為最終輸出尺寸。
- `winding_at(path, point)` 回傳帶符號繞數，供星形與圓環中心測量使用。在目前 y 向下的座標慣例中，順時針輪廓為正，逆時針為負。
- 填色使用像素中心；邊的 y 範圍採下界包含、上界排除，水平邊與零長度邊不產生 crossing。
- 同位置的 crossings 合併帶符號貢獻；填色區間採左界包含、右界排除，避免共同頂點被重複計數。
- 保留畫布外的邊參與繞數計算，僅在像素寫入時限制到畫布範圍。
- 開放子路徑、非有限座標、零尺寸、零取樣倍率與尺寸溢位回傳錯誤。

### bmp：自行輸出圖檔

- `Image<Color>::save_bmp(file_path) -> Result<(), Error>` 提供使用者方便的存檔入口。
- 實際編碼集中於 `bmp::write(image, writer)`，可輸出到檔案或記憶體，方便驗證。
- 預設 24-bit、未壓縮 `BI_RGB`、正高度、bottom-up；標頭整數以 little-endian 明確序列化，不直接寫入 Rust struct 的記憶體。
- 寫出 BGR 像素、每列的補齊 bytes、正確的檔案大小與像素偏移。
- 量化採 `round(clamp(channel, 0, 1) * 255)`；非有限顏色值、超出 BMP 欄位範圍的尺寸與 I/O 失敗回傳錯誤。

### 作業流程與測量

- 第一題：矩陣公式取樣與中點細分各自產生點／折線；誤差測量為公式曲線取樣點到折線最近線段距離的最大值。公式取樣的相鄰點距離須小於 0.1 px。
- 曲線公式取樣、誤差統計、計時及測量檔案由作業程式組合工具庫提供的運算。
- 第二題：建立星形或多子路徑圓環，flatten 後送入 Rasterizer，另查詢中心繞數。
- 第三題：對相同幾何圖案與最終尺寸使用 `k = 1, 2, 4, 8`，由 `Image<f64>::row` 取得覆蓋率測量列。
- 覆蓋率影像轉為彩色影像後輸出 BMP；數值測量可保存為 CSV。

### 錯誤處理與驗證

- 型別保證矩陣乘法尺寸；一般索引依 Rust 慣例處理越界，Image 的查詢方法則回傳 Option。
- 影像／幾何／輸出操作使用具體的 `Error`，區分無效參數、非有限資料、尺寸或細分長度溢位與 I/O 失敗。
- 驗證矩陣乘法及轉置、Bézier 端點與獨立 Bernstein 計算、中點細分前後曲線的一致性。
- 驗證退化線段、星形中心繞數、內圈反向前後的圓環與孔洞、掃描線通過頂點時的結果。
- 驗證 `k = 1` 不改變結果，以及已知覆蓋率區塊的平均。
- 驗證有補齊資料的 BMP 與上下左右顏色不同的測試影像，使用獨立解碼器確認通道與列順序。

## 交付狀態

介面、預設 BMP 格式、共用工具庫及前三題場景、測量、回答與 PDF 已完成。可依 [README](../README.md) 重現；各票證據見 [檢查點](checkpoints.md) 與 [稽核記錄](audit.md)。
