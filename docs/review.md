# 雙軸程式審查

使用者指定基準：`6a3c23775cc2e1dc7c4ce2f9bef16eb35c854a25`。初審提交：`22958be`，比較命令 `git diff 6a3c237...22958be`。Standards、Spec 由兩個獨立代理並行審查。下列原始程式片段與行號指向初審提交；修正後的檔案已不同。

## Standards

Documented-standard breaches:

1. **[P2] Preserve allocation errors throughout fallible geometry operations.** README promised “配置失敗…有具體錯誤.” `src/path.rs:99`: `let mut vertices = vec![self.start];` allocates infallibly. At line 170, `.collect::<Result<Vec<_>, _>>()?` propagates element errors but cannot return outer-vector allocation errors. Supersampling also calls `FlattenedPath::scaled_about`, whose nested `.collect()` calls at lines 189 and 192 allocate infallibly. Memory pressure can abort instead of returning `Error::Allocation`. Reserve fallibly and propagate failures through the flatten/render chain.

2. **[P2] Remove the unhandled sorting allocation.** Same README rule. `src/raster.rs:124`: `crossings.sort_by(|a, b| a.0.total_cmp(&b.0));`. Stable sorting can allocate auxiliary storage, bypassing fallible reservations. Equal-crossing order does not affect the grouped signed sum, so `sort_unstable_by` avoids this extra allocation. See [Rust slice documentation](https://doc.rust-lang.org/std/primitive.slice.html#method.sort_unstable_by).

3. **[P2] Keep audit and provenance validation active under Python optimization.** README says the PDF generator rejects changed sources/artifacts. `scripts/build_report.py:144`: `assert hashlib.sha256((OUT/name).read_bytes()).hexdigest() == digest`. Run-status and source-hash checks also use assertions. They disappear with `python -O` or `PYTHONOPTIMIZE`; changed evidence can produce a report. Likewise, `scripts/audit.py:95`: `assert identity == run["source_fingerprint"]`, and remaining assertions disappear while a passed manifest is still written. Use explicit checks that raise exceptions.

Baseline judgment calls: no additional smell warrants a finding.

### 修正與驗證

- flatten 的初始頂點及子路徑集合採 try_reserve。supersampling 使用內部 fallible 幾何縮放；配置失敗向上保留 `Error::Allocation`，不經 infallible 公開變換入口。
- scanline 改用 sort_unstable_by，保留交點分組及 signed sum，避免額外排序配置。
- 全部稽核及 PDF 安全閘改為 require／ValueError，在 -O 模式仍執行。新增三個回歸測試，分別證明過期來源、變動成果、變動來源被拒絕，且不產生 passed manifest／PDF。
- 修正後完整 Rust 13 個整合測試＋1 compile-fail、fmt／Clippy、3 個 Python 回歸測試均通過。以 -O 重跑真實稽核及 PDF 成功；新來源指紋 `2d67dc85d9f3e840`，46 個非計時成果可在新資料夾重現。

修正提交：`a9c3f3e`。Standards 代理直接複核目前檔案，確認三項初審問題全部解決：flatten／render fallible reservation、無額外配置的排序，以及在 Python 最佳化模式仍有效的 require／ValueError。此聚焦複核沒有殘留問題；代理未重跑測試，實際測試由主代理執行並記錄於稽核文件。

## Spec

**No findings.** Reviewed against `docs/spec.md`, checkpoints 01–10, design decisions, formal tickets, and original assignment pages 1–5. The authorized Parts 1–3 scope is satisfied:

- Independent matrix Bézier evaluation and midpoint subdivision; dense sampling, correctly directed segment-distance error, measured convergence.
- Three-segment join comparison, five-vertex self-intersecting star, consistent non-zero winding scanlines, reversal of only the inner Bézier circle.
- Shared rotated star for supersampling and timing; isolated near-end row across the shallowest edge; equal-angle wheel wedges/gaps.
- Raw measurements, source/artifact traceability, reproducible execution and a readable report with required figures/answers.

The rendered report’s curve, joins, measurement, wheel and environment pages were inspected. Shallow curve differences, row transition and central wheel aliasing are visible. Presentation uses computed Part 1 points/segments, measurement plots and Rust bitmaps within permitted exceptions. No missing requirement, scope creep or concrete correctness defect warranted a finding.

修正後 Spec 聚焦複核：無發現。配置修改保留幾何，等位置交點先累加再填區段，因此排序方式不改變 scanline 行為；明確 Python 檢查維持最佳化模式的稽核／報告完整性。沒有引入規格退步，沒有阻礙交付的問題。

初審總計：Standards 3 項（最高 P2，全部已修正並複核，剩餘 0）；Spec 0 項（無未符合需求）。
