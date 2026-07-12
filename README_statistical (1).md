## Classical & Deep Learning Statistical Models — ARIMA, SARIMA, Prophet, DLinear

### მოკლე შეჯამება (TL;DR)

4 მოდელი გამოვცადეთ — 3 classical statistical (ARIMA, SARIMA, Prophet) და 1 deep
learning (DLinear). **მთავარი აღმოჩენები:**

- **ARIMA** სუსტია holiday-heavy სერიებზე სეზონურობის იგნორირების გამო
- **SARIMA**-ს ეფექტი სერია-სპეციფიკურია — ერთსა და იმავე ფიქსირებულ `(P,D,Q,s)`-ს
  ერთ სერიაზე 78%-იანი გაუმჯობესება მოაქვს, მეორეზე — 71%-იანი გაუარესება
- **Prophet** ორივე ტესტირებულ სერიაზე საუკეთესო classical მოდელი აღმოჩნდა,
  თუმცა holiday-ების დამატებამ მოულოდნელად ოდნავ **გააუარესა** კი შედეგი
- **DLinear**-მა (ერთი გაზიარებული-წონებიანი მოდელი, ტრენინგდებოდა 271 სერიაზე
  ერთდროულად) ყველა classical მოდელს დაამარცხა Store 1/Dept 1-ზე — Prophet-ის
  საუკეთესო შედეგსაც კი ~3-ჯერ გადააჭარბა
- **ზოგადი დასკვნა:** classical per-series tuning არ არის scalable 3000+ სერიაზე;
  scalable global მოდელები (DLinear, და შემდგომში LightGBM/XGBoost) როგორც
  პრაქტიკულობით, ისე სიზუსტით სჯობს

---

### მეთოდოლოგია

**შეფასების მეტრიკა — WMAE (Weighted Mean Absolute Error):**

```
WMAE = Σ(wᵢ · |yᵢ - ŷᵢ|) / Σwᵢ,   სადაც wᵢ = 5 თუ holiday კვირაა, 1 — სხვა შემთხვევაში
```

Holiday კვირების 5-ჯერ მეტი წონა პირდაპირ გავლენას ახდენს მოდელის შერჩევასა და
loss-ის დიზაინზეც (იხ. DLinear-ის სექცია — weighted MAE პირდაპირ ტრენინგის loss-ადაც
გამოვიყენეთ).

**წარმომადგენლობითი სერიების შერჩევა — რატომ არა ყველა 3000+:**
Classical statistical მოდელები ცალკეული სერიისთვის მაღალი გამოთვლითი ღირებულებისაა
(განსაკუთრებით SARIMA `m=52` სეზონურ lag-ზე), ხოლო დავალების მიზანი თეორიული
გარჩევაა, არა საუკეთესო შესაძლო score-ის ძებნა. ამიტომ:

| მოდელი | ტესტირებული სერიები | მიზეზი |
|---|---|---|
| ARIMA / SARIMA | 2 (`Store 1/Dept 1`, `Store 20/Dept 92`) | ყველაზე ნელი — ხელით შერჩეული პარამეტრები |
| Prophet | 5 | სწრაფია, მეტ სერიაზე ეტევა |
| DLinear | 271 (ერთდროულად, `MAX_SERIES=300`-დან) | წონები გაზიარებულია — ერთი ტრენინგი მოიცავს ყველა სერიას |

**Experiment tracking:** ყველა run დალოგილია Weights & Biases-ზე:
[`walmart-forecasting-statistical`](https://wandb.ai/adane21-free-university-of-tbilisi-/walmart-forecasting-statistical)

---

### 1. ARIMA

**თეორია:** სამი კომპონენტი — AR(p) (რეგრესია საკუთარ წარსულზე), I(d)
(differencing stationarity-სთვის), MA(q) (რეგრესია წინა ცდომილებებზე). პარამეტრები
შეირჩა `pmdarima.auto_arima`-თი, AIC-ის მინიმიზაციით.

| Store | Dept | Order (p,d,q) | WMAE |
|---|---|---|---|
| 1 | 1 | (0, 0, 2) | 21 595.40 |
| 20 | 92 | (3, 1, 1) | 11 002.42 |

**დაკვირვება:** Store 20/Dept 92-ის ADF ტესტმა p-value = 0.5214 (>0.05) აჩვენა —
სერია non-stationary-ია, რაც ხსნის `auto_arima`-ს `d=1`-ის არჩევანს. Store 1/Dept 1-ზე
კი `d=0` აირჩა (საერთოდ არ დაადიფერენცირა) — საეჭვო არჩევანი holiday-heavy სერიაზე,
რაც სავარაუდოდ ხსნის შედარებით მაღალ (უარეს) WMAE-საც.

### 2. SARIMA

**თეორია:** ARIMA + სეზონური კომპონენტი `(P,D,Q,s)`. `s=52` წლიურ სეზონურობას
შეესაბამება (Christmas, Thanksgiving, Super Bowl, Labor Day ყოველწლიურად თითქმის
იმავე კვირაში მეორდება).

**მეთოდოლოგიური გადაწყვეტილება:** `auto_arima(seasonal=True, m=52)`-ის stepwise
search ძალიან ნელია — ამიტომ ორივე სერიისთვის ხელით დავაფიქსირეთ `(1,1,1)(0,1,1,52)`
(`D=1` სეზონური differencing-ისთვის, `Q=1` სეზონურ ცდომილებებზე).

| Store | Dept | ARIMA WMAE | SARIMA WMAE | ცვლილება |
|---|---|---|---|---|
| 1 | 1 | 21 595.40 | **4 772.71** | **-78%** |
| 20 | 92 | 11 002.42 | 18 788.34 | **+71%** |

**დაკვირვება — მთავარი თეორიული დასკვნა:** SARIMA-ს ეფექტი მკვეთრად საწინააღმდეგო
მიმართულებისაა ორ სერიაზე, რადგან ორივესთვის **იგივე** ფიქსირებული `(P,D,Q,s)`
გამოვიყენეთ. ეს ცხადყოფს, რომ სერია-სპეციფიკური tuning აუცილებელია SARIMA-სთვის —
რაც 3000+ სერიაზე გამოთვლითად რეალისტურად შეუძლებელია.

### 3. Prophet

**თეორია:** ფუნდამენტურად განსხვავებული მიდგომა — decomposable model
(`y = trend + seasonality + holidays`), არა ავტორეგრესიული. Stationarity არ არის
საჭირო, robust-ია missing data-ზე, built-in `holidays` პარამეტრი.

**Holiday dataframe:** იგივე 4 Walmart holiday, რაც `feature_engineering.py`-ის
`add_holiday_proximity`-შია (Super Bowl, Labor Day, Thanksgiving, Christmas),
`lower_window=-7`-ით.

**შედეგები (5 სერია, holidays-იანი ვერსია):**

| Store | Dept | WMAE |
|---|---|---|
| 1 | 1 | 3 332.51 |
| 20 | 92 | 14 349.13 |
| 4 | 38 | 6 748.82 |
| 33 | 5 | 62.97 |
| 10 | 72 | 13 727.32 |

**Holiday ablation (with vs without holidays):**

| Store | Dept | Prophet (no holidays) | Prophet (holidays) |
|---|---|---|---|
| 1 | 1 | **3 159.08** | 3 332.51 |
| 20 | 92 | **13 831.08** | 14 349.13 |

**დაკვირვება — მოულოდნელი შედეგი:** ორივე სერიაზე holidays-ის **გარეშე** ვერსია
ოდნავ სჯობდა. სავარაუდო მიზეზი: მხოლოდ 4 holiday ტიპი × 4 წელი (=16 თარიღი)
არასაკმარისია holiday ეფექტის საიმედოდ შესაფასებლად ასეთ მცირე sample-ზე —
`lower_window=-7` სავარაუდოდ noise-ს მატებდა სასარგებლო სიგნალის ნაცვლად.

### 4. DLinear (Deep Learning)

**თეორია:** [Zeng et al., 2023](https://arxiv.org/abs/2205.13504) — მარტივმა linear
მოდელმა შეიძლება გადააჭარბოს კომპლექსურ Transformer-ზე დაფუძნებულ არქიტექტურებს.
ორი ნაწილი: **Series Decomposition** (moving average-ით trend/seasonal გამოყოფა) და
**ორი Linear layer** (`lookback→horizon` mapping თითოეულ კომპონენტზე).

**მთავარი მეთოდოლოგიური განსხვავება:** წონები **გაზიარებულია** ყველა store-dept
სერიას შორის — ერთი მოდელის ტრენინგი ბევრ სერიაზე ერთდროულად, per-series fitting-ის
გარეშე. ეს პირდაპირ პასუხობს წინა სექციებში აღმოჩენილ scalability პრობლემას.

**Setup:** `lookback=52` კვირა, `horizon=13` კვირა, instance normalization (თითო
window თავისივე mean/std-ით), weighted MAE loss (WMAE-ს დიფერენცირებადი ვერსია,
holiday წონა 5), GPU (T4×2), early stopping (`patience=5`).

**შედეგები (271 სერია ერთდროულად, `MAX_SERIES=300`-დან):**

| მეტრიკა | მნიშვნელობა |
|---|---|
| Early stopping | epoch 12 (best val_loss_norm=0.5415) |
| აგრეგირებული WMAE (ყველა სერია ერთად) | 1 865.65 |
| საშუალო per-series WMAE | 1 864.05 |
| მედიანური per-series WMAE | 1 137.20 |
| **Store 1, Dept 1** (პირდაპირ შედარებადი) | **1 075.21** |

**დაკვირვება:** Aggregate WMAE დომინირებულია მაღალი მოცულობის სერიებით
(აბსოლუტური დოლარული ცდომილება), ამიტომ პირდაპირ არ ადარდება ცალკეულ classical
run-ებს — ამისთვის per-series breakdown გამოვთვალეთ (იხ. სექცია 5).

### 5. საბოლოო შედარება

**Store 1, Dept 1 — ყველა მოდელი:**

| მოდელი | WMAE |
|---|---|
| ARIMA | 21 595.40 |
| SARIMA | 4 772.71 |
| Prophet (holidays) | 3 332.51 |
| Prophet (no holidays) | 3 159.08 |
| **DLinear** (global, 271 სერიაზე ერთად) | **1 075.21** |

**Store 20, Dept 92 — classical მოდელები** (DLinear ვერ შემოწმდა ამ სერიაზე —
იხ. [შეზღუდვები](#შეზღუდვები)):

| მოდელი | WMAE |
|---|---|
| ARIMA | 11 002.42 |
| SARIMA | 18 788.34 |
| Prophet (holidays) | 14 349.13 |
| **Prophet (no holidays)** | **13 831.08** |

**Classical მოდელებს შორის** საუკეთესო არჩევანი სერიაზეა დამოკიდებული — ორივე
ტესტირებულ სერიაზე Prophet (holidays-ის გარეშე) გამოვიდა საუკეთესო, თუმცა SARIMA-მაც
Store 1/Dept 1-ზე ძალიან კარგი შედეგი აჩვენა.

**DLinear-ის შედეგი გამორჩეულია:** Store 1/Dept 1-ზე DLinear-მა — ერთდროულად 271
სერიაზე ტრენინგისას — ყველა classical მოდელი დაამარცხა, Prophet-ის საუკეთესო
შედეგსაც (3 159.08) ~3-ჯერ გადააჭარბა (1 075.21). ეს მიღწეულია ერთი **გაზიარებული**
მოდელით, ცალკეული per-series tuning-ის გარეშე.

### 6. ზოგადი დასკვნა და შემდეგი ნაბიჯები

არცერთი classical statistical მოდელი არ არის universally საუკეთესო — შედეგი
მკვეთრად სერია-სპეციფიკურია, და არცერთი მათგანი არ არის პრაქტიკულად scalable
3000+ სერიაზე იმ ფორმით, რომლითაც აქ გამოვცადეთ (თითოეულ სერიას საკუთარი
ოპტიმალური პარამეტრები სჭირდება).

DLinear-ის შედეგებმა ეს ვარაუდი დაადასტურა და ერთი ბიჯითაც წინ წავიდა: ერთმა
გაზიარებული-წონებიან linear მოდელმა, რომელიც 271 სერიაზე ერთდროულად ტრენინგდებოდა,
scalability პრობლემაც მოაგვარა და accuracy-შიც აჯობა ყველა classical მოდელს. ეს
ძლიერი მოტივაციაა **tree-based** (LightGBM/XGBoost) **და/ან deep learning global
მოდელებისკენ** გადასვლის საჭიროებისთვის — ანუ, ტიმმეითის XGBoost-ის მიდგომაც
სწორედ ამ დასკვნის ვალიდაციაა.

**შემდეგი ბუნებრივი ნაბიჯები** (თუ დრო/scope იძლევა): DLinear-ის `MAX_SERIES`-ის
გაზრდა სრულ დატასეტამდე, შედარება უფრო კომპლექსურ deep learning არქიტექტურასთან
(N-BEATS/PatchTST), და საბოლოო ჯამში `model_inference.ipynb`-ში საუკეთესო მოდელის
Model Registry-ში რეგისტრაცია.

### შეზღუდვები

- **Val split classical-ებში:** walk-forward CV (3 fold), DLinear-ში კი ერთჯერადი
  ბოლო-13-კვირიანი split — გამოთვლითი ტრადეოფის გამო, პირდაპირ 100%-ით
  შედარებადი არ არის მეთოდოლოგიურად, თუმცა ორივე იმავე horizon-ზეა (13 კვირა)
- **`Store 20/Dept 92` DLinear-ში ვერ შემოწმდა** — `MAX_SERIES=300`-ის
  შერჩევაში (default თანმიმდევრობით) არ მოხვდა
- **Prophet-ის holiday ablation** მხოლოდ 2 სერიაზეა ტესტირებული — 16 holiday
  თარიღი შესაძლოა არასაკმარისია სტატისტიკურად საიმედო დასკვნისთვის
- **DLinear-ის aggregate WMAE** დომინირებულია მაღალი მოცულობის სერიებით და არ
  წარმოადგენს ყველა სერიის თანაბარ შეწონილ საშუალოს

### რეპროდუცირება

**საჭირო გარემო:** Kaggle Notebook, competition input დამატებული
(`walmart-recruiting-store-sales-forecasting`), W&B ანგარიში (`adane21`).

| Notebook | დამატებითი დაყენება |
|---|---|
| `model_experiment_ARIMA_SARIMA.ipynb` | `pip install pmdarima wandb` |
| `model_experiment_Prophet.ipynb` | `pip install prophet wandb` |
| `model_experiment_DLinear.ipynb` | `pip install wandb` (torch წინასწარაა დაყენებული), რეკომენდირებულია GPU |

ყველა notebook-ს აქვს inline `data_prep`/`evaluation` helper ფუნქციები (repo-ს
`src/`-ის დამოუკიდებელი ასლი) და W&B Secrets-ზე დაფუძნებული login, prompt-ის გარეშე.

### ფაილები (ამ სექციასთან დაკავშირებული)

- `notebooks/model_experiment_ARIMA_SARIMA.ipynb` — ARIMA და SARIMA experiments
- `notebooks/model_experiment_Prophet.ipynb` — Prophet experiments (holiday ablation-ის ჩათვლით)
- `notebooks/model_experiment_DLinear.ipynb` — DLinear (deep learning) global model, 271 სერიაზე
- Experiment tracking: [wandb project](https://wandb.ai/adane21-free-university-of-tbilisi-/walmart-forecasting-statistical)
