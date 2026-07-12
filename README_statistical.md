# Statistical Time-Series Models — ARIMA, SARIMA, Prophet

## მიმოხილვა

ეს ნაწილი მოიცავს Walmart Store Sales Forecasting ამოცანისთვის გამოცდილ classical
statistical მოდელებს: **ARIMA**, **SARIMA** და **Prophet**. დავალების პირობის მიხედვით,
ARIMA/SARIMA-ს დროში დიდი ტრენინგის ნაცვლად აქცენტი კეთდება მათ თეორიულ გარჩევასა და
დასაბუთებულ დასკვნებზე.

**მეთოდოლოგიური გადაწყვეტილება — representative sampling:** დატასეტში 3000+ store-dept
წყვილია, ხოლო classical statistical მოდელები ცალკეული სერიისთვის მაღალი გამოთვლითი
ღირებულებისაა (განსაკუთრებით SARIMA `m=52` სეზონურ lag-ზე). ამიტომ ARIMA/SARIMA-სთვის
2 წარმომადგენლობითი store-dept წყვილი (`Store 1/Dept 1`, `Store 20/Dept 92`) გამოვცადეთ,
Prophet-ისთვის კი — რომელიც შედარებით სწრაფია — 5 წყვილი. ეს არ ისახავს მიზნად საუკეთესო
შესაძლო score-ის მიღწევას, არამედ თეორიული დასკვნების რიცხვებით დასაბუთებას.

ყველა experiment დალოგილია **Weights & Biases**-ზე:
`https://wandb.ai/adane21-free-university-of-tbilisi-/walmart-forecasting-statistical`

---

## 1. ARIMA

**თეორია:** ARIMA სამი კომპონენტისგან შედგება — AR(p) (რეგრესია საკუთარ წარსულზე),
I(d) (differencing-ის ხარისხი stationarity-სთვის), MA(q) (რეგრესია წინა ცდომილებებზე).
პარამეტრები შეირჩა `pmdarima.auto_arima`-თი, AIC-ის მინიმიზაციით.

**შედეგები:**

| Store | Dept | Order (p,d,q) | WMAE |
|---|---|---|---|
| 1 | 1 | (0, 0, 2) | 21 595.40 |
| 20 | 92 | (3, 1, 1) | 11 002.42 |

**დაკვირვება:** Store 20/Dept 92-ის ADF ტესტმა აჩვენა p-value = 0.5214 (>0.05) —
ორიგინალი სერია non-stationary-ია, რაც ხსნის `auto_arima`-ს არჩევანს `d=1`-ზე. Store
1/Dept 1-ისთვის კი `auto_arima`-მ `d=0` აირჩია (საერთოდ არ დაადიფერენცირა) — ეს საეჭვო
არჩევანია holiday-heavy სერიაზე და, სავარაუდოდ, სწორედ ამან გამოიწვია შედარებით მაღალი
(უარესი) WMAE ამ სერიაზე.

---

## 2. SARIMA

**თეორია:** SARIMA = ARIMA + სეზონური კომპონენტი `(P,D,Q,s)`. `s=52` წლიურ სეზონურობას
შეესაბამება (Christmas, Thanksgiving, Super Bowl, Labor Day ყოველწლიურად თითქმის იმავე
კვირაში მეორდება).

**მეთოდოლოგიური გადაწყვეტილება:** `auto_arima(seasonal=True, m=52)`-ის stepwise search
ძალიან ნელია (candidate მოდელების სრული fit 52-სეზონურ lag-ზე). ამიტომ ორივე სერიისთვის
ხელით დავაფიქსირეთ გონივრული `(1,1,1)(0,1,1,52)` პარამეტრები (`D=1` — სეზონური
differencing წლიური ციკლის მოსახსნელად, `Q=1` — სეზონურ ცდომილებებზე).

**შედეგები:**

| Store | Dept | ARIMA WMAE | SARIMA WMAE | ცვლილება |
|---|---|---|---|---|
| 1 | 1 | 21 595.40 | **4 772.71** | **-78%** |
| 20 | 92 | 11 002.42 | 18 788.34 | **+71%** |

**დაკვირვება — მთავარი თეორიული დასკვნა:** SARIMA-ს ეფექტი მკვეთრად საწინააღმდეგო
მიმართულებისაა ორ სერიაზე. Store 1/Dept 1-ზე სეზონურმა კომპონენტმა მნიშვნელოვნად
გააუმჯობესა შედეგი, ხოლო Store 20/Dept 92-ზე — გააუარესა. ეს იმიტომ ხდება, რომ ორივე
სერიისთვის **იგივე** ფიქსირებული `(P,D,Q,s)` გამოვიყენეთ დროის დაზოგვის მიზნით —
თითოეულ სერიას სინამდვილეში თავისი ოპტიმალური სეზონური პარამეტრები სჭირდება. ეს
პრაქტიკულადაც აჩვენებს, რატომაა 3000+ სერიაზე ცალკეული classical statistical tuning
რეალისტურად შეუძლებელი — ცალკეული auto-tuning თითოეულ სერიაზე გამოთვლითად ძალიან ძვირი
იქნებოდა.

---

## 3. Prophet

**თეორია:** Prophet-ის მიდგომა ფუნდამენტურად განსხვავებულია ARIMA/SARIMA-სგან —
decomposable model (`y = trend + seasonality + holidays`), არა ავტორეგრესიული.
Stationarity არ არის საჭირო, robust-ია missing data-ზე, და აქვს built-in `holidays`
პარამეტრი.

**Holiday dataframe:** გამოყენებულია იგივე 4 Walmart holiday, რაც `feature_engineering.py`-ის
`add_holiday_proximity`-შია განსაზღვრული (Super Bowl, Labor Day, Thanksgiving, Christmas),
`lower_window=-7`-ით (holiday-ს წინა კვირაც შედის ეფექტში, markdown-ების გამო).

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

**დაკვირვება — მოულოდნელი შედეგი:** ორივე სერიაზე Prophet holidays-ის **გარეშე** ოდნავ
სჯობდა holidays-იან ვერსიას. ეს ეწინააღმდეგება საწყის თეორიულ მოლოდინს (რომ explicit
holiday მოდელირება უნდა აუმჯობესებდეს ჰოლიდეი-მძიმე retail დატასეტზე). სავარაუდო მიზეზი:
მხოლოდ 4 holiday ტიპი × 4 წელი (=16 თარიღი) სავარაუდოდ არასაკმარისია Prophet-ის holiday
ეფექტის საიმედოდ შესაფასებლად ასეთ მცირე sample-ზე — `lower_window=-7`-ის დამატებამ
შესაძლოა noise შეიტანა სასარგებლო სიგნალის ნაცვლად. მეტ სერიაზე/მეტ მონაცემზე ტესტვისას
სავარაუდოდ ეს ეფექტი შემობრუნდებოდა დადებითობისკენ.

---

## 4. სამივე მოდელის საბოლოო შედარება

| | ARIMA | SARIMA | Prophet (no holidays) | Prophet (holidays) |
|---|---|---|---|---|
| **Store 1, Dept 1** | 21 595.40 | **4 772.71** | 3 159.08 | 3 332.51 |
| **Store 20, Dept 92** | 11 002.42 | 18 788.34 | **13 831.08** | 14 349.13 |

**საუკეთესო მოდელი სერიაზეა დამოკიდებული:**
- Store 1/Dept 1-ზე Prophet (holidays-ის გარეშე) აჩვენა ყველაზე დაბალი WMAE, თუმცა
  SARIMA-მაც ძალიან კარგი შედეგი აჩვენა (4772.71) — ორივე მნიშვნელოვნად სჯობდა pure
  ARIMA-ს.
- Store 20/Dept 92-ზე ისევ Prophet (holidays-ის გარეშე) იყო საუკეთესო, ARIMA-მაც
  საკმაოდ კარგად იმუშავა, ხოლო SARIMA — ყველაზე ცუდად (ფიქსირებული სეზონური
  პარამეტრების overfitting-ის გამო).

## 5. ზოგადი დასკვნა

არცერთი classical statistical მოდელი არ არის universally საუკეთესო ყველა სერიაზე —
შედეგი მკვეთრად სერია-სპეციფიკურია. ასევე, არცერთი ეს მოდელი არ არის პრაქტიკულად
scalable 3000+ store-dept სერიაზე იმ ფორმით, რომლითაც აქ გამოვცადეთ: თითოეულ სერიას
თავისი ოპტიმალური პარამეტრები სჭირდება, ხოლო ავტომატური ძებნა (`auto_arima` seasonal
search) გამოთვლითად ძალიან ძვირია `m=52`-ზე.

ეს არის მთავარი მოტივაცია **tree-based global მოდელისკენ** (LightGBM/XGBoost) გადასვლის
საჭიროებისთვის — ერთი მოდელი, რომელიც Store/Dept ID-ებს კატეგორიულ ფიჩერებად იყენებს და
ყველა სერიას ერთდროულად სწავლობს, ცალკეული tuning-ის საჭიროების გარეშე.

---

## ფაილები

- `notebooks/model_experiment_ARIMA_SARIMA.ipynb` — ARIMA და SARIMA experiments
- `notebooks/model_experiment_Prophet.ipynb` — Prophet experiments (holiday ablation-ის ჩათვლით)
- Experiment tracking: [wandb project](https://wandb.ai/adane21-free-university-of-tbilisi-/walmart-forecasting-statistical)
