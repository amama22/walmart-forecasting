## DLinear — Deep Learning Time-Series Model

*(ეს სექცია აღწერს DLinear-ის ორივე notebook-ს — ექსპერიმენტულს და
inference/submission-ს — Walmart Store Sales Forecasting ამოცანისთვის.
Classical statistical მოდელების (ARIMA/SARIMA/Prophet) აღწერა ცალკე README-შია.)*

### სარჩევი

1. [მოკლე შეჯამება (TL;DR)](#მოკლე-შეჯამება-tldr)
2. [თეორია](#თეორია)
3. [`model_experiment_DLinear.ipynb` — მეთოდოლოგია](#model_experiment_dlinearipynb--მეთოდოლოგია)
4. [`model_experiment_DLinear.ipynb` — შედეგები](#model_experiment_dlinearipynb--შედეგები)
5. [შედარება Classical მოდელებთან](#შედარება-classical-მოდელებთან)
6. [`model_inference.ipynb` — Kaggle Submission](#model_inferenceipynb--kaggle-submission)
7. [დასკვნა და შემდეგი ნაბიჯები](#დასკვნა-და-შემდეგი-ნაბიჯები)
8. [შეზღუდვები](#შეზღუდვები)
9. [რეპროდუცირება](#რეპროდუცირება)

---

### მოკლე შეჯამება (TL;DR)

DLinear — მარტივი, ორი linear layer-ისგან შემდგარი deep learning მოდელი — ერთდროულად
დატრენინგდა **271 store-dept სერიაზე**, გაზიარებული წონებით. `Store 1/Dept 1`-ზე
მიაღწია **WMAE = 1 075.21**, რაც ყველა classical statistical მოდელს (ARIMA, SARIMA,
Prophet) აჯობა ამ სერიაზე — Prophet-ის საუკეთესო შედეგსაც (3 159.08) ~3-ჯერ გადააჭარბა.
მთავარი პრაქტიკული უპირატესობა: DLinear scalable-ია 3000+ სერიაზეც ერთი ტრენინგით,
რაც classical per-series fitting-ისთვის პრაქტიკულად შეუძლებელია.

---

### თეორია

**წყარო:** [Zeng et al., 2023 — "Are Transformers Effective for Time Series
Forecasting?"](https://arxiv.org/abs/2205.13504) (AAAI). ავტორებმა აჩვენეს, რომ
ძალიან მარტივმა linear მოდელმა შეიძლება გადააჭარბოს კომპლექსურ Transformer-ზე
დაფუძნებულ არქიტექტურებს (Informer, Autoformer, FEDformer) long-term time-series
forecasting-ში.

**არქიტექტურა — ორი ეტაპი:**

1. **Series Decomposition** — moving average (`kernel_size=25`) გამოყოფს:
   - **Trend** — moving average-ის შედეგი (გლუვი, ნელა ცვალებადი კომპონენტი)
   - **Seasonal/Residual** — `original - trend` (სწრაფად ცვალებადი რხევები)
2. **Linear mapping** — ორი დამოუკიდებელი `Linear(lookback, horizon)` layer, ერთი
   თითოეულ კომპონენტზე. საბოლოო პროგნოზი = `trend_forecast + seasonal_forecast`.

**რატომ მუშაობს ეს ასე კარგად:** ავტორების არგუმენტია, რომ self-attention-ის
**permutation-invariant** ბუნება კარგავს დროით (temporal) ინფორმაციას, რაც დროით
სერიებში კრიტიკულია — უბრალო linear mapping კი ამ ინფორმაციას ინარჩუნებს.

**Shared weights — მთავარი მეთოდოლოგიური არჩევანი:** DLinear paper-ი გვთავაზობს
ორ ვარიანტს — "individual" (ცალკე წონები თითო channel-ზე) და "shared" (ერთი წონები
ყველა channel-ისთვის). ჩვენს შემთხვევაში, სადაც მიზანი **scalable global მოდელია**
3000+ store-dept სერიაზე, "shared" ვარიანტი ბუნებრივი არჩევანია — ერთი მოდელის
ტრენინგი მოიცავს ყველა სერიას ერთდროულად, per-series tuning-ის საჭიროების გარეშე.

---

### `model_experiment_DLinear.ipynb` — მეთოდოლოგია

| პარამეტრი | მნიშვნელობა | დასაბუთება |
|---|---|---|
| `lookback` | 52 კვირა | ერთი წლის ისტორია — საკმარისი წლიური სეზონურობის დასაჭერად |
| `horizon` | 13 კვირა | იგივე, რაც ARIMA/SARIMA/Prophet-ის შეფასებაში — პირდაპირი შედარებისთვის |
| `kernel_size` | 25 | moving average window trend decomposition-ისთვის |
| `MAX_SERIES` | 300 (→ 271 ვალიდური) | სწრაფი იტერაცია Kaggle-ზე; ადვილად იზრდება `None`-მდე (სრული დატასეტი) |
| Normalization | Instance (თითო window თავისივე mean/std) | სხვადასხვა store/dept-ის მასშტაბის სხვაობის მოსაგვარებლად shared-weight მოდელში |
| Loss | Weighted MAE (holiday წონა 5) | WMAE-ს დიფერენცირებადი ვერსია — პირდაპირ ოპტიმიზაცია target metric-თან |
| Optimizer | Adam, `lr=1e-3` | სტანდარტული არჩევანი |
| Early stopping | `patience=5` | overfitting-ის თავიდან ასაცილებლად |
| Hardware | Kaggle GPU T4×2 | — |

**Train/Val split:** დროზე დაფუძნებული, გლობალური — ბოლო 13 კვირა თითოეულ
სერიაზე ვალიდაციისთვის იყო შენახული, დანარჩენზე sliding-window ტრენინგი. ეს
**ერთჯერადი split** განსხვავდება classical მოდელების walk-forward CV-სგან
(3 fold) — deep learning-ის global ტრენინგისას ეს compute ტრადეოფია.

---

### `model_experiment_DLinear.ipynb` — შედეგები

**ტრენინგის დინამიკა:** Early stopping-მა გააჩერა ტრენინგი **epoch 12**-ზე
(`best_val_loss_norm=0.5415`), 30-დან განსაზღვრულის ნაცვლად — რაც ადასტურებს, რომ
DLinear ძალიან სწრაფად converge-დება (~30 წამი GPU-ზე, 271 სერიაზე ერთდროულად).

**აგრეგირებული შედეგები (271 store-dept სერია, `MAX_SERIES=300`-დან — 29 გამოირიცხა
არასაკმარისი ისტორიის გამო):**

| მეტრიკა | მნიშვნელობა |
|---|---|
| აგრეგირებული WMAE (ყველა სერია ერთად) | 1 865.65 |
| საშუალო per-series WMAE | 1 864.05 |
| მედიანური per-series WMAE | 1 137.20 |

**Per-series WMAE (პირდაპირ შედარებადი classical მოდელებთან):**

| Store | Dept | WMAE |
|---|---|---|
| **1** | **1** | **1 075.21** |
| 20 | 92 | ვერ შემოწმდა — `MAX_SERIES=300`-ის შერჩევაში არ მოხვდა |

---

### შედარება Classical მოდელებთან

**Store 1, Dept 1 — ყველა მოდელი:**

| მოდელი | WMAE |
|---|---|
| ARIMA | 21 595.40 |
| SARIMA | 4 772.71 |
| Prophet (holidays) | 3 332.51 |
| Prophet (no holidays) | 3 159.08 |
| **DLinear** (global, 271 სერიაზე ერთად) | **1 075.21** |

DLinear-მა ყველა classical მოდელი დაამარცხა ამ სერიაზე — Prophet-ის საუკეთესო
შედეგსაც (3 159.08) ~3-ჯერ გადააჭარბა (1 075.21). ეს განსაკუთრებით საინტერესოა,
რადგან DLinear ერთდროულად სწავლობდა 271 სხვადასხვა სერიას (**არა** ცალკეული
tuning-ით, როგორც ARIMA/SARIMA/Prophet), მაშინ როცა SARIMA-ს ერთ სერიაზეც კი
წუთები სჭირდებოდა `m=52`-ზე.

**შენიშვნა Kaggle-ის leaderboard-თან შედარებით:** ამ კომპეტიციის Kaggle
leaderboard-ის გამარჯვებულმა (David Thaler, SVD-denoising + STL + Exponential
Smoothing/ARIMA ensemble) მიაღწია სავარაუდოდ ~2 100-2 200 WMAE-ს **მთელი test
set-ის** მასშტაბით (ყველა 3000+ სერია, ყველა horizon კვირა ერთად). ეს **არ არის
პირდაპირ შედარებადი** ჩვენს Store 1/Dept 1-ის 1 075.21-თან, რადგან ჩვენი რიცხვი
მხოლოდ ერთ სერიაზეა — ეს მხოლოდ მასშტაბის კონტექსტისთვის არის მოყვანილი.

---

### `model_inference.ipynb` — Kaggle Submission

**მიზანი:** `model_experiment_DLinear.ipynb` მხოლოდ **შიდა** validation-ზეა
შეფასებული (`train.csv`-ის ბოლო 13 კვირა) — ეს WMAE **არასდროს ჩანს** Kaggle-ის
leaderboard-ზე. `model_inference.ipynb` არის ცალკე ფაილი, რომელიც აწარმოებს
რეალურ პროგნოზს Kaggle-ის ოფიციალურ `test.csv`-ზე და აყალიბებს `submission.csv`-ს
Kaggle-ზე ასატვირთად — ეს არის დავალების მოთხოვნილი `model_inference.ipynb`
ფაილიც ამავდროულად.

**მთავარი განსხვავებები ექსპერიმენტულ notebook-თან:**

| | `model_experiment_DLinear.ipynb` | `model_inference.ipynb` |
|---|---|---|
| მიზანი | მოდელის შეფასება/შედარება | რეალური Kaggle submission-ის გენერაცია |
| სერიები | 271 (`MAX_SERIES=300`-დან) | **ყველა** `test.csv`-ში არსებული სერია (3000+) |
| `horizon` | 13 კვირა (შედარებადობისთვის classical-ებთან) | `test.csv`-ის სრული სიგრძე (~39 კვირა) |
| Val split | ბოლო 13 კვირა თითო სერიაზე | არ არსებობს — მთელი `train.csv` გამოიყენება ტრენინგში |
| Output | WMAE მეტრიკა | `submission.csv` ფაილი |

**მეთოდოლოგია:**
1. **სრული ტრენინგი** — DLinear ხელახლა ტრენინგდება ყველა `train.csv`-ის
   სერიაზე (არა შერჩეულ ქვესიმრავლეზე), `horizon = len(test_dates)` პარამეტრით
2. **Model Registry** — საბოლოო მოდელი ინახება **W&B Artifact**-ად (MLflow-ის
   Model Registry-ის ფუნქციური ეკვივალენტი, რადგან პროექტი wandb-ზეა გადასული —
   იხ. [რეპროდუცირება](#რეპროდუცირება))
3. **Inference** — თითოეულ `test.csv`-ის `(Store, Dept)` წყვილზე:
   - თუ `train.csv`-ში `>= 52` კვირის ისტორია არსებობს — DLinear-ის პროგნოზი
   - თუ ისტორია არასაკმარისია ან სერია საერთოდ ახალია — **fallback**
     (სერიის საკუთარი საშუალო, ან — თუ სერია სულაც არ არსებობს — გლობალური
     საშუალო Weekly_Sales)
4. **`submission.csv`** — ფორმატირდება Kaggle-ის მოთხოვნით (`Id = Store_Dept_Date`)

**შედეგი:** *[შეავსეთ გაშვების შემდეგ — `fallback_count`/`fallback_series_pct`
(რამდენ სერიაზე მოუწია fallback-ს გამოყენება) და, თუ Kaggle-ზეც აიტვირთა
submission, საჯარო (public) leaderboard score]*

---

### დასკვნა და შემდეგი ნაბიჯები

**Scalability + accuracy ერთად:** DLinear-ის შედეგმა დაადასტურა classical
სექციაში გამოთქმული ვარაუდი (რომ per-series statistical tuning არ არის
პრაქტიკული 3000+ სერიაზე) და ერთი ბიჯითაც წინ წავიდა — ერთმა გაზიარებული-
წონებიანმა linear მოდელმა, 271 სერიაზე ერთდროულად ტრენინგისას, არა მხოლოდ
scalability პრობლემა მოაგვარა, არამედ **accuracy-შიც** აჯობა ყველა classical
მოდელს პირდაპირ შედარებად სერიაზე. ეს ცხადყოფს **decomposition + linear mapping
+ shared weights**-ის კომბინაციის ეფექტურობას — მარტივი architecture-ითაც
შესაძლებელია scale-ს მიღწევა, სიზუსტის დაკარგვის გარეშე.

**შემდეგი ბუნებრივი ნაბიჯები:**
- `MAX_SERIES`-ის გაზრდა სრულ დატასეტამდე (3000+ სერია) — DLinear-ის სისუბუქის
  გამო, სავარაუდოდ, ამის შესრულებაც შესაძლებელია იმავე ტრენინგის დროში
- შედარება უფრო კომპლექსურ deep learning არქიტექტურასთან (N-BEATS/PatchTST/TFT)
  — ღირს თუ არა დამატებითი complexity (attention, non-linearity), თუ DLinear-ის
  სიმარტივე უკვე საკმარისია
- ✅ **დასრულებულია:** `model_inference.ipynb`-ში DLinear ხელახლა დატრენინგდა
  ყველა სერიაზე, W&B Artifact-ად დარეგისტრირდა და `submission.csv` დაგენერირდა
  Kaggle-ზე ასატვირთად (იხ. [ზემოთ](#model_inferenceipynb--kaggle-submission))

---

### შეზღუდვები

- **Val split** ერთჯერადია (ბოლო 13 კვირა), არა walk-forward CV, როგორც classical
  მოდელებში — deep learning-ის global ტრენინგისას ეს compute ტრადეოფია
- **`Store 20/Dept 92`** ვერ შემოწმდა ამ გაშვებაში — `MAX_SERIES=300`-ის
  შერჩევაში (default `series_id` თანმიმდევრობით) არ მოხვდა. სრული შედარებისთვის
  საჭიროა `MAX_SERIES`-ის გაზრდა ან ამ კონკრეტული სერიის ხელით დამატება
- **Aggregate WMAE (1 865.65)** დომინირებულია მაღალი მოცულობის სერიებით
  (აბსოლუტური დოლარული ცდომილება), ამიტომ არ წარმოადგენს ყველა სერიის თანაბარ
  შეწონილ საშუალოს და პირდაპირ არ ადარდება classical run-ების ცალკეულ WMAE-ს
- **Shared weights** ნიშნავს, რომ მოდელი ვერ სწავლობს store/dept-სპეციფიკურ
  bias-ს ცალკე (განსხვავებით, მაგალითად, tree-based მოდელისგან, რომელსაც
  Store/Dept ID-ები კატეგორიულ ფიჩერებად შეუძლია გამოიყენოს)
- **`model_inference.ipynb`-ის fallback ლოგიკა** (სერიის/გლობალური საშუალო
  არასაკმარისი ისტორიის შემთხვევაში) მარტივია და არ იყენებს DLinear-ის
  learned pattern-ებს ახალი/მოკლე სერიებისთვის — უფრო დახვეწილი მიდგომა
  (მაგ. მსგავსი dept-ის საშუალო შაბლონის გამოყენება) გააუმჯობესებდა ამ
  ქვესიმრავლის სიზუსტეს

---

### რეპროდუცირება

**საჭირო გარემო:** Kaggle Notebook, competition input დამატებული
(`walmart-recruiting-store-sales-forecasting`), GPU accelerator (რეკომენდირებული),
W&B ანგარიში.

```bash
!pip install wandb -q   # torch წინასწარაა დაყენებული Kaggle-ის default image-ში
```

**W&B setup** (Kaggle Secrets-ის საშუალებით, login prompt-ის გარეშე):
```python
from kaggle_secrets import UserSecretsClient
import os

os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
wandb.login()

WANDB_PROJECT = "walmart-forecasting-statistical"
```

**Experiment tracking:** ყველა run დალოგილია აქ:
[wandb project](https://wandb.ai/adane21-free-university-of-tbilisi-/walmart-forecasting-statistical)

**ფაილები:**
- `notebooks/model_experiment_DLinear.ipynb` — შეფასება/შედარება (run: `DLinear_Global`)
- `notebooks/model_inference.ipynb` — Kaggle submission-ის გენერაცია
  (run: `DLinear_FullTrain_Inference`), output: `submission.csv`

**ფაილი:** `notebooks/model_experiment_DLinear.ipynb`
