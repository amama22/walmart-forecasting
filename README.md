# Walmart Recruiting - Store Sales Forecasting

Kaggle-ის კონკურსში **Walmart Recruiting - Store Sales Forecasting** ამოცანაა 45 Walmart-ის მაღაზიის (თითოეულში რამდენიმე დეპარტამენტის) კვირეული გაყიდვების პროგნოზირება. ეს არის კლასიკური Time-Series ამოცანა, სადაც შევადარეთ სხვადასხვა არქიტექტურის მოდელები (Tree-Based, Deep Learning, კლასიკური სტატისტიკური მოდელები).

## რეპოზიტორიის სტრუქტურა

```
walmart-forecasting/
├── README.md
├── notebooks/
│   └── eda_feature_engineering.ipynb
├── model_experiment_LightGBM.ipynb
├── model_experiment_XGBoost.ipynb
├── model_experiment_N-BEATS.ipynb
├── model_experiment_PatchTST.ipynb
├── model_inference.ipynb
└── src/
    ├── data_prep.py
    ├── feature_engineering.py
    ├── evaluation.py
    └── utils.py
```

---

## მონაცემები და EDA

*(იხ. `notebooks/eda_feature_engineering.ipynb`)*

მონაცემები შედგება 4 ფაილისგან: `train.csv`, `test.csv`, `features.csv`, `stores.csv`. `train`-ს და `features`-ს (და `stores`-ს) გავაერთიანეთ `Store`/`Date`/`Store` გასწვრივ (`merge_all` ფუნქცია, `src/data_prep.py`).

## შეფასების მეტრიკა

- **WMAE** (`src/evaluation.py`) - კონკურსის ოფიციალური მეტრიკა: holiday კვირები 5x წონით, დანარჩენი 1x წონით.



## Tree-Based Models

ეს ჯგუფი მოიცავს LightGBM-სა და XGBoost-ს - ორივე იყენებს ერთსა და იმავე engineered feature ნაკრებს და შეფასების სქემას:

- **Walk-Forward Split** (`walk_forward_splits`) - 3 fold, თითოეული 38-კვირიანი validation window (იმეორებს test set-ის რეალურ სიგრძეს), ისე რომ ყოველი fold-ის train ნაწილი მხოლოდ წარსულ მონაცემებს შეიცავს - არანაირი leakage მომავლიდან წარსულში.

### Cleaning

ლოგირებულია MLflow-ზე, ცალკე run თითოეული მოდელის experiment-ში (`{Model}_Cleaning`).

- **MarkDown1-5**: null → 0 (ეს ველები რეალურად ნიშნავს "ამ კვირას ფასდაკლება არ ყოფილა", არა missing data).
- **CPI / Unemployment**: დაგეგმილი იყო per-store forward-fill (`clean_data`, `src/data_prep.py`), თუმცა ფაქტობრივად ორივე მოდელის cleaning run-ზე null count 0-ზე დაბრუნდა ჯერ კიდევ merge-ის შემდეგ - ანუ `features.csv` მთლიანად ფარავს საჭირო თარიღებს ორივესთვის (train-ისთვისაც). ffill ლოგიკა ჩავტოვეთ დაცვის მიზნით, თუმცა ამ dataset-ზე რეალურად საქმეს არ აკეთებდა.


### Feature Engineering

აქ ყველაზე მნიშვნელოვანი დიზაინის გადაწყვეტილება ეხება იმას, თუ **რომელი feature-ები რჩება უსაფრთხო ერთჯერადი (non-recursive) inference-ისთვის**.

#### რატომ არ გამოვიყენეთ short lags და rolling features

`src/feature_engineering.py`-ში არსებული `add_lag_features` (lag 1/2/4/52) და `add_rolling_features` (rolling mean/std, window 4/8) დამოკიდებულია **ბოლო კვირების რეალურ გაყიდვებზე**. `train`-სა და `test`-ს შორის სულ 7 დღიანი ხარვეზია, მაგრამ `test` თავად 39 კვირას მოიცავს - ანუ test-ის 2-ე, მე-3, ... კვირისთვის lag_1/lag_2/lag_4-ს სჭირდება წინა კვირების გაყიდვები, რომლებიც ჯერ **არ არის ცნობილი** (სწორედ ესაა ჩვენი prediction target). ამის გამოსაყენებლად საჭირო იქნებოდა **რეკურსიული პროგნოზირება** (predict week 1 → გამოვიყენოთ როგორც "ცნობილი" წინა კვირა → predict week 2 → ...), რაც error-ის დაგროვებას (compounding error) იწვევს - მოდელის მცირე bias-იც კი პირველივე კვირაზე თანდათან ძლიერდება.

გადავწყვიტეთ ეს რისკი არ აგვეღო და ორივე tree-based მოდელისთვის:
- **დავტოვეთ მხოლოდ `lag_52`-ის მსგავსი feature**, რადგან 52 კვირით უკან ყოველთვის test-ის ნებისმიერი თარიღისთვისაც კი `train`-შივე ხვდება.
- **წავშალეთ ყველა rolling mean/std feature** მთლიანად.

#### 53-კვირიანი (NRF) კალენდრის პრობლემა და მისი გამოსწორება

უბრალო shift(52)-ზე (row-based lag, ისე როგორც `add_lag_features`-შია დაწერილი) უკეთესი მიდგომაა კვირის ნომერზე (`WeekOfYear`, ISO) მორგებული lag, რადგან სინამდვილეში Walmart იყენებს NRF 4-5-4 საფინანსო კალენდარს და 2010 წელი 53-კვირიანი იყო. ეს აცდენა განსაკუთრებით საზიანოა holiday კვირებში (სადაც WMAE-ის წონა 5x-ია).

ვუერთებთ წინა წლის იმავე `WeekOfYear`-ის გაყიდვას (`Store`+`Dept`+`Year-1`+`WeekOfYear` join), row-count-ზე დაფუძნებული shift-ის ნაცვლად. ეს გვცავს calendar drift-ისგან.

#### Named Holiday Flags

ცალკეული boolean flag თითოეული 4 holiday-სთვის (`IsSuperBowl`, `IsLaborDay`, `IsThanksgiving`, `IsChristmas`), ერთი საერთო `IsHoliday`-ის ნაცვლად - რადგან ეს 4 holiday განსხვავებულად მოქმედებს გაყიდვებზე (მაგ. Christmas კვირას მაღაზიები იკეტება და რეალური spike ხშირად წინა კვირაზე გადადის - ცნობილი quirk ამ dataset-ში).

#### Expanding Averages - "frozen snapshot" Test-ისთვის

`add_expanding_dept_avg`/`add_expanding_store_avg` (`src/feature_engineering.py`) ითვლის row-ისთვის ყველა წინა (`shift(1)`-ით საკუთარი კვირის გამორიცხვით) კვირის საშუალო გაყიდვას - training-ის დროს ეს ნამდვილად "expanding" (მზარდი) feature-ია და მოდელს ეხმარება ისწავლოს store/dept-ის დონის ტრენდი.

Test-ზე inference-ისას კი expanding average-ის გაზრდა შეუძლებელია ახალი რეალური მონაცემის გარეშე - რაც არ გვაქვს. ამიტომ საბოლოო pipeline-ში ეს feature **იყინება** train-ის ბოლო მნიშვნელობაზე (frozen snapshot), იმავე non-recursive ლოგიკით, რაც lag-ებთან გამოვიყენეთ თანმიმდევრულობისთვის. კონკრეტულად, snapshot ითვლის `shift(1)`-ის სემანტიკის ზუსტ ანალოგს - ანუ თითოეული ჯგუფის უკანასკნელი კვირის გამოკლებით საშუალოს.

#### Feature Selection

ორივე მოდელისთვის `feature_importances_`-ის მიხედვით გამოვავლინეთ დაბალი წვლილის feature-ები - `IsSuperBowl`, `IsLaborDay`, `IsChristmas` (მათი signal `days_to_nearest_holiday`-ში უკვე ჩართულია), და `MarkDown1/2/4/5`, `Year`. 24 → 16 feature-მდე შემცირებამ LightGBM-ზე გააუმჯობესა WMAE (2940.71 → 2914.29).

**საბოლოო feature ნაკრები (16):** `Store, Dept, Type, Size, Temperature, Fuel_Price, CPI, Unemployment, MarkDown3, Month, WeekOfYear, sales_lag_52wk, IsThanksgiving, days_to_nearest_holiday, dept_avg_sales_expanding, store_avg_sales_expanding`.

#### კატეგორიული ცვლადების დამუშავება

`Store` (45 კატეგორია) და `Dept` (99 კატეგორია) დავტოვეთ native categorical dtype-ის სახით, One-Hot Encoding-ის ნაცვლად - ორივე ტესტირებული მოდელისთვის (LightGBM-ის `categorical_feature`, XGBoost-ის `enable_categorical=True`). ეს არჩევანი წინასწარ ცნობილი, დადასტურებული საუკეთესო პრაქტიკაა high-cardinality კატეგორიებისთვის tree-based მოდელებში (One-Hot-ი გაზრდიდა განზომილებას უსარგებლოდ და გააუარესებდა split ხარისხს) - ამიტომ ცალკე შედარების run-ი აღარ გავუშვით.


### LightGBM

*(იხ. `model_experiment_LightGBM.ipynb`, MLflow experiment: `LightGBM_Training`)*

| ეტაპი | WMAE (mean, 3-fold CV) |
|---|---|
| Baseline, ყველა feature (24) | 2940.71 |
| Pruned features (16) | 2914.29 |
| HPO-ს შემდეგ (Optuna, 30 trial) | **2477.74** |
| Pipeline check, ცალკე fold (fold 2) | 1665.13 |

**საუკეთესო ჰიპერპარამეტრები:** `num_leaves=187, learning_rate=0.036, n_estimators=659, min_child_samples=33, feature_fraction=0.66, bagging_fraction=0.92, bagging_freq=6`.

**Kaggle შედეგი:** Public **2769**, Private **2857**.

CV-სა (2477.74) და Kaggle-ს (2769/2857) შორის სხვაობა მოსალოდნელია (1) frozen snapshot-ის "სიძველე" ნებისმიერ CV fold-ზე მეტია რეალურ test-ზე (test უფრო შორსაა train-ის ბოლოდან, ვიდრე ნებისმიერი CV fold იყო), (2) Christmas-ის timing quirk რეალურადაც ვლინდება test-ზეც. Public-სა და Private-ს შორის სხვაობა მცირეა (88 ქულა) - ანუ overfitting public leaderboard-ზე არ შეინიშნება.

მოდელი დარეგისტრირებულია Model Registry-ში, სახელით **`walmart-lightgbm-store-sales`** (v1), `cloudpickle` სერიალიზაციით (skops ვერ უმკლავდებოდა custom `WalmartFeatureBuilder` transformer-ს).


### XGBoost

*(იხ. `model_experiment_XGBoost.ipynb`, MLflow experiment: `XGBoost_Training`)*

იგივე feature ნაკრები და CV სქემა, გადაწერილი XGBoost-ისთვის (`enable_categorical=True`, `tree_method="hist"`).

| ეტაპი | WMAE (mean, 3-fold CV) |
|---|---|
| Baseline, pruned features (16) | 2722.33 |
| HPO-ს შემდეგ (Optuna, 30 trial) | **2432.06** |
| Pipeline check, ცალკე fold (fold 2) | 1615.19 |

**საუკეთესო ჰიპერპარამეტრები:** `max_depth=11, learning_rate=0.029, n_estimators=385, min_child_weight=7, subsample=0.84, colsample_bytree=0.84, reg_alpha=0.0035, reg_lambda=0.0043`.

**Kaggle შედეგი:** Public **2712**, Private **2806**.

Feature importance-ის მიხედვით XGBoost-მა იგივე თანმიმდევრობა დაადასტურა, რაც LightGBM-მა (`Dept`, `Store` დომინირებს, შემდეგ expanding averages და `sales_lag_52wk`, `Type` ~0 წვლილით) - ეს ორი დამოუკიდებელი არქიტექტურის თანხმობა კარგი დადასტურებაა, რომ pruned feature set რეალურ სიგნალს იჭერს, და არა ცალკეული მოდელის split-ის თავისებურებას.

მოდელი დარეგისტრირებულია Model Registry-ში, სახელით **`walmart-xgboost-store-sales`** (v1), ასევე `cloudpickle` სერიალიზაციით.


### ხეზე დაფუძნებული მოდელების შედარება (LightGBM vs XGBoost)

| მოდელი | Baseline (pruned) WMAE | Tuned CV WMAE | Fold-2 Pipeline Check | Kaggle Public | Kaggle Private |
|---|---|---|---|---|---|
| LightGBM | 2914.29 | 2477.74 | 1665.13 | 2769 | 2857 |
| XGBoost | 2722.33 | **2432.06** | **1615.19** | **2712** | **2806** |

**XGBoost უსწრებს LightGBM-ს ყველა ეტაპზე** - baseline-ზეც (untuned), tuning-ის შემდეგაც, ცალკეულ fold-ზეც და საბოლოო Kaggle score-ზეც (~55-60 ქულით ორივე public/private-ზე). ეს არის თანმიმდევრული, არა შემთხვევითი შედეგი. შესაძლო ახსნა: HPO-მ XGBoost-ისთვის უფრო ძლიერი რეგულარიზაცია (`reg_alpha`/`reg_lambda`) და შედარებით ღრმა, მაგრამ (histogram-based split-ის გამო) ნაკლებად "ხარბი" ხე-ს სტრუქტურა აირჩია, რაც ამ სეზონურ, holiday-ების გამო noisy dataset-ზე უკეთ განზოგადდება, ვიდრე LightGBM-ის leaf-wise growth.


## Deep Learning models

ეს ჯგუფი მოიცავს N-BEATS-ს, PatchTST-სა და DLinear-ს - სამივე გლობალური, long-format მონაცემებზე დატრენინგებული მოდელია, იმავე 1-fold ვალიდაციის შეზღუდვით (იხ. თითოეული ქვესექცია დეტალებისთვის).

### N-BEATS

*(იხ. `model_experiment_N-BEATS.ipynb`, MLflow experiment: `NBEATS_Training`)*

N-BEATS პრინციპულად განსხვავდება tree-based მოდელებისგან - ეს არის **გლობალური, უწყვეტი (pure deep learning) univariate** არქიტექტურა, რომელიც არ იღებს ცალკეულ engineered feature-ებს (`PRUNED_FEATURE_COLS`-ს აქ აზრი არ აქვს), არამედ სწავლობს პირდაპირ თითოეული Store-Dept სერიის საკუთარი ისტორიიდან, ყველა სერიაზე გაზიარებული (shared) წონებით.

#### მონაცემების გარდაქმნა (Long Format)

Tree-based მოდელების wide feature-table-ის ნაცვლად, საჭირო გახდა **long format**: თითო row თითო (სერია, თარიღი) წყვილზე - `unique_id = Store_Dept`, `ds/Date`, `y = Weekly_Sales`. სულ **3,331 უნიკალური სერია**.

#### Data Validation - რატომ არის ეს განსხვავებული tree-model-ებისგან

აღმოვაჩინეთ, რომ **605 სერიას (18%) აქვს შიდა ხარვეზი** (მაგ. დეპარტამენტი დროებით არ ფიქსირდება მაღაზიაში) და **340 სერია (10%) 1 წელზე ნაკლებ ისტორიას მოიცავს** (მინიმუმი - სულ 1 კვირა). ეს Tree-Based მოდელებისთვის პრობლემა არ იყო (row-level feature-ები დამოუკიდებელია), მაგრამ Sequence მოდელისთვის (რომელსაც სჭირდება რეგულარული, უწყვეტი grid) აუცილებელი გახდა **გლობალური reindex** მთელი train-ის თარიღების დიაპაზონზე (2010-02-05 – 2012-10-26), **ნულით შევსებული** ხარვეზებით - იმავე ლოგიკით, რაც ადრე გადავწყვიტეთ: row-ის არარსებობა ნიშნავს "ამ დეპარტამენტს გაყიდვა არ ჰქონია", და არა დაკარგულ მონაცემს.

#### Direct Multi-Horizon Forecasting

N-BEATS ბუნებრივად აკეთებს **direct multi-horizon** პროგნოზს - ერთი გამოძახებით გამოსცემს მთელ 38-39 კვირიან horizon-ს, კვირა-კვირა რეკურსიის გარეშე. შესაბამისად, მოდელს არ სჭირდება ის non-recursive feature-freezing ხრიკები (lag-ების/expanding average-ების გაყინვა), რაც LightGBM/XGBoost-ის pipeline-ისთვის საჭირო იყო - ეს ამ არქიტექტურის ბუნებრივი უპირატესობაა.

#### გამოწვევა: მხოლოდ 1 CV Fold

Tree-model-ების მსგავსად 3 rolling fold-ის დაყენება ვცადეთ, თუმცა აღმოჩნდა შეუძლებელი: `input_chunk_length (lookback) + output_chunk_length (38)` არის საჭირო მინიმალური სიგრძე ერთი training sample-ის ასაგებად. მთლიანი სერიის სიგრძე მხოლოდ 143 კვირაა - 3 fold-ის დაყოფის შემთხვევაში ყველაზე ადრეულ fold-ს მხოლოდ 28 კვირა დარჩებოდა history-ად, რაც არასაკმარისია ნებისმიერი გონივრული lookback window-სთვის. ამიტომ საბოლოოდ **მხოლოდ 1 fold**-ზე (ბოლო 38-39 კვირა) ავაწყვეთ evaluation.

#### შედეგები

| ეტაპი | WMAE (1 fold, ბოლო 38-39 კვირა) |
|---|---|
| Baseline (`input=104`, epochs=30) | 1793.25 |
| HPO-ს შემდეგ (Optuna, 26/30 trial დასრულდა - Colab session ჩაიშალა trial 27-ზე) | **1767.99** |
| საბოლოო refit, 100 epoch-ით | 1878.02 (**გაუარესდა** - overfitting მცირე dataset-ზე) |
| საბოლოო model, 15 epoch-ით (HPO-ს კონფიგურაციის იდენტური) | **1767.99** (დადასტურდა) |

**საუკეთესო ჰიპერპარამეტრები:** `input_chunk_length=46, num_stacks=2, num_blocks=2, layer_widths=128, lr=0.00093`.

**მნიშვნელოვანი შენიშვნა შედარებისას:** ეს 1767.99 **არ არის** პირდაპირ შედარებადი LightGBM/XGBoost-ის 3-fold საშუალო მაჩვენებელთან (2477.74/2432.06) - ის მხოლოდ ერთი, ყველაზე "ადვილი" fold-ია. სამართლიანი შედარებისთვის საჭიროა LightGBM/XGBoost-ის იმავე fold-ის ცალკე შედეგი (fold 2 pipeline check: 1665.13 / 1615.19) - რომელთანაც შედარებით **N-BEATS ოდნავ ჩამორჩება** ორივე tree-model-ს (დაახლოებით 100-150 ქულით).

100 epoch-ზე refit-მა ცხადყო ცალსახა **overfitting** - მცირე (~2.75 წლიანი) dataset-ზე მეტი training-ი ეხმარება არა generalization-ს, არამედ noise-ის დამახსოვრებას. საბოლოო მოდელისთვის დავუბრუნდით HPO-ს დროს ნაპოვნ 15 epoch-იან კონფიგურაციას.

**Kaggle შედეგი:** Public **3707**, Private **3885** - მნიშვნელოვნად უარესი, ვიდრე CV-ის 1767.99 გვთავაზობდა. ამის მთავარი მიზეზი უკვე ზემოთ ავხსენით: ჩვენი ერთადერთი validation fold (2012-02-03 – 2012-10-26) **საერთოდ არ მოიცავდა Thanksgiving-ს, Christmas-ს და Super Bowl-ს** - WMAE-ის ყველაზე მაღალი წონის (5x) მქონე კვირებს - ხოლო რეალური Kaggle test პერიოდი (2012-11-02 – 2013-07-26) ყველა ამ holiday-ს შეიცავს. დამატებით, ვანილურ N-BEATS-ს **საერთოდ არ გააჩნია holiday-სპეციფიკური feature** (არც `IsThanksgiving`, არც `days_to_nearest_holiday`) - მისი ერთადერთი "ცოდნა" სეზონურობაზე მოდის ნაწილობრივ, ცალკეული სერიის lookback window-დან. ეს ორივე ფაქტორი ერთად ხსნის, რატომ იყო ნამდვილი test-ის ქულა CV-ს მაჩვენებელზე მკვეთრად უარესი.

ასევე გვქონდა 11 Store-Dept წყვილი test-ში, რომელიც train-ში საერთოდ არ გვხვდება (ახალი დეპარტამენტი) - ამათთვის fallback მნიშვნელობად 0 გამოვიყენეთ, რადგან მოდელს არანაირი საფუძველი არ გააჩნია ამ სერიების პროგნოზირებისთვის.

მოდელი დარეგისტრირებულია Model Registry-ში `mlflow.pyfunc`-ის საშუალებით (custom wrapper, `walmart-nbeats-store-sales`), რადგან `darts`-ის მოდელები არ ერგება `mlflow.sklearn.log_model`-ს.


### PatchTST

*(იხ. `model_experiment_PatchTST.ipynb`, MLflow experiment: `PatchTST_Training`)*

TFT-სა და PatchTST-ს შორის PatchTST ავირჩიეთ **სისწრაფის გამო**: PatchTST channel-independent არქიტექტურაა (თითოეულ სერიას დამოუკიდებლად ამუშავებს, static/known-future/observed-past ცალკეული encoder-ების გარეშე) - მნიშვნელოვნად ნაკლები მოძრავი ნაწილი, ვიდრე TFT-ს (რომელსაც აქვს LSTM encoder/decoder + attention + gating ერთდროულად). ამის სანაცვლოდ, PatchTST ვერ იყენებს Store/CPI/MarkDown-ების msგავს კოვარიატებს რიჩ ფორმით - ისიც, N-BEATS-ის მსგავსად, pure univariate მოდელია.

#### Data Validation და 1-Fold შეზღუდვა

იმავე გლობალური reindex ლოგიკის (605 გახარვეზებული / 340 მოკლე სერია, ნულით შევსება) ხელახლა გაშვება საჭირო აღარ იყო - მონაცემები იდენტურია N-BEATS-ის ეტაპისთვის გამოყენებულთან, ამიტომ მხოლოდ დავადასტურეთ იგივე რიცხვები ახალი experiment-ის ქვეშ ლოგირებით (`PatchTST_Data_Validation`), თავიდან აღმოჩენის გარეშე.

**Fold რაოდენობის საკითხი ხელახლა განვიხილეთ** მას შემდეგ, რაც გარე წყარომ (მცდარად) დაასკვნა, რომ 3 (ან თუნდაც 2) fold "მარტივად" შესაძლებელი იყო N-BEATS-ისთვის. სიღრმისეული შემოწმებით დავადასტურეთ: **3 fold ამ dataset-ზე მათემატიკურად შეუძლებელია** ნებისმიერი lookback window-თი (38-კვირიანი 3 fold-ი მოითხოვს 114 კვირას, 143-კვირიან სერიაზე ეს ტოვებს მხოლოდ 29 კვირას პირველი fold-ის ტრენინგისთვის - ნეგატიური lookback კი არ არსებობს). **2 fold ტექნიკურად შესაძლებელია**, მაგრამ მოითხოვს ≤29-კვირიან lookback-ს (ვერ ხედავს სრულ სეზონურ ციკლს). ამ ტრეიდ-ოფის გამო, PatchTST-სთვისაც **1 fold** გამოვიყენეთ, იმავე მიზეზების გამო.

#### შედეგები

| ეტაპი | WMAE (1 fold) |
|---|---|
| Baseline (`input_size=104`, `max_steps=500`) | 1802.91 |
| HPO-ს შემდეგ (Optuna, 20 trial) | **1555.89** |
| შემოწმება: `max_steps=800`-ზე refit | 1609.30 (**გაუარესდა** - იგივე overfitting-ის ნიმუში, რაც N-BEATS-ზეც დავინახეთ) |
| საბოლოო model, `max_steps=300`-ზე (HPO-ს კონფიგურაციის იდენტური) | **1555.89** (დადასტურდა) |

**საუკეთესო ჰიპერპარამეტრები:** `input_size=54, patch_len=16, stride=4, hidden_size=256, n_heads=2, lr=0.00066`.

ეს 1555.89 არის **საუკეთესო CV შედეგი ოთხივე მოდელს შორის** ამ ეტაპისთვის - სჯობია N-BEATS-საც (1767.99) და tree-model-ების ანალოგიურ single-fold შემოწმებასაც (LightGBM 1665.13, XGBoost 1615.19).

**Kaggle შედეგი:** Public **3019**, Private **3162** - იგივე მკვეთრი გაუარესება, რაც N-BEATS-ზეც დავინახეთ, და იმავე მიზეზით: ჩვენი ერთადერთი validation fold არ მოიცავს Thanksgiving/Christmas/Super Bowl-ს, ხოლო ვანილურ PatchTST-საც, N-BEATS-ის მსგავსად, საერთოდ არ გააჩნია holiday-სპეციფიკური feature. **საინტერესო დაკვირვება:** PatchTST-ის Kaggle შედეგი (3019/3162) მაინც შესამჩნევად სჯობია N-BEATS-ის (3707/3885) - შესაძლო ახსნა (ჰიპოთეზური, არა დადასტურებული) ის არის, რომ PatchTST-ის attention მექანიზმი patch-ებზე შესაძლოა უკეთ განზოგადდეს უცნობ სეზონურ pattern-ებზე, ვიდრე N-BEATS-ის fully-connected residual stacks, holiday-ცოდნის გარეშეც კი.

მოდელი დარეგისტრირებულია Model Registry-ში `mlflow.pyfunc`-ის საშუალებით (`walmart-patchtst-store-sales`), N-BEATS-ის იდენტური wrapper-პატერნით - იგივე შეზღუდვით, რომ საჭიროებს წინასწარ გლობალურად reindex-ილ input-ს, არა ნედლ test.csv-ს.


### DLinear



## ექსპერიმენტების ბმული
https://dagshub.com/amama22/walmart-forecasting.mlflow/#/experiments

