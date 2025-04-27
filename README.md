![StockPatchTST_banner_proportional_1024x400](https://github.com/user-attachments/assets/89bad72a-5a85-4883-9666-171765832b15)
![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.6.0-%23ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

## 📚 Table of Contents
- [Input Features](#-input-features)
- [Key Features](#-key-features)
- [Target Selection Strategy](#-target-selection-strategy)
- [Model Hyperparameters](#-model-hyperparameters)
- [Labeling & Ranking](#%EF%B8%8F-labeling--ranking)
- [Return Evaluation](#-return-evaluation)
- [Case Study](#-case-study-specific-stocks)
- [Environment](#-environment)
- [Experiment Notebook](#-experiment-notebook)
- [License & Acknowledgements](#-license--acknowledgements)

## Overview of Deep Learning Model Evolution
A chronological overview of model improvements, from LSTM to Transformer variants.

![model_history](https://github.com/user-attachments/assets/3723d215-bd04-4fae-94a2-d0b3c8534d82)

## Transformer-Based Ranking Model for Stock Selection
📘 See [TRANSFORMER.md](docs/TRANSFORMER.md) for a detailed architecture explanation.

![stock_patch_tst_model](https://github.com/user-attachments/assets/0d4c28c7-1bc2-4ea8-93cb-51896741d3c6)
> Patches are extracted via Conv1D to compress temporal data, and a Transformer encoder captures inter-patch dependencies for ranking prediction.

---

## 🧠 Input Features
📘 Full feature list: [FEATURES_TECHNICAL_INDICATORS.md](docs/FEATURES_TECHNICAL_INDICATORS.md)

- **Feature engineering:**
Ratio-based transformations, log-scaling for long-tail features, stable across different stocks.

### 🏭 Stock Metadata
- `industry_id`, `is_kospi`

### 💹 Price Flow Indicators
- `close_rate`, `open_to_close`, `high_to_low`, `rsi`, `ato`, `macd` related

### 📊 VWMA & Bollinger Bands
- `vwma5_gap`, `vwma20_gap`, `vwma_bb_width` etc.

### 🌍 Market Indices
- KOSPI, KOSDAQ, S&P500 Futures, Nasdaq 100 Futures, VIX, etc.

### 🔁 Volume & Flow
- Volume volatility ratios, net buy rates

### 📈 Candlestick Patterns
- Upper/lower tail ratios, body ratios

## Feature Heatmap
The following heatmap shows pairwise correlations between selected input features.
![heatmap](https://github.com/user-attachments/assets/d91c5728-0bc5-4b72-8a17-26e8d5f5fe5e)

## Distribution per Feature
Below are the individual distributions of input features used in model training.
|  |  |  |  |
|--|--|--|--|
| ![is_kospi](https://github.com/user-attachments/assets/792e12c2-18a5-4510-a2b4-3bc625395e2a) | ![close_rate](https://github.com/user-attachments/assets/33ffa740-deb6-402b-bff1-7987fcc633e4) | ![vwma5_gap](https://github.com/user-attachments/assets/4f7337dd-8ea2-4ccc-b29f-bfdb3380742e) | ![vwma20_gap](https://github.com/user-attachments/assets/156dd8c7-4285-487c-9a0d-21b98c93c906) | 
| ![vwma_bb_upper_ratio](https://github.com/user-attachments/assets/f5f9837d-a47e-49f0-8c75-cc1cde26fe46) | ![vwma_bb_lower_ratio](https://github.com/user-attachments/assets/a84162a2-b220-41d0-ac66-c33b0dc5c7e8) | ![vwma_bb_width](https://github.com/user-attachments/assets/af0529c2-769c-4144-9e8c-77e15075a88f) | ![kospi_vwma5_gap](https://github.com/user-attachments/assets/91c21f1d-aced-4f83-897d-eea6ba8b112b) | 
| ![kospi_vwma20_gap](https://github.com/user-attachments/assets/aa361acc-43cf-460d-9746-6b0b0ae37dd7) | ![kosdaq_vwma5_gap](https://github.com/user-attachments/assets/e0492513-b559-4252-bf33-cb0a90e85ad2) | ![kosdaq_vwma20_gap](https://github.com/user-attachments/assets/a0b89e95-c223-492e-8e57-484729e7d9a3) | ![open_to_close](https://github.com/user-attachments/assets/d51f6627-4954-4819-adbd-26049ae3e3b6) | 
| ![high_to_low](https://github.com/user-attachments/assets/0c9227c6-382d-4313-bc93-ea48f7ade250) | ![rsi](https://github.com/user-attachments/assets/cf91a673-10a3-445c-8e42-1e1f4667dd34) | ![macd_ratio](https://github.com/user-attachments/assets/b412ee5f-eac5-40c0-b1ad-0e6207528bbf) | ![macd_signal_ratio](https://github.com/user-attachments/assets/42141dca-b4c7-4a36-8e4d-32882cfaa7a2) | 
| ![macd_golden_cross](https://github.com/user-attachments/assets/7b9e0df4-e4ed-42a6-88a1-9e35d3271023) | ![macd_dead_cross](https://github.com/user-attachments/assets/f3645df7-eae2-4c0d-a831-3409d4e8b74e) | ![atr_ratio](https://github.com/user-attachments/assets/bc942118-0526-4bf5-bba0-a9bc776eded5) | ![kospi_close_rate](https://github.com/user-attachments/assets/1ed5cebd-60b2-4a09-8aeb-33cc1f125dec) | 
| ![kosdaq_close_rate](https://github.com/user-attachments/assets/8a06c2b9-f9b4-43fc-ab38-c17533253197) | ![sp500f_close_rate](https://github.com/user-attachments/assets/e3f0791c-80e7-43ce-8266-8c6dfdbce076) | ![sp500f_ma5_gap](https://github.com/user-attachments/assets/1f2fa8e7-ba72-4c85-8ad8-d387c308c596) | ![sp500f_ma20_gap](https://github.com/user-attachments/assets/835db012-3021-44fb-8926-0c02f09f287a) | 
| ![nasdaq100f_close_rate](https://github.com/user-attachments/assets/adc0fc5f-5878-4be0-b517-70190e2cfa43) | ![nasdaq100f_ma5_gap](https://github.com/user-attachments/assets/f697aa28-46f1-4bf2-b44b-aeac0106124d) | ![nasdaq100f_ma20_gap](https://github.com/user-attachments/assets/ad64c62d-adc0-45fd-bab6-296b9744dcf9) | ![vix_close_rate](https://github.com/user-attachments/assets/34a69461-9480-4532-a8dc-488609e87e67) | 
| ![sp500v_close_rate](https://github.com/user-attachments/assets/ffb5ec1a-452f-467e-a8ff-76925d3d92f8) | ![trading_volume_volatility_ratio](https://github.com/user-attachments/assets/532edee0-7dbf-4aab-bb94-c0af60e7eb12) | ![trading_change](https://github.com/user-attachments/assets/0ca135f0-0484-4927-ba1e-0df332a28a6d) | ![trading_rolling_change](https://github.com/user-attachments/assets/d527377c-7eda-439b-b612-b117d3f2e3d0) | 
| ![foreign_rate](https://github.com/user-attachments/assets/49a70ad6-f283-41ca-98eb-3eee0cd8d52f) | ![institution_rate](https://github.com/user-attachments/assets/7dda2579-2285-4ba8-a85d-13ce2e40aa6a) | ![individual_rate](https://github.com/user-attachments/assets/16a34c2b-214f-4667-8533-becf7ce6f830) | ![foreign_net_buy_days](https://github.com/user-attachments/assets/8298aaa9-8462-4e11-8fd2-16be299067b1) | 
| ![institution_net_buy_days](https://github.com/user-attachments/assets/3f7b75a6-0fd1-4397-a979-cf04cc846c8d) | ![candle_upper_tail_ratio](https://github.com/user-attachments/assets/686e45ec-d450-4a8d-b14a-bd017c775d7b) | ![candle_lower_tail_ratio](https://github.com/user-attachments/assets/059314ef-b958-4c1c-a1c5-5807d0b408b4) | ![candle_body_ratio](https://github.com/user-attachments/assets/62ff6aed-9915-47ba-9d1a-bb2b284db7c3) | 
| ![candle_sign](https://github.com/user-attachments/assets/99817acb-76f1-46b4-8742-3a25f782c81c) | | |

---

## 🔧 Key Features
- Patch-wise Transformer encoder
- Industry embeddings
- Soft label generation from clipped & shifted returns
- LambdaRankLoss for ranking optimization
- Real-time applicability (15:40–16:00 trading window)

## 🎯 Target Selection Strategy
- Top 200 stocks by daily trading volume
- Market cap ≥ 500B KRW
- Excludes limit-up and newly listed stocks

## 🔧 Model Hyperparameters
- Input Dimension: 41
- Industry Embedding Dimension: 4
- Model Dimension: 64
- Sliding Window Size: 30
- Patch Length: 6, Stride: 3
- Transformer Attention Heads: 4, Encoder Layers: 2
- Dropout: 2.5
- Learning Rate: 5e-4
- Weight Decay: 5e-5
- Early Stopping: 10 epochs

## 🏷️ Labeling & Ranking
- 5-day future returns are used as soft labels.
- **Training set:**
  - Clipping between -50% and +100%
  - Shifted to make minimum return zero
- **Validation/Test sets:**
  - No clipping (raw returns preserved)
  - Only shifted to make minimum return zero
- **Daily TOP3 stock selection** based on predicted scores.

### 🧮 5-Day Future Return Label Distribution (After Clipping & Shifting)
|  |  |  |
|--|--|--|
| ![train_soft_labels](https://github.com/user-attachments/assets/16188e31-3f70-4cdb-bf66-0c9165413bdc) | ![val_soft_labels](https://github.com/user-attachments/assets/e3115dc8-c14c-40bf-b473-d7232ef36f5a) | ![test_soft_labels](https://github.com/user-attachments/assets/f7fb2e29-e2ae-454d-8dee-c40286d01588) |

### 📉 Distribution of Predicted Scores
|  |  |
|--|--|
| ![val_top3_pred](https://github.com/user-attachments/assets/6fb8fe45-513e-418a-b622-30b88ec5065d) | ![test_top3_pred](https://github.com/user-attachments/assets/87bb62e1-91dd-4964-8cbf-43372c93ce83) |

### 🔍 Label Raw Distribution
|  |  |  |
|--|--|--|
| ![train_raw_labels](https://github.com/user-attachments/assets/3526516a-59f9-4cb0-beaa-b05c9afc5356) | ![val_raw_labels](https://github.com/user-attachments/assets/c16b2585-4cdb-42f3-b4fa-a5121b288b28) | ![test_raw_lables](https://github.com/user-attachments/assets/6103eb82-d610-4b58-8ce9-2d1fd56b364b) |

### 📚 Learn More: Ranking Metrics & Loss

For an in-depth explanation of the ranking metric and training objective used in this model, see:

- 📘 [NDCG Explained](docs/NDCG.md):  
  Understand how Normalized Discounted Cumulative Gain (NDCG) measures ranking quality in stock selection.

- ⚙️ [LambdaRank Loss Guide](docs/LAMBDA_RANK_LOSS.md):  
  Dive into the pairwise ranking loss function that optimizes NDCG by comparing stock relevance in every batch.

---

## 🔍 Post-Filtering Rules

### 🛒 Buy Signal Post-Filtering
- `pred_rank <= topn` (e.g., TOP3)
- `pred > 0.26` (confidence threshold)
- `atr_ratio > 0.03` (sufficient volatility)
- `close_rate > -10%` (avoiding sharp decliners)

**💡 Purpose:**
- Select only strong, active, and relatively stable candidates for buying.

---

### 💵 Sell Signal Detection

- **Stop-loss:** Closing price drops > 20% from highest price since entry.
- **Max holding period**: Forced exit after fixed days (e.g., 5 days).
- **Trailing Entry Extension**:
If a new buy signal occurs during holding, reset holding period.

**💡 Purpose:**
- Cut losses early
- Avoid stale positions
- Maximize gains on strong performers

---

## 📈 Return Evaluation
| Item                          | 2024 TOP3                       | 2025 TOP3                       |
|-------------------------------|----------------------------------|----------------------------------|
| Number of Samples             | 195,866                         | 42,966                          |
| Number of Buy / Sell Trades   | 147 / 147                       | 23 / 23                         |
| **Win Rate (Count, Ratio)**   | 94 trades (63.95%)            | 17 trades (73.91%)             |
| **Loss Rate (Count, Ratio)**  | 53 trades (36.05%)             | 6 trades (26.09%)             |
| Average Return (Win)          | 7.89%                           | 9.40%                           |
| Average Return (Loss)         | -6.89%                          | -4.28%                          |
| Avg. Holding Period (Win)     | 9.4 calendar days (6.3 trading days) | 9.9 calendar days (6.1 trading days) |
| Avg. Holding Period (Loss)    | 10.2 calendar days (7.0 trading days) | 15.3 calendar days (9.5 trading days) |
| Return Deciles                | [-26.9, -8.2, -3.6, -1.1, 0.7, 2.5, 4.2, 6.1, 9.2, 13.6, 34.9] | [-7.0, -5.5, -0.9, 1.3, 1.8, 5.6, 6.8, 9.6, 12.5, 14.4, 26.2] |
| Trade Capital                 | 10,000,000                     | 10,000,000                     |
| **Expected Net Return**       | **2.560%**                      | **5.831%**                      |
| **Cumulative Net Profit**     | **37,633,908**                  | **13,412,196**                  |

### 📅 Monthly Return
![val_month_roi](https://github.com/user-attachments/assets/9e55f812-926d-4ed0-9f40-fa1e4f2fe23d)
![test_month_roi](https://github.com/user-attachments/assets/4633754f-a7e1-4ac4-9b3c-a9ca90f71525)

### 📊 Daily Return
![val_january_roi](https://github.com/user-attachments/assets/7bdfe6e9-ae9e-4f27-a00d-f1b592ebaa15)
![test_january_roi](https://github.com/user-attachments/assets/55e121d0-1c05-4b39-b856-aee3b2f71c1a)

### 🏦 Return by Stock
![val_stocks_roi](https://github.com/user-attachments/assets/e599cd1b-140f-4915-8908-980215978025)

### 🤔 Return Distribution by Purchase Decision
|  |  |
|--|--|
| ![매수에따라n일후수익률평균값](https://github.com/user-attachments/assets/df8b2120-881e-400e-9e35-8e145c538f95) | ![매수에따라n일후수익률분포](https://github.com/user-attachments/assets/c6d90f5f-d62e-4e69-8521-abc3eff268f7) |
| ![매수False일때n일후수익률분포](https://github.com/user-attachments/assets/fdce00d5-03ad-4217-ad8d-ab9912a87061) | ![매수True일때n일후수익률분포](https://github.com/user-attachments/assets/ebc42a34-3aac-45d8-b30d-da7d2451d4c7) |

## 🔍 Case Study: Specific Stocks
![val_한화오션](https://github.com/user-attachments/assets/01d51ab8-93cf-460c-b383-25c1e93d9768)

---

## 💻 Environment

- Python 3.12.8
- PyTorch 2.6.0 + CUDA 12.6  
- pykrx 1.0.48  
- See [requirements.txt](./requirements.txt) for full dependency list

## 🧪 Experiment Notebook

The full end-to-end workflow is implemented in the following notebook:<br>
[Stock_PatchTST_Ranking.ipynb](./Stock_PatchTST_Ranking.ipynb)

This includes:
- Raw data retrieval
- Feature engineering and preprocessing
- Model training and validation
- Return evaluation and analysis
- Case studies on selected stocks

---

## 📄 License & Acknowledgements

This project is licensed under the MIT License.  
See the [LICENSE](./LICENSE) file for details.

This work is inspired by [PatchTST](https://github.com/yuqinie98/PatchTST).  
It was developed and tested on KRX daily stock data from 2020 to 2025.
KRX data was primarily retrieved using the [pykrx](https://github.com/sharebook-kr/pykrx) library.  
Industry classification codes were retrieved via the Open API from [Korea Investment & Securities](https://github.com/koreainvestment/open-trading-api).

