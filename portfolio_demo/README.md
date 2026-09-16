# Reproducible MLOps demonstration

A small, CPU-only experiment that runs without the research project's datasets,
Spark, Kafka, cloud accounts, or an API key. This is a separate synthetic demonstration;
it does not validate the full research pipeline or any real-network performance claims.

## Run from the repository root

Python 3.11+ (do not use the old checked-in virtual environment):

```bash
python -m venv .venv-demo
# Linux/macOS:
source .venv-demo/bin/activate
# Windows PowerShell alternative: .venv-demo\Scripts\Activate.ps1
python -m pip install -r portfolio_demo/requirements.txt
python -m portfolio_demo.run
python -m pytest -q portfolio_demo/tests
```

Inspect `portfolio_demo/output/run.json` for the run ID and metrics.
`output/artifacts/` contains results, dataset hashes/split IDs and the selected model.
`output/mlruns/` and `output/mlartifacts/` contain local MLflow tracking and artifacts.
No tracking server or UI is required. Only load joblib model files you trust.

## What happens

1. Generate two fixed-seed windows with telecom-like latency, loss and throughput.
   Labels are synthetic. The second window changes feature distributions and the label rule.
2. Train a random forest on 70% of the reference window; reserve 30% as a baseline test set.
3. Split the shifted window into 50% labeled calibration, 25% validation and 25% holdout.
4. Compare calibration means to training means in units of training standard deviation.
   A maximum shift above 0.75 triggers candidate training on labeled calibration rows.
5. Promote the candidate only if validation F1 improves by at least 0.01.
6. After the choice is fixed, measure both the baseline and selected model on the untouched
   shifted holdout. Log configuration, scores, model, versions and data hashes to MLflow.

A mean-shift score is a **feature drift heuristic**, not a statistical test or proof of
concept drift. It can miss distribution changes that preserve the mean. Real adaptation
also depends on label availability and delay; this demo assumes labels are already available.
No deployed model is replaced: “promotion” selects only a local demonstration artifact.

## Measured example (seed 42)

| Evaluation set | Model | F1 |
|---|---|---:|
| Reference test | Baseline | 0.9062 |
| Shifted validation | Baseline | 0.8510 |
| Shifted validation | Candidate | 0.9326 |
| Shifted holdout | Baseline | 0.8442 |
| Shifted holdout | Selected candidate | 0.9432 |

These are actual local demo results, not real telecom accuracy or expected production
improvements. See [example_results.json](example_results.json) for all metrics and versions.
The fixed generator/seed supports reproduction; small numerical differences across
platforms or dependency versions are possible. Run IDs and paths naturally change.

## Automated checks

Four tests cover reproducible data, a detectable shift, disjoint training/evaluation
records, validation-based selection, model reload, real MLflow metrics/artifacts, and
the no-drift path. The dedicated GitHub workflow runs this demo only; legacy research
and infrastructure tests were not run as part of this isolated demonstration.

## Sena için anlatım ve alıştırma

**90 saniyelik anlatım:** “İlk model referans veride eğitiliyor. Yeni pencerenin
özellik dağılımı değişince aday model eğitiyorum. İyileşmeyi doğrulama kümesinde ölçüp
kararı kilitliyorum. En son hiç kullanılmamış test verisindeki sonucu raporluyorum.
MLflow parametreleri, sonuçları ve model dosyasını saklıyor. Veriler sentetik.”

**Kendi yapacağın değişiklik:** Ayrı bir çıktı klasöründe seed 7 ile çalıştır.
F1 değişirse bunun nedenini açıkla. `min_f1_gain` değerini 0.5 yapıp adayın neden
seçilmediğini incele. Holdout skorunu gördükten sonra parametre ayarlayıp aynı skoru
“bağımsız test” diye sunmaman gerektiğini anlat.

Sorular: precision ile recall farkı; neden üç ayrı shifted split var; drift neden
etiketsiz ölçülebilir ama performans neden etiket gerektirir; yanlış alarmın maliyeti;
üretimde geri alma, erişim kontrolü ve izleme nasıl eklenir?
