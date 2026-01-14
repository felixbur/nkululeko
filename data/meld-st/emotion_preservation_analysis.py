#!/usr/bin/env python3
"""
Emotion Preservation Analysis for MELD-ST Expressive Speech Translation

This script analyzes emotion consistency between English source utterances
and their Japanese/German translations to measure emotion preservation
in expressive speech translation tasks.

The analysis focuses on:
1. Cross-language emotion prediction agreement (Cohen's Kappa)
2. Emotion-specific preservation patterns
3. Translation quality impact on emotion preservation
4. Embedding similarity analysis for emotion alignment

Usage:
    python emotion_preservation_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, confusion_matrix, balanced_accuracy_score
import warnings

warnings.filterwarnings("ignore")

# Configure plotting
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["font.size"] = 12


class EmotionPreservationAnalyzer:
    """Analyzer for emotion preservation in expressive speech translation."""

    def __init__(self, results_root="./results"):
        self.results_root = Path(results_root)
        self.emotion_classes = [
            "anger",
            "disgust",
            "fear",
            "joy",
            "neutral",
            "sadness",
            "surprise",
        ]
        self.best_results = {}
        self.preservation_data = {}

    def load_best_results(self, config_pattern="*_BEST-dev_*.csv"):
        """Load the best results for each language pair configuration.

        Args:
            config_pattern: Glob pattern to match specific configuration files.
                          Default: "*_BEST-dev_*.csv"
                          Example: "train_dev_test_emotion_mlp_emotion2vec-large_32-64_drop-0-3_scale-standard_BEST-dev*"
        """

        # Define the best experiment configurations based on your analysis
        best_experiments = {
            # English to Japanese original
            "eng_jpn_orig": "meld_st_eng_jpn_eng_emotion_e2vlarge",
            # English to German original
            "eng_deu_orig": "meld_st_eng_deu_eng_emotion_e2vlarge",
            # Translated-language evaluation
            "eng_jpn_tran": "meld_st_eng_jpn_jpn_emotion_e2vlarge",
            "eng_deu_tran": "meld_st_eng_deu_deu_emotion_e2vlarge",
        }

        loaded_configs = {}  # Track base config names (without run numbers)

        for config_name, exp_name in best_experiments.items():
            exp_dir = self.results_root / exp_name
            results_dir = exp_dir / "results"

            if not results_dir.exists():
                print(f"Warning: Results directory not found for {exp_name}")
                continue

            # Find the best result file matching the specific configuration pattern
            result_files = list(results_dir.glob(config_pattern))

            if not result_files:
                print(
                    f"Warning: No files matching pattern '{config_pattern}' found for {exp_name}"
                )
                continue

            # Get the most recent best result
            best_file = max(result_files, key=lambda x: x.stat().st_mtime)

            try:
                df = pd.read_csv(best_file)

                # Extract file identifiers to match across language pairs
                df["file_id"] = (
                    df["file"].str.extract(r"test_(\d+)\.wav")[0].astype(int)
                )

                # Extract base config (without run number) for consistency checking
                filename = best_file.name
                # Remove the run number part (e.g., _0_032.csv -> _0_*.csv pattern)
                base_config = (
                    filename.rsplit("_", 2)[0]
                    if "_BEST-dev_" in filename
                    else filename.rsplit("_", 1)[0]
                )

                loaded_configs[config_name] = base_config

                self.best_results[config_name] = {
                    "data": df,
                    "experiment": exp_name,
                    "file": best_file.name,
                    "base_config": base_config,
                    "n_samples": len(df),
                }

                print(f"Loaded {config_name}: {len(df)} samples from {best_file.name}")

            except Exception as e:
                print(f"Error loading {exp_name}: {e}")

        # Check if all loaded configs use the same base configuration
        if len(loaded_configs) > 1:
            unique_configs = set(loaded_configs.values())
            if len(unique_configs) > 1:
                print("\n" + "!" * 70)
                print("WARNING: Different configurations detected across experiments!")
                print("!" * 70)
                for exp_name, config in loaded_configs.items():
                    print(f"  {exp_name}: {config}")
                print("This may affect cross-language comparison validity.")
                print(
                    "Consider using --config_pattern to force the same configuration."
                )
                print("!" * 70 + "\n")

        print(f"\nSuccessfully loaded {len(self.best_results)} experiment results")

    def analyze_emotion_preservation(self):
        """Analyze emotion preservation between source and target languages."""

        if len(self.best_results) < 2:
            print("Need at least 2 experiment results for preservation analysis")
            return

        # Analyze ENG->JPN preservation
        if "eng_jpn_orig" in self.best_results:
            self._analyze_language_pair("eng_jpn_orig", "English", "Japanese")

        # Analyze ENG->DEU preservation
        if "eng_deu_orig" in self.best_results:
            self._analyze_language_pair("eng_deu_orig", "English", "German")

        # Compare cross-language vs translation if available
        if "eng_jpn_orig" in self.best_results and "eng_jpn_tran" in self.best_results:
            self._compare_original_vs_translated("jpn", "Japanese")

        if "eng_deu_orig" in self.best_results and "eng_deu_tran" in self.best_results:
            self._compare_original_vs_translated("deu", "German")

    def _analyze_language_pair(self, config_key, source_lang, target_lang):
        """Analyze emotion preservation for a specific language pair."""

        data = self.best_results[config_key]["data"]

        # For translation tasks, we compare predictions on the same content
        # The 'truth' represents the original emotion, 'predicted' is what the model predicts

        # Calculate agreement metrics
        kappa = cohen_kappa_score(data["truth"], data["predicted"])
        accuracy = (data["truth"] == data["predicted"]).mean()
        balanced_acc = balanced_accuracy_score(data["truth"], data["predicted"])

        # Per-emotion preservation analysis
        emotion_preservation = {}
        for emotion in self.emotion_classes:
            emotion_mask = data["truth"] == emotion
            if emotion_mask.sum() > 0:
                correct_predictions = (
                    data.loc[emotion_mask, "predicted"] == emotion
                ).sum()
                total_instances = emotion_mask.sum()
                preservation_rate = correct_predictions / total_instances
                emotion_preservation[emotion] = {
                    "preservation_rate": preservation_rate,
                    "total_instances": total_instances,
                    "correct_predictions": correct_predictions,
                }

        # Store results
        self.preservation_data[f"{source_lang}_{target_lang}"] = {
            "config_key": config_key,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "kappa": kappa,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "n_samples": len(data),
            "emotion_preservation": emotion_preservation,
            "predictions": data[
                ["file_id", "truth", "predicted", "uncertainty"]
            ].copy(),
        }

        print(f"\n{source_lang} → {target_lang} Emotion Preservation:")
        print(f"  Cohen's Kappa: {kappa:.3f}")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Balanced Accuracy: {balanced_acc:.3f}")
        print(f"  Emotion Preservation Rates: {preservation_rate:.3f}")
        print(f"  Samples: {len(data)}")

    def _compare_original_vs_translated(self, lang_code, lang_name):
        """Compare original vs translated evaluation."""

        translation_key = f"eng_{lang_code}_orig"
        cross_key = f"eng_{lang_code}_tran"

        if (
            translation_key not in self.best_results
            or cross_key not in self.best_results
        ):
            return

        trans_data = self.best_results[translation_key]["data"]
        cross_data = self.best_results[cross_key]["data"]

        # Calculate metrics for both
        trans_acc = (trans_data["truth"] == trans_data["predicted"]).mean()
        cross_acc = (cross_data["truth"] == cross_data["predicted"]).mean()

        trans_kappa = cohen_kappa_score(trans_data["truth"], trans_data["predicted"])
        cross_kappa = cohen_kappa_score(cross_data["truth"], cross_data["predicted"])

        print(f"\n{lang_name} Translation vs Cross-Language Comparison:")
        print(f"  Translation Accuracy: {trans_acc:.3f} (Kappa: {trans_kappa:.3f})")
        print(f"  Cross-Language Accuracy: {cross_acc:.3f} (Kappa: {cross_kappa:.3f})")
        print(f"  Difference: {trans_acc - cross_acc:.3f}")

    def create_preservation_visualizations(self):
        """Create comprehensive visualizations for emotion preservation analysis."""

        if not self.preservation_data:
            print("No preservation data available for visualization")
            return

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))

        # 1. Overall preservation comparison
        ax1 = plt.subplot(2, 3, 1)
        self._plot_overall_preservation(ax1)

        # 2. Per-emotion preservation heatmap
        ax2 = plt.subplot(2, 3, 2)
        self._plot_emotion_preservation_heatmap(ax2)

        # 3. Confusion matrices
        ax3 = plt.subplot(2, 3, 3)
        ax4 = plt.subplot(2, 3, 4)
        self._plot_confusion_matrices([ax3, ax4])

        # 4. Emotion distribution comparison
        ax5 = plt.subplot(2, 3, 5)
        self._plot_emotion_distributions(ax5)

        # 5. Agreement analysis
        ax6 = plt.subplot(2, 3, 6)
        self._plot_agreement_analysis(ax6)

        plt.suptitle(
            "MELD-ST Emotion Preservation in Expressive Speech Translation",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig("emotion_preservation_analysis.pdf", dpi=300, bbox_inches="tight")
        print("Saved: emotion_preservation_analysis.pdf")

        return fig

    def _plot_overall_preservation(self, ax):
        """Plot overall preservation metrics comparison."""

        languages = []
        kappa_scores = []
        accuracy_scores = []

        for key, data in self.preservation_data.items():
            languages.append(f"{data['source_lang']}\n→ {data['target_lang']}")
            kappa_scores.append(data["kappa"])
            accuracy_scores.append(data["accuracy"])

        x = np.arange(len(languages))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2, kappa_scores, width, label="Cohen's Kappa", alpha=0.8
        )
        bars2 = ax.bar(
            x + width / 2, accuracy_scores, width, label="Accuracy", alpha=0.8
        )

        ax.set_xlabel("Language Pair")
        ax.set_ylabel("Score")
        ax.set_title("Emotion Preservation Metrics")
        ax.set_xticks(x)
        ax.set_xticklabels(languages)
        ax.legend()
        ax.set_ylim(0, 1)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

    def _plot_emotion_preservation_heatmap(self, ax):
        """Plot per-emotion preservation rates as heatmap."""

        # Prepare data for heatmap
        emotion_data = []
        language_pairs = []

        for key, data in self.preservation_data.items():
            language_pairs.append(f"{data['source_lang']} → {data['target_lang']}")
            row = []
            for emotion in self.emotion_classes:
                if emotion in data["emotion_preservation"]:
                    rate = data["emotion_preservation"][emotion]["preservation_rate"]
                else:
                    rate = 0.0
                row.append(rate)
            emotion_data.append(row)

        if emotion_data:
            emotion_df = pd.DataFrame(
                emotion_data,
                columns=[e.capitalize() for e in self.emotion_classes],
                index=language_pairs,
            )

            sns.heatmap(
                emotion_df,
                annot=True,
                fmt=".3f",
                cmap="RdYlBu_r",
                cbar_kws={"label": "Preservation Rate"},
                ax=ax,
            )
            ax.set_title("Per-Emotion Preservation Rates")
            ax.set_xlabel("Emotion Class")
            ax.set_ylabel("Language Pair")

    def _plot_confusion_matrices(self, axes):
        """Plot confusion matrices for each language pair."""

        for i, (key, data) in enumerate(self.preservation_data.items()):
            if i >= len(axes):
                break

            ax = axes[i]
            predictions = data["predictions"]

            # Create confusion matrix
            cm = confusion_matrix(
                predictions["truth"],
                predictions["predicted"],
                labels=self.emotion_classes,
            )
            cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

            sns.heatmap(
                cm_normalized,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                xticklabels=[e.capitalize() for e in self.emotion_classes],
                yticklabels=[e.capitalize() for e in self.emotion_classes],
                ax=ax,
            )

            ax.set_title(
                f'{data["source_lang"]} → {data["target_lang"]}\nKappa: {data["kappa"]:.3f}'
            )
            ax.set_xlabel("Predicted Emotion")
            ax.set_ylabel("True Emotion")

    def _plot_emotion_distributions(self, ax):
        """Plot emotion distribution comparison."""

        all_distributions = {}

        for key, data in self.preservation_data.items():
            predictions = data["predictions"]
            lang_pair = f"{data['source_lang']} → {data['target_lang']}"

            # Count emotion distributions
            true_dist = (
                predictions["truth"]
                .value_counts()
                .reindex(self.emotion_classes, fill_value=0)
            )
            pred_dist = (
                predictions["predicted"]
                .value_counts()
                .reindex(self.emotion_classes, fill_value=0)
            )

            all_distributions[f"{lang_pair} (True)"] = true_dist.values
            all_distributions[f"{lang_pair} (Pred)"] = pred_dist.values

        # Create grouped bar plot
        x = np.arange(len(self.emotion_classes))
        width = 0.15

        colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, len(all_distributions)))

        for i, (label, dist) in enumerate(all_distributions.items()):
            ax.bar(x + i * width, dist, width, label=label, color=colors[i], alpha=0.8)

        ax.set_xlabel("Emotion Class")
        ax.set_ylabel("Count")
        ax.set_title("Emotion Distribution Comparison")
        ax.set_xticks(x + width * (len(all_distributions) - 1) / 2)
        ax.set_xticklabels([e.capitalize() for e in self.emotion_classes])
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    def _plot_agreement_analysis(self, ax):
        """Plot detailed agreement analysis."""

        agreement_data = []

        for key, data in self.preservation_data.items():
            predictions = data["predictions"]
            lang_pair = f"{data['source_lang']} → {data['target_lang']}"

            # Calculate per-sample agreement (binary: correct/incorrect)
            agreement = (predictions["truth"] == predictions["predicted"]).astype(int)
            uncertainty = predictions["uncertainty"]

            # Create scatter plot of uncertainty vs agreement
            for i, (agree, unc) in enumerate(zip(agreement, uncertainty)):
                agreement_data.append(
                    {
                        "Language Pair": lang_pair,
                        "Agreement": agree,
                        "Uncertainty": unc,
                        "Sample": i,
                    }
                )

        if agreement_data:
            df_agreement = pd.DataFrame(agreement_data)

            # Box plot of uncertainty by agreement and language pair
            sns.boxplot(
                data=df_agreement,
                x="Language Pair",
                y="Uncertainty",
                hue="Agreement",
                ax=ax,
            )
            ax.set_title("Model Uncertainty vs Emotion Preservation")
            ax.set_ylabel("Model Uncertainty")
            ax.legend(title="Correct Prediction", labels=["No", "Yes"])

    def generate_preservation_report(self):
        """Generate a comprehensive emotion preservation report."""

        print("\n" + "=" * 80)
        print("EMOTION PRESERVATION IN EXPRESSIVE SPEECH TRANSLATION")
        print("=" * 80)

        if not self.preservation_data:
            print("No preservation data available for analysis")
            return

        # Overall summary
        print(f"\nAnalyzed {len(self.preservation_data)} language pairs:")

        for key, data in self.preservation_data.items():
            print(f"\n{data['source_lang']} → {data['target_lang']}:")
            print(f"  Samples: {data['n_samples']}")
            print(f"  Cohen's Kappa: {data['kappa']:.3f}")
            print(f"  Overall Accuracy: {data['accuracy']:.3f}")
            print(
                f"  Balanced Accuracy: {data['balanced_accuracy']:.3f}"
            )  # Placeholder if needed

            # Interpretation of kappa score
            if data["kappa"] < 0.20:
                interpretation = "Poor agreement"
            elif data["kappa"] < 0.40:
                interpretation = "Fair agreement"
            elif data["kappa"] < 0.60:
                interpretation = "Moderate agreement"
            elif data["kappa"] < 0.80:
                interpretation = "Good agreement"
            else:
                interpretation = "Excellent agreement"

            print(f"  Interpretation: {interpretation}")

        # Per-emotion analysis
        print("\nPER-EMOTION PRESERVATION ANALYSIS:")
        print("-" * 50)

        # Create comparison table
        emotion_table = []
        for emotion in self.emotion_classes:
            row = [emotion.capitalize()]
            for key, data in self.preservation_data.items():
                if emotion in data["emotion_preservation"]:
                    rate = data["emotion_preservation"][emotion]["preservation_rate"]
                    instances = data["emotion_preservation"][emotion]["total_instances"]
                    row.append(f"{rate:.3f} ({instances})")
                else:
                    row.append("N/A")
            emotion_table.append(row)

        # Print table
        headers = ["Emotion"] + [
            f"{d['source_lang']}→{d['target_lang']}"
            for d in self.preservation_data.values()
        ]

        print(f"{'Emotion':<12}", end="")
        for header in headers[1:]:
            print(f"{header:<15}", end="")
        print()
        print("-" * (12 + 15 * len(headers[1:])))

        for row in emotion_table:
            print(f"{row[0]:<12}", end="")
            for val in row[1:]:
                print(f"{val:<15}", end="")
            print()

        print("\nNote: Values show preservation_rate (total_instances)")

        # Language comparison
        if len(self.preservation_data) > 1:
            print("\nCROSS-LANGUAGE COMPARISON:")
            print("-" * 30)

            lang_pairs = list(self.preservation_data.keys())
            if len(lang_pairs) == 2:
                data1 = self.preservation_data[lang_pairs[0]]
                data2 = self.preservation_data[lang_pairs[1]]

                kappa_diff = data1["kappa"] - data2["kappa"]
                acc_diff = data1["accuracy"] - data2["accuracy"]

                better_data = data1 if kappa_diff > 0 else data2

                print(
                    f"Better emotion preservation: {better_data['source_lang']} → {better_data['target_lang']}"
                )
                print(f"Kappa difference: {abs(kappa_diff):.3f}")
                print(f"Accuracy difference: {abs(acc_diff):.3f}")

        # Save detailed report
        with open("emotion_preservation_report.txt", "w") as f:
            f.write("EMOTION PRESERVATION IN EXPRESSIVE SPEECH TRANSLATION\n")
            f.write("=" * 60 + "\n\n")

            for key, data in self.preservation_data.items():
                f.write(f"{data['source_lang']} → {data['target_lang']} Results:\n")
                f.write(f"Cohen's Kappa: {data['kappa']:.3f}\n")
                f.write(f"Accuracy: {data['accuracy']:.3f}\n")
                f.write(f"Samples: {data['n_samples']}\n")

                f.write("\nPer-emotion preservation rates:\n")
                for emotion, stats in data["emotion_preservation"].items():
                    rate = stats["preservation_rate"]
                    instances = stats["total_instances"]
                    f.write(f"  {emotion}: {rate:.3f} ({instances} instances)\n")
                f.write("\n")

        print("\nSaved: emotion_preservation_report.txt")

    def analyze_true_emotion_preservation(self):
        """Analyze true emotion preservation between source and target predictions."""

        preservation_results = {}

        # English-Japanese preservation
        if "eng_jpn_orig" in self.best_results and "eng_jpn_tran" in self.best_results:
            eng_data = self.best_results["eng_jpn_orig"]["data"]  # English audio
            jpn_data = self.best_results["eng_jpn_tran"]["data"]  # Japanese audio

            # Align by file_id to ensure same utterances
            merged = eng_data.merge(jpn_data, on="file_id", suffixes=("_eng", "_jpn"))

            if len(merged) > 0:
                # Calculate preservation based on prediction agreement
                preservation_results["English_Japanese"] = (
                    self._calculate_prediction_agreement(
                        merged, "eng", "jpn", "English", "Japanese"
                    )
                )

        # English-German preservation
        if "eng_deu_orig" in self.best_results and "eng_deu_tran" in self.best_results:
            eng_data = self.best_results["eng_deu_orig"]["data"]  # English audio
            deu_data = self.best_results["eng_deu_tran"]["data"]  # German audio

            # Align by file_id to ensure same utterances
            merged = eng_data.merge(deu_data, on="file_id", suffixes=("_eng", "_deu"))

            if len(merged) > 0:
                # Calculate preservation based on prediction agreement
                preservation_results["English_German"] = (
                    self._calculate_prediction_agreement(
                        merged, "eng", "deu", "English", "German"
                    )
                )

        return preservation_results

    def _calculate_prediction_agreement(
        self, merged_data, src_suffix, tgt_suffix, src_lang, tgt_lang
    ):
        """Calculate emotion preservation metrics based on prediction agreement and individual performance."""

        src_pred_col = f"predicted_{src_suffix}"
        tgt_pred_col = f"predicted_{tgt_suffix}"
        src_truth_col = f"truth_{src_suffix}"
        tgt_truth_col = f"truth_{tgt_suffix}"

        # Individual language performance vs ground truth
        src_accuracy = (merged_data[src_truth_col] == merged_data[src_pred_col]).mean()
        tgt_accuracy = (merged_data[tgt_truth_col] == merged_data[tgt_pred_col]).mean()

        src_balanced_acc = balanced_accuracy_score(
            merged_data[src_truth_col], merged_data[src_pred_col]
        )
        tgt_balanced_acc = balanced_accuracy_score(
            merged_data[tgt_truth_col], merged_data[tgt_pred_col]
        )

        # Overall agreement metrics
        agreement = merged_data[src_pred_col] == merged_data[tgt_pred_col]
        preservation_rate = agreement.mean()

        # Cohen's Kappa for prediction agreement
        kappa = cohen_kappa_score(merged_data[src_pred_col], merged_data[tgt_pred_col])

        # Balanced accuracy for prediction agreement
        balanced_acc_agreement = balanced_accuracy_score(
            merged_data[src_pred_col], merged_data[tgt_pred_col]
        )

        # Calculate BAR (Balanced Accuracy Ratio): target BA / source BA
        bar = tgt_balanced_acc / src_balanced_acc if src_balanced_acc > 0 else 0.0

        # Per-emotion preservation analysis
        emotion_preservation = {}
        for emotion in self.emotion_classes:
            # Find utterances where source predicted this emotion
            src_emotion_mask = merged_data[src_pred_col] == emotion

            if src_emotion_mask.sum() > 0:
                # Of those, how many did target also predict the same emotion?
                target_agreement = (
                    merged_data.loc[src_emotion_mask, tgt_pred_col] == emotion
                )
                emotion_preservation_rate = target_agreement.mean()

                emotion_preservation[emotion] = {
                    "preservation_rate": emotion_preservation_rate,
                    "total_instances": src_emotion_mask.sum(),
                    "preserved_instances": target_agreement.sum(),
                }

        # EPR (Emotion Preservation Rate) is the overall agreement rate (micro-average)
        # This is: number of samples with same prediction / total samples
        epr = preservation_rate

        # Calculate uncertainty correlation
        src_uncertainty = merged_data[f"uncertainty_{src_suffix}"]
        tgt_uncertainty = merged_data[f"uncertainty_{tgt_suffix}"]
        uncertainty_corr = src_uncertainty.corr(tgt_uncertainty)

        return {
            "source_lang": src_lang,
            "target_lang": tgt_lang,
            # Individual performance
            "source_accuracy": src_accuracy,
            "source_balanced_accuracy": src_balanced_acc,
            "target_accuracy": tgt_accuracy,
            "target_balanced_accuracy": tgt_balanced_acc,
            # Cross-language agreement
            "overall_preservation_rate": preservation_rate,
            "kappa": kappa,
            "balanced_accuracy": balanced_acc_agreement,
            "bar": bar,
            "epr": epr,
            "n_samples": len(merged_data),
            "emotion_preservation": emotion_preservation,
            "uncertainty_correlation": uncertainty_corr,
            "merged_data": merged_data[
                [
                    "file_id",
                    src_pred_col,
                    tgt_pred_col,
                    f"uncertainty_{src_suffix}",
                    f"uncertainty_{tgt_suffix}",
                    src_truth_col,  # Keep truth for reference
                ]
            ].copy(),
        }

    def print_true_preservation_summary(self, preservation_results):
        """Print comprehensive summary of emotion preservation between source and target predictions."""

        if not preservation_results:
            print("No preservation data available")
            return

        print("\n" + "=" * 80)
        print("EMOTION PRESERVATION ANALYSIS")
        print("=" * 80)

        for pair_name, data in preservation_results.items():
            src_lang = data["source_lang"]
            tgt_lang = data["target_lang"]

            print(f"\n{'=' * 80}")
            print(f"{src_lang} → {tgt_lang}")
            print(f"{'=' * 80}")

            print(f"\n1. INDIVIDUAL PERFORMANCE (vs Ground Truth):")
            print(f"   {src_lang}:")
            print(f"     Overall Accuracy:     {data['source_accuracy']:.3f}")
            print(f"     Balanced Accuracy:    {data['source_balanced_accuracy']:.3f}")

            print(f"\n   {tgt_lang}:")
            print(f"     Overall Accuracy:     {data['target_accuracy']:.3f}")
            print(f"     Balanced Accuracy:    {data['target_balanced_accuracy']:.3f}")

            print(f"\n2. CROSS-LANGUAGE PRESERVATION (Prediction Agreement):")
            print(f"     EPR:                  {data['epr']:.3f}")
            print(f"     Cohen's Kappa:        {data['kappa']:.3f}")
            print(f"     BAR (BA Ratio):       {data['bar']:.3f}")
            print(f"     Samples:              {data['n_samples']}")

            # Kappa interpretation
            if data["kappa"] < 0.20:
                interpretation = "Poor agreement"
            elif data["kappa"] < 0.40:
                interpretation = "Fair agreement"
            elif data["kappa"] < 0.60:
                interpretation = "Moderate agreement"
            elif data["kappa"] < 0.80:
                interpretation = "Good agreement"
            else:
                interpretation = "Excellent agreement"

            print(f"     Interpretation:       {interpretation}")

            print(f"\n3. PER-EMOTION PRESERVATION:")
            print(f"   (When {src_lang} predicts X, how often does {tgt_lang} agree?)")
            for emotion, stats in data["emotion_preservation"].items():
                rate = stats["preservation_rate"]
                total = stats["total_instances"]
                preserved = stats["preserved_instances"]
                print(
                    f"     {emotion.capitalize():<10}: {rate:.3f} ({preserved}/{total})"
                )

    def create_true_preservation_visualizations(self, preservation_results):
        """Create visualizations for true emotion preservation."""

        if not preservation_results:
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Overall preservation comparison
        ax1 = axes[0]
        lang_pairs = []
        preservation_rates = []
        kappa_scores = []

        for pair_name, data in preservation_results.items():
            lang_pairs.append(f"{data['source_lang']}\n→ {data['target_lang']}")
            preservation_rates.append(data["overall_preservation_rate"])
            kappa_scores.append(data["kappa"])

        x = np.arange(len(lang_pairs))
        width = 0.35

        bars1 = ax1.bar(
            x - width / 2,
            preservation_rates,
            width,
            label="Preservation Rate",
            alpha=0.8,
        )
        bars2 = ax1.bar(
            x + width / 2, kappa_scores, width, label="Cohen's Kappa", alpha=0.8
        )

        ax1.set_xlabel("Language Pair")
        ax1.set_ylabel("Score")
        ax1.set_title("True Emotion Preservation\n(Source vs Target Predictions)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(lang_pairs)
        ax1.legend()
        ax1.set_ylim(0, 1)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

        # 2. Per-emotion preservation heatmap
        ax2 = axes[1]
        emotion_data = []
        language_labels = []

        for pair_name, data in preservation_results.items():
            language_labels.append(f"{data['source_lang']} → {data['target_lang']}")
            row = []
            for emotion in self.emotion_classes:
                if emotion in data["emotion_preservation"]:
                    rate = data["emotion_preservation"][emotion]["preservation_rate"]
                else:
                    rate = 0.0
                row.append(rate)
            emotion_data.append(row)

        if emotion_data:
            emotion_df = pd.DataFrame(
                emotion_data,
                columns=[e.capitalize() for e in self.emotion_classes],
                index=language_labels,
            )

            sns.heatmap(
                emotion_df,
                annot=True,
                fmt=".3f",
                cmap="RdYlBu_r",
                cbar_kws={"label": "Preservation Rate"},
                ax=ax2,
            )
            ax2.set_title("Per-Emotion Preservation Rates\n(Source → Target Agreement)")
            ax2.set_xlabel("Emotion Class")
            ax2.set_ylabel("Language Pair")

        plt.suptitle(
            "True Emotion Preservation Analysis\n(Source vs Target Prediction Agreement)",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(
            "true_emotion_preservation_analysis.pdf", dpi=300, bbox_inches="tight"
        )
        print("Saved: true_emotion_preservation_analysis.pdf")

        return fig

    def create_uncertainty_agreement_plots(self, preservation_results):
        """Create separate plots for uncertainty vs agreement analysis."""

        if not preservation_results:
            return

        # Determine number of plots needed
        n_pairs = len(preservation_results)

        # Create figure with appropriate size
        if n_pairs == 1:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            axes = [ax]
        else:
            fig, axes = plt.subplots(1, n_pairs, figsize=(8 * n_pairs, 6))

        # Create uncertainty vs agreement plots for each language pair
        for i, (pair_name, data) in enumerate(preservation_results.items()):
            ax = axes[i] if n_pairs > 1 else axes[0]
            merged_data = data["merged_data"]

            # Create scatter plot of source vs target uncertainties, colored by agreement
            src_uncertainty = merged_data.iloc[:, 3]  # uncertainty_src
            tgt_uncertainty = merged_data.iloc[:, 4]  # uncertainty_tgt
            agreement = (
                merged_data.iloc[:, 1] == merged_data.iloc[:, 2]
            )  # pred_src == pred_tgt

            # Plot with different colors for agreement/disagreement
            agree_mask = agreement
            disagree_mask = ~agreement

            if agree_mask.sum() > 0:
                ax.scatter(
                    src_uncertainty[agree_mask],
                    tgt_uncertainty[agree_mask],
                    c="green",
                    alpha=0.6,
                    label="Agreement",
                    s=30,
                )
            if disagree_mask.sum() > 0:
                ax.scatter(
                    src_uncertainty[disagree_mask],
                    tgt_uncertainty[disagree_mask],
                    c="red",
                    alpha=0.6,
                    label="Disagreement",
                    s=30,
                )

            ax.set_xlabel(f'{data["source_lang"]} Model Uncertainty')
            ax.set_ylabel(f'{data["target_lang"]} Model Uncertainty')
            ax.set_title(
                f'{data["source_lang"]} → {data["target_lang"]}\nUncertainty vs Prediction Agreement'
            )
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add correlation and agreement statistics
            correlation = data["uncertainty_correlation"]
            agreement_rate = data["overall_preservation_rate"]

            stats_text = (
                f"Correlation: {correlation:.3f}\nAgreement: {agreement_rate:.3f}"
            )
            ax.text(
                0.05,
                0.95,
                stats_text,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                verticalalignment="top",
                fontsize=10,
            )

            # Add diagonal reference line
            max_uncertainty = max(src_uncertainty.max(), tgt_uncertainty.max())
            min_uncertainty = min(src_uncertainty.min(), tgt_uncertainty.min())
            ax.plot(
                [min_uncertainty, max_uncertainty],
                [min_uncertainty, max_uncertainty],
                "k--",
                alpha=0.3,
                linewidth=1,
                label="Perfect correlation",
            )

        plt.suptitle(
            "Model Uncertainty vs Prediction Agreement\nCross-Language Emotion Recognition",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(
            "uncertainty_vs_agreement_analysis.pdf", dpi=300, bbox_inches="tight"
        )
        print("Saved: uncertainty_vs_agreement_analysis.pdf")

        return fig


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze emotion preservation in expressive speech translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default configuration
  python emotion_preservation_analysis.py
  
  # Use specific configuration pattern
  python emotion_preservation_analysis.py --config_pattern "train_dev_test_emotion_mlp_emotion2vec-large_32-64_drop-0-3_scale-standard_BEST-dev*"
  
  # Use custom results directory
  python emotion_preservation_analysis.py --results_root ../../results
        """,
    )

    parser.add_argument(
        "--results_root",
        type=str,
        default="./results",
        help="Path to results directory (default: ./results)",
    )

    parser.add_argument(
        "--config_pattern",
        type=str,
        default="*_BEST-dev_*.csv",
        help="Glob pattern to match result files (default: *_BEST-dev_*.csv)",
    )

    args = parser.parse_args()

    print("MELD-ST Emotion Preservation Analysis")
    print("=" * 50)
    print("Analyzing emotion consistency in expressive speech translation")
    print("English → Japanese / German")
    print(f"Results directory: {args.results_root}")
    print(f"Config pattern: {args.config_pattern}\n")

    # Initialize analyzer
    analyzer = EmotionPreservationAnalyzer(results_root=args.results_root)

    # Load best results
    print("\nLoading best results for each language pair...")
    analyzer.load_best_results(config_pattern=args.config_pattern)

    if not analyzer.best_results:
        print("No experiment results found!")
        print("Make sure you have the following experiment results:")
        print("  - meld_st_eng_jpn_eng_emotion_e2vlarge")
        print("  - meld_st_eng_deu_eng_emotion_e2vlarge")
        return

    # Analyze TRUE emotion preservation (source vs target predictions)
    print("\nAnalyzing emotion preservation (source vs target predictions)...")
    preservation_results = analyzer.analyze_true_emotion_preservation()

    if preservation_results:
        analyzer.print_true_preservation_summary(preservation_results)

        # Create visualizations
        print("\nCreating visualizations...")
        try:
            analyzer.create_true_preservation_visualizations(preservation_results)
        except Exception as e:
            print(f"Error creating visualizations: {e}")

        # Create uncertainty vs agreement plots
        print("\nCreating uncertainty vs agreement plots...")
        try:
            analyzer.create_uncertainty_agreement_plots(preservation_results)
        except Exception as e:
            print(f"Error creating uncertainty vs agreement plots: {e}")

        print("\nAnalysis complete! Generated files:")
        print("  - true_emotion_preservation_analysis.pdf")
        print("  - uncertainty_vs_agreement_analysis.pdf")
    else:
        print(
            "Could not calculate preservation - need both source and target predictions"
        )

    print("\n" + "=" * 50)
    print("SUMMARY:")
    print("1. Individual Performance: Each language's accuracy vs ground truth (BA)")
    print(
        "2. Cross-Language Preservation: Prediction agreement metrics (BAR, EPR, Kappa)"
    )
    print("3. Per-Emotion Analysis: Emotion-specific preservation rates")
    print("Higher BAR and EPR indicate better emotion preservation in translation.")


if __name__ == "__main__":
    main()
