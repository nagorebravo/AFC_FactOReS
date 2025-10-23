import os
import json
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
from io import StringIO
import sys
import argparse
import numpy as np
import sklearn
import scipy
import nltk
from nltk import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)


def plot_confusion_matrix(y_pred, y_true, labels, model_name, stance, output_path):
    if "zero_shot" in output_path:
        title_name = "Zero-Shot"
    elif "few_shot" in output_path:
        title_name = "Few-Shot"
    elif "chain_of_thought" in output_path:
        title_name = "Chain of Thought"
    else:
        title_name = "Few-Shot Cot"

    src_labels = [x["pred_label"].lower() for x in y_pred]
    tgt_labels = [x["label"].lower() for x in y_true]
    cm = confusion_matrix(tgt_labels, src_labels, labels=FULL_LABELS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=SHORT_LABELS)

    plt.rcParams.update({'font.size': 20})

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap='Blues', ax=ax, colorbar=True)
    disp.im_.set_clim(0, 335)

    # Cambiar tamaño de etiquetas de clases
    ax.set_xticklabels(SHORT_LABELS, fontsize=20)
    ax.set_yticklabels(SHORT_LABELS, fontsize=20)

    # Cambiar tamaño de "True label" y "Predicted label"
    ax.set_xlabel("Predicted label", fontsize=20)
    ax.set_ylabel("True label", fontsize=20)

    plt.title(f"{title_name} - {model_name} - {stance} - Confusion Matrix", fontsize=18)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def plot_class_metrics3(y_pred, y_true, labels, model_name, stance, output_path):
    if "zero_shot" in output_path:
        title_name = "Zero-Shot"
    elif "few_shot" in output_path:
        title_name = "Few-Shot"
    elif "chain_of_thought" in output_path:
        title_name = "Chain of Thought"
    else:
        title_name = "Few-Shot Cot" 

    src_labels = [x["pred_label"].lower() for x in y_pred]
    tgt_labels = [x["label"].lower() for x in y_true]
    report = classification_report(tgt_labels, src_labels, labels=labels, output_dict=True)
    
    metrics = ['precision', 'recall', 'f1-score']
    data = {metric: [report[label][metric] for label in labels] for metric in metrics}
    
    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(10, 6))
    for i, metric in enumerate(metrics):
        bars = plt.bar(x + i * width, data[metric], width=width, label=metric)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.01, f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    plt.xticks(x + width, labels, rotation=45)
    plt.xlabel("Veracity Label")
    plt.ylabel("Score")
    plt.title(f"{title_name} - {model_name} - {stance} - Metrics per label")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Class metrics saved to {output_path}")


def plot_class_metrics2(y_pred, y_true, labels, model_name, stance, output_path):
    if "zero_shot" in output_path:
        title_name = "Zero-Shot"
    elif "few_shot" in output_path:
        title_name = "Few-Shot"
    elif "chain_of_thought" in output_path:
        title_name = "Chain of Thought"
    else:
        title_name = "Few-Shot Cot" 

    src_labels = [x["pred_label"].lower() for x in y_pred]
    tgt_labels = [x["label"].lower() for x in y_true]
    report = classification_report(tgt_labels, src_labels, labels=labels, output_dict=True)
    
    metrics = ['precision', 'recall', 'f1-score']
    data = {metric: [report[label][metric] for label in labels] for metric in metrics}
    
    # Mapeo para mostrar labels más cortas y capitalizadas
    label_display_map = {
        "supported": "Supported",
        "refuted": "Refuted",
        "not enough evidence": "Not Enough Evidence",
        "conflicting/cherrypicking": "Conflicting/Cherr."
    }
    display_labels = [label_display_map.get(lbl.lower(), lbl.capitalize()) for lbl in labels]
    
    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(10, 6))
    for i, metric in enumerate(metrics):
        bars = plt.bar(x + i * width, data[metric], width=width, label=metric.capitalize())  # Leyenda con mayúscula inicial
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.01, f'{height:.3f}', 
                     ha='center', va='bottom', fontsize=8)

    plt.xticks(x + width, display_labels)
    plt.xlabel("Veracity Label")
    plt.ylabel("Score")
    plt.title(f"{title_name} - {model_name} - {stance} - Metrics per Label")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Class metrics saved to {output_path}")


def plot_class_metrics(y_pred, y_true, labels, model_name, stance, output_path):
    if "zero_shot" in output_path:
        title_name = "Zero-Shot"
    elif "few_shot" in output_path:
        title_name = "Few-Shot"
    elif "chain_of_thought" in output_path:
        title_name = "Chain of Thought"
    else:
        title_name = "Few-Shot Cot" 

    # Obtener etiquetas predichas y reales
    src_labels = [x["pred_label"].lower() for x in y_pred]
    tgt_labels = [x["label"].lower() for x in y_true]

    # Reporte solo con f1-score
    report = classification_report(tgt_labels, src_labels, labels=labels, output_dict=True)
    f1_scores = [report[label]['f1-score'] for label in labels]

    # Mapeo para mostrar etiquetas cortas y capitalizadas
    label_display_map = {
        "supported": "S",
        "refuted": "R",
        "not enough evidence": "N",
        "conflicting evidence/cherrypicking": "C"
    }
    display_labels = [label_display_map.get(lbl.lower(), lbl.capitalize()) for lbl in labels]

    # Colores por clase
    color_map = {
        "supported": "#79CE79",           # Verde claro
        "refuted": "#E4837C",             # Rojo claro
        "conflicting evidence/cherrypicking": "#EEAD87",  # Naranja claro
        "not enough evidence": "#9FC6DF"  # Azul claro
    }
    colors = [color_map.get(lbl.lower(), "#D3D3D3") for lbl in labels]

    x = np.arange(len(labels))
    width = 0.4

    # Figura más estrecha
    plt.figure(figsize=(6, 5))
    bars = plt.bar(x, f1_scores, color=colors, width=width)

    # Etiquetas en las barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.01, f'{height:.3f}', 
                 ha='center', va='bottom', fontsize=8)

    plt.xticks(x, display_labels)
    plt.xlabel("Veracity Label")
    plt.ylabel("F1-Score")
    plt.title(f"{title_name} - {model_name} - {stance} - F1-score per Label")
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Class metrics saved to {output_path}")




def plot_macro_metrics(y_pred, y_true, labels, output_path):
    src_labels = [x["pred_label"].lower() for x in y_pred]
    tgt_labels = [x["label"].lower() for x in y_true]
    
    report = classification_report(tgt_labels, src_labels, labels=labels, output_dict=True)
    
    macro_metrics = report['macro avg']
    metrics = ['precision', 'recall', 'f1-score']
    scores = [macro_metrics[metric] for metric in metrics]
    
    x = np.arange(len(metrics))
    
    plt.figure(figsize=(6, 5))
    bars = plt.bar(x, scores, width=0.4)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.01, f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    plt.xticks(x, metrics)
    plt.ylim(0, 1.1)
    plt.ylabel("Score")
    plt.title("Macro-Averaged Metrics")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Macro metrics saved to {output_path}")




def pairwise_meteor(candidate, reference):
    return nltk.translate.meteor_score.single_meteor_score(
        word_tokenize(reference), word_tokenize(candidate)
    )


def compute_all_pairwise_scores(src_data, tgt_data, metric):
    scores = np.empty((len(src_data), len(tgt_data)))
    for i, src in enumerate(src_data):
        for j, tgt in enumerate(tgt_data):
            scores[i][j] = metric(src, tgt)
    return scores


def print_with_space(left, right, left_space=45):
    print_spaces = " " * (left_space - len(left))
    print(left + print_spaces + right)


class Evaluator:

    averitec_verdicts = [
        "supported",
        "refuted",
        "not enough evidence",
        "conflicting evidence/cherrypicking"
        #"unknown"
    ]


    factores_verdicts = [
        "supported",
        "refuted",
        "not enough evidence"
    ]


    pairwise_metric = None
    max_questions = 10
    metric = None
    averitec_reporting_levels = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]

    def __init__(self, metric="meteor", dataset_type="averitec"):
        self.metric = metric
        if metric == "meteor":
            self.pairwise_metric = pairwise_meteor
        
        if dataset_type == "averitec":
            self.verdicts = self.averitec_verdicts
        elif dataset_type == "factores":
            self.verdicts = self.factores_verdicts
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        

    def evaluate_averitec_veracity_by_type(self, srcs, tgts, threshold=0.25):
        types = {}
        for src, tgt in zip(srcs, tgts):
            score = self.compute_pairwise_evidence_score(src, tgt)
            if score <= threshold:
                score = 0
            for t in tgt["claim_types"]:
                if t not in types:
                    types[t] = []
                types[t].append(score)
        return {t: np.mean(v) for t, v in types.items()}

    def evaluate_averitec_score(self, srcs, tgts):
        scores = []
        for src, tgt in zip(srcs, tgts):
            score = self.compute_pairwise_evidence_score(src, tgt)
            this_example_scores = [0.0 for _ in self.averitec_reporting_levels]
            for i, level in enumerate(self.averitec_reporting_levels):
                if score > level:
                    this_example_scores[i] = src["pred_label"] == tgt["label"]
            scores.append(this_example_scores)
        return np.mean(np.array(scores), axis=0)

    
    def evaluate_veracity(self, src, tgt):
        src_labels = [x["pred_label"].lower() for x in src]
        tgt_labels = [x["label"].lower() for x in tgt]

        acc = np.mean([s == t for s, t in zip(src_labels, tgt_labels)])
        f1 = {
            self.verdicts[i]: x
            for i, x in enumerate(
                sklearn.metrics.f1_score(
                    tgt_labels, src_labels, labels=self.verdicts, average=None
                )
            )
        }
        f1["macro"] = sklearn.metrics.f1_score(
            tgt_labels, src_labels, labels=self.verdicts, average="macro"
        )
        f1["acc"] = acc
        return f1
    
    

    def evaluate_questions_only(self, srcs, tgts):
        all_utils = []
        for src, tgt in zip(srcs, tgts):
            if "evidence" not in src:
                src_questions = self.extract_full_comparison_strings(
                    src, is_target=False
                )[: self.max_questions]
            else:
                src_questions = [
                    qa["question"] for qa in src["evidence"][: self.max_questions]
                ]
            tgt_questions = [qa["question"] for qa in tgt["questions"]]
            pairwise_scores = compute_all_pairwise_scores(
                src_questions, tgt_questions, self.pairwise_metric
            )
            assignment = scipy.optimize.linear_sum_assignment(
                pairwise_scores, maximize=True
            )
            assignment_utility = pairwise_scores[assignment[0], assignment[1]].sum()
            reweight_term = 1 / float(len(tgt_questions))
            assignment_utility *= reweight_term
            all_utils.append(assignment_utility)
        return np.mean(all_utils)

    def evaluate_questions_and_answers(self, srcs, tgts):
        all_utils = []
        for src, tgt in zip(srcs, tgts):
            src_strings = self.extract_full_comparison_strings(src, is_target=False)[
                : self.max_questions
            ]
            tgt_strings = self.extract_full_comparison_strings(tgt)
            pairwise_scores = compute_all_pairwise_scores(
                src_strings, tgt_strings, self.pairwise_metric
            )
            assignment = scipy.optimize.linear_sum_assignment(
                pairwise_scores, maximize=True
            )
            assignment_utility = pairwise_scores[assignment[0], assignment[1]].sum()
            reweight_term = 1 / float(len(tgt_strings))
            assignment_utility *= reweight_term
            all_utils.append(assignment_utility)
        return np.mean(all_utils)

    def compute_pairwise_evidence_score(self, src, tgt):
        src_strings = self.extract_full_comparison_strings(src, is_target=False)[
            : self.max_questions
        ]
        tgt_strings = self.extract_full_comparison_strings(tgt)
        pairwise_scores = compute_all_pairwise_scores(
            src_strings, tgt_strings, self.pairwise_metric
        )
        assignment = scipy.optimize.linear_sum_assignment(
            pairwise_scores, maximize=True
        )
        assignment_utility = pairwise_scores[assignment[0], assignment[1]].sum()
        reweight_term = 1 / float(len(tgt_strings))
        assignment_utility *= reweight_term
        return assignment_utility

    def extract_full_comparison_strings(self, example, is_target=True):
        example_strings = []
        if is_target:
            if "questions" in example:
                for evidence in example["questions"]:
                    if not isinstance(evidence["answers"], list):
                        evidence["answers"] = [evidence["answers"]]
                    for answer in evidence["answers"]:
                        s = evidence["question"] + " " + answer["answer"]
                        if "answer_type" in answer and answer["answer_type"] == "Boolean":
                            s += ". " + answer.get("boolean_explanation", "")
                        example_strings.append(s)
                    if len(evidence["answers"]) == 0:
                        example_strings.append(
                            evidence["question"] + " No answer could be found."
                        )
        else:
            if "evidence" in example:
                for evidence in example["evidence"]:
                    example_strings.append(
                        evidence["question"] + " " + evidence["answer"]
                    )
        if "string_evidence" in example:
            for full_string_evidence in example["string_evidence"]:
                example_strings.append(full_string_evidence)
        return example_strings



def load_true_labels(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {item['claim'].lower(): item['label'].lower() for item in data}



def evaluate_predictions_distribution(predictions_path, true_labels_path):
    with open(predictions_path, 'r', encoding='utf-8') as f:
        preds = json.load(f)
    true_labels_dict = load_true_labels(true_labels_path)
    print("length of true_labels_dict: ", len(true_labels_dict))
    print("length of preds: ", len(preds))

    matched_preds = []
    matched_truths = []

    for item in preds:
        claim = item['claim'].lower()
        pred_label = item['pred_label'].lower()
        if claim in true_labels_dict:
            matched_preds.append(pred_label)
            matched_truths.append(true_labels_dict[claim])
        else:
            print(f"Claim not found in ground truth: {claim}")

    ### PREDICTIONS DISTRIBUTION
    label_counts = Counter(matched_preds)
    total = sum(label_counts.values())

    print(f'\nClass distribution for file {predictions_path}:')
    for label, count in label_counts.items():
        percentage = (count / total) * 100
        print(f"- {label}: {count} ({percentage:.2f}%)")

    max_count = max(label_counts.values())
    min_count = min(label_counts.values())
    percentage_diff = ((max_count - min_count) / total) * 100


    if percentage_diff <= 10:
        print("\nPreds are balanced.")
        #accuracy = accuracy_score(matched_truths, matched_preds)
        #print(f"Accuracy: {accuracy:.4f}")
    else:
        print("\nPreds are imbalanced.")
        #f1 = f1_score(matched_truths, matched_preds, average='macro')
        #print(f"Macro F1-score: {f1:.4f}")



    ### DEV DISTRIBUTION
    label_counts_truths = Counter(matched_truths)
    total_truths = sum(label_counts_truths.values())

    print(f'\nClass distribution for file {true_labels_path}:')
    for label_truths, count_truths in label_counts_truths.items():
        percentage_truths = (count_truths / total_truths) * 100
        print(f"- {label_truths}: {count_truths} ({percentage_truths:.2f}%)")

    max_count_truths = max(label_counts_truths.values())
    min_count_truths = min(label_counts_truths.values())
    percentage_diff_truths = ((max_count_truths - min_count_truths) / total_truths) * 100


    if percentage_diff_truths <= 10:
        print("\nClasses are balanced.")
        #accuracy = accuracy_score(matched_truths, matched_preds)
        #print(f"Accuracy: {accuracy:.4f}")
    else:
        print("\nClasses are imbalanced.")
        #f1 = f1_score(matched_truths, matched_preds, average='macro')
        #print(f"Macro F1-score: {f1:.4f}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", choices=["averitec_dataset", "factores_dataset"], required=True)


    modes = [
    #"zero_shot",                    #withStance
    #"few_shot",                     #withStance
    #"chain_of_thought",             #withStance
    "zero_shot_withStance_pydantic",
    "zero_shot_noStance_pydantic",
    "few_shot_withStance_pydantic",
    "few_shot_noStance_pydantic",
    "chain_of_thought_withStance_pydantic",
    "chain_of_thought_noStance_pydantic",
    "cot_fewshot_withStance_pydantic",
    "cot_fewshot_noStance_pydantic"
    ]



    dataset = parser.parse_args().dataset_type

    


    if dataset == "factores_dataset":

        scorer = Evaluator(dataset_type="factores")
        labels = scorer.factores_verdicts
        #label_file = os.path.join(dataset, "dev.json")
        LABEL_MAP = {
            "supported": "S",
            "refuted": "R",
            "not enough evidence": "N"
        }


    else:
        scorer = Evaluator(dataset_type="averitec")
        labels = scorer.averitec_verdicts
        LABEL_MAP = {
            "supported": "S",
            "refuted": "R",
            "not enough evidence": "N",
            "conflicting evidence/cherrypicking": "C"
        }

    SHORT_LABELS = list(LABEL_MAP.values())
    FULL_LABELS = list(LABEL_MAP.keys())



    label_file = os.path.join(dataset, "dev.json")
    print("Label file: ", label_file)

    for mode in modes:
        #mode = "few_shot_withStance_pydantic"  # "chain_of_thought", "zero_shot", "few_shot", "zero_shot_withStance_pydantic", "few_shot_withStance_pydantic"
        input_dir = dataset + "/veracity/" + mode
        #label_file = os.path.join("averitec_dataset", "dev.json")
        output_dataset = dataset.replace("_dataset", "")
        output_dir = "src/veracity/ver_eval_results/" + output_dataset + "/" + mode
        os.makedirs(output_dir, exist_ok=True)


        for filename in os.listdir(input_dir):
            if filename.endswith("_veracity_predictions.json"):
                parts = filename.split("_")
                evidences = "_".join(parts[0:4])
                
                if "averitec" in filename:
                    model_name = "averitec"
                    #output_filename = f"{evidences}_result_{model_name}_veracity_evaluation__output.txt"
                    output_filename = filename.replace("_veracity_predictions.json", "_veracity_evaluation_output.txt")

                elif "factores" in filename:
                    model_name = "factores"
                    #output_filename = f"{evidences}_result_{model_name}_veracity_evaluation__output.txt"
                    output_filename = filename.replace("_veracity_predictions.json", "_veracity_evaluation_output.txt")
                
                else:
                    mode = parts[-4] + "_shot"
                    #model_name = " ".join(parts[5:-5])  # nombre del modelo
                    model_name = parts[5].replace("-", " ")  # nombre del modelo
                    #output_filename = f"{mode}_{evidences}_result_{model_name}_veracity_evaluation__output.txt"
                    output_filename = filename.replace("_veracity_predictions.json", "_veracity_evaluation_output.txt")

                output_path = os.path.join(output_dir, output_filename)
                print(output_filename)
                prediction_file = os.path.join(input_dir, filename)
                

                with open(prediction_file) as f:
                    predictions = json.load(f)
                with open(label_file) as f:
                    references = json.load(f)

                print("Amount of predictions: ", len(predictions))
                print("Amount of references: ", len(references))


            
                with open(output_path, "w", encoding="utf-8") as out:
                    def write_line(s): out.write(s + "\n")

                    write_line(f"------------------- Evaluation for {model_name} ({mode}) -------------------\n")
                    write_line("====================")

                    if dataset == "averitec_dataset":
                        write_line("AVeriTeC scoring method:")

                        q_score = scorer.evaluate_questions_only(predictions, references)
                        write_line(f"Question-only score (HU-{scorer.metric}):".ljust(50) + str(q_score))

                        p_score = scorer.evaluate_questions_and_answers(predictions, references)
                        write_line(f"Question-answer score (HU-{scorer.metric}):".ljust(50) + str(p_score))

                    v_score = scorer.evaluate_veracity(predictions, references)
                    write_line("\nVeracity F1 scores:")
                    for k, v in v_score.items():
                        write_line(f" * {k}:".ljust(50) + str(v))



                    if dataset == "averitec_dataset":

                        write_line("\n--------------------")
                        write_line("AVeriTeC scores:")
                        v_score = scorer.evaluate_averitec_score(predictions, references)
                        for i, level in enumerate(scorer.averitec_reporting_levels):
                            write_line(
                                f" * Veracity scores ({scorer.metric} @ {level}):".ljust(50) + str(v_score[i])
                            )

                        write_line("\n--------------------")
                        write_line("AVeriTeC scores by claim type @ 0.25:")
                        type_scores = scorer.evaluate_averitec_veracity_by_type(
                            predictions, references, threshold=0.25
                        )
                        for t, v in type_scores.items():
                            write_line(f" * Veracity scores ({t}):".ljust(50) + str(v))

                    # Distribución de clases
                    write_line("\n====================")
                    # Capturamos print() de evaluate_predictions_distribution
                    
                    backup_stdout = sys.stdout
                    sys.stdout = StringIO()
                    evaluate_predictions_distribution(prediction_file, label_file)
                    output = sys.stdout.getvalue()
                    sys.stdout = backup_stdout
                    write_line(output.strip())


                    base_filename = filename.replace("_veracity_predictions.json", "")
                    confusion_path = os.path.join(output_dir, base_filename + "_confusion_matrix.png")
                    metrics_path = os.path.join(output_dir, base_filename + "_class_metrics.png")
                    macro_metrics_path = os.path.join(output_dir, base_filename + "_macro_metrics.png")

                    if "llama" in model_name.lower():
                        model_name = model_name.replace("llama", "LLamA")
                    if "mistral" in model_name.lower():
                        model_name = model_name.replace("mistral", "Mistral")
                    if "gpt" in model_name.lower():
                        model_name = model_name.replace("gpt", "GPT")
                    if "qwen" in model_name.lower():
                        model_name = model_name.replace("qwen", "Qwen")
                    if "instruct" in model_name.lower():
                        model_name = model_name.replace("instruct", "Instruct")

                    if "withStance" in filename:
                        stance = "with Stance"
                    else:
                        stance = "without Stance"

                    # Generar y guardar
                    plot_confusion_matrix(predictions, references, labels, model_name, stance, confusion_path)
                    plot_class_metrics(predictions, references, labels, model_name, stance, metrics_path)
                    #plot_macro_metrics(predictions, references, labels, macro_metrics_path)




        print("\nEvaluations saved in ", output_dir)
