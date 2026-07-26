import os
import sys
import csv
import statistics

ON_KAGGLE = os.path.exists("/kaggle/working")

if ON_KAGGLE:
    OUTPUT_DIR = "/kaggle/working/output"
else:
    ROOT = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(ROOT, "output")

def run_fairness(input_csv):
    rows = list(csv.reader(open(input_csv)))
    header = rows[0]
    data_rows = rows[1:]
    category_index = header.index("category")
    score_index = header.index('hallucination_score')

    scores_by_category = {}
    for row in data_rows:
        category = row[category_index]
        score = float(row[score_index])
        if category not in scores_by_category:
            scores_by_category[category] = []
        scores_by_category[category].append(score)

    category_means = {}
    for category, scores in scores_by_category.items():
        category_means[category] = sum(scores) / len(scores)

    overall_scores = [float(row[score_index]) for row in data_rows]
    overall_mean = sum(overall_scores) / len(overall_scores)
    zero_count = len([s for s in overall_scores if s == 0])
    zero_pct = 100 * zero_count / len(overall_scores)
    fairness_std = statistics.pstdev(list(category_means.values()))

    print("overall mean hallucination score:", overall_mean)
    print("fairness std dev across categories:", fairness_std)
    for category, mean_score in category_means.items():
        print("category", category, "mean score", mean_score, "n", len(scores_by_category[category]))

    out_name = os.path.basename(input_csv).replace(".csv", "_fairness.csv")
    out_path = os.path.join(OUTPUT_DIR, out_name)
    f = open(out_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["category", "mean_hallucination_score", "n_questions"])
    for category, mean_score in category_means.items():
        writer.writerow([category, mean_score, len(scores_by_category[category])])
    writer.writerow(["OVERALL", overall_mean, len(overall_scores)])
    writer.writerow(["FAIRNESS_STD_DEV", fairness_std, ""])
    writer.writerow(["PCT_ZERO_HALLUCINATION", zero_pct, ""])
    f.close()
    print("saved", out_path)

run_fairness(sys.argv[1])
