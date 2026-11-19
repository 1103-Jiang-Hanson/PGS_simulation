import random
import numpy

def logistic(x):
    return 1 / (1 + numpy.exp(-x))

def create_genotype(n_genes: int, allele_frequencies: list, epistasis_influence: list, linear_influence: list, linkage_strength: float = 0.0) -> tuple:

    """Create one person.

    allele_frequencies and linear_influence are lists of length n_genes.
    epistasis_influence is an n_genes by n_genes matrix.
    linkage_strength ranges from -1 (strong negative linkage) through 0 (independent) to 1 (strong positive linkage).
    """

    if not -1.0 <= linkage_strength <= 1.0:
        raise ValueError("linkage_strength must be between -1 and 1")

    genes = []
    shared_linkage = random.random()

    for i in range(n_genes):
        allele_count = 0
        for j in range(2):
            if linkage_strength >= 0:
                linked_probability = (
                    (1 - linkage_strength) * allele_frequencies[i]
                    + linkage_strength * shared_linkage
                )
            else:
                anti_linkage = shared_linkage if i % 2 == 1 else 1 - shared_linkage
                linked_probability = (
                    (1 + linkage_strength) * allele_frequencies[i]
                    + (-linkage_strength) * anti_linkage
                )
            if random.random() < linked_probability:
                allele_count += 1

        genes.append(allele_count)

    linear_component = 0
    epistasis_component = 0

    for i in range(n_genes):
        linear_component += genes[i] * linear_influence[i]
        for j in range(n_genes):
            epistasis_component += genes[i] * genes[j] * epistasis_influence[i][j]

    risk = linear_component + epistasis_component
    condition = int(random.random() < logistic(risk))

    return (genes, condition)

def create_biobank(n_genes: int, allele_frequencies: list, epistasis_influence: list, linear_influence: list, n_people: int, linkage_strength: float = 0.0):
    people = []

    for i in range(n_people):
        people.append(create_genotype(n_genes, allele_frequencies, epistasis_influence, linear_influence, linkage_strength))

    return people

def make_basic_pgs_model(biobank, n_genes, n_people):
    """Compute the per-gene correlation between gene values and the condition."""
    if len(biobank) != n_people:
        raise ValueError("biobank length does not match n_people")

    genes = numpy.array([person[0] for person in biobank], dtype=float)
    conditions = numpy.array([person[1] for person in biobank], dtype=float)

    if genes.shape[1] != n_genes:
        raise ValueError("gene matrix width does not match n_genes")

    correlations = []
    for i in range(n_genes):
        gene_vector = genes[:, i]
        correlations.append(numpy.corrcoef(gene_vector, conditions)[0][1])

    return correlations

def make_advanced_pgs_model(biobank, n_genes, n_people):
    """Fit a linear regression model to predict condition from gene counts."""
    genes = numpy.array([person[0] for person in biobank], dtype=float)
    conditions = numpy.array([person[1] for person in biobank], dtype=float)

    if len(biobank) == 0:
        raise ValueError("biobank is empty")

    if genes.shape[1] != n_genes:
        raise ValueError("gene matrix width does not match n_genes")

    if len(biobank) != n_people:
        raise ValueError("biobank length does not match n_people")

    design_matrix = numpy.column_stack((numpy.ones(len(genes)), genes))
    coefficients, _, _, _ = numpy.linalg.lstsq(design_matrix, conditions, rcond=None)

    return (coefficients[1:], coefficients[0])  # Return gene coefficients and intercept


def run_simple_pgs_model(correlations, biobank, n_genes, n_people):
    """Return continuous simple-model risk scores using correlations plus mean-condition bias."""
    scores = []
    mean_condition = numpy.mean([person[1] for person in biobank]) if biobank else 0.0

    for i in range(n_people):
        predicted_risk = mean_condition + sum(correlations[j] * biobank[i][0][j] for j in range(n_genes))
        scores.append(predicted_risk)

    return scores

def run_advanced_pgs_model(coefficients, intercept, biobank, n_genes, n_people):
    """Return continuous advanced-model risk scores from linear regression."""
    scores = []
    for i in range(n_people):
        predicted_risk = intercept + sum(coefficients[j] * biobank[i][0][j] for j in range(n_genes))
        scores.append(predicted_risk)

    return scores

def evaluate_model(scores, biobank):
    """Calculate ROC AUC from continuous model scores and true conditions."""
    scores = numpy.asarray(scores, dtype=float)
    true_conditions = numpy.asarray([person[1] for person in biobank], dtype=int)

    positive_scores = scores[true_conditions == 1]
    negative_scores = scores[true_conditions == 0]

    if len(positive_scores) == 0 or len(negative_scores) == 0:
        return float("nan")

    auc = 0.0
    for positive_score in positive_scores:
        for negative_score in negative_scores:
            if positive_score > negative_score:
                auc += 1.0
            elif positive_score == negative_score:
                auc += 0.5

    return auc / (len(positive_scores) * len(negative_scores))

def compare_models(biobank, n_genes, n_people):
    """Compare the simple and advanced PGS models in a readable summary."""
    correlations = make_basic_pgs_model(biobank, n_genes, n_people)
    coefficients, intercept = make_advanced_pgs_model(biobank, n_genes, n_people)

    simple_predictions = run_simple_pgs_model(correlations, biobank, n_genes, n_people)
    advanced_predictions = run_advanced_pgs_model(coefficients, intercept, biobank, n_genes, n_people)

    simple_auc = evaluate_model(simple_predictions, biobank)
    advanced_auc = evaluate_model(advanced_predictions, biobank)

    return (
        "\n\n\n\n\nSimple model\n"
        f"  AUC: {simple_auc:.3f}\n\n"
        "Advanced model\n"
        f"  AUC: {advanced_auc:.3f}"
    )

def run_simulation(n_genes, allele_frequencies, epistasis_influence, linear_influence, n_people, linkage_strength=0.0):
    """Run the entire simulation and return the comparison of models."""
    biobank = create_biobank(n_genes, allele_frequencies, epistasis_influence, linear_influence, n_people, linkage_strength)
    model_comparison = compare_models(biobank, n_genes, n_people)

    return model_comparison

if __name__ == "__main__":
    n_genes = int(input("Enter the number of genes: "))
    allele_frequencies = [random.random() for i in range(n_genes)]
    epistasis_influence = [[random.uniform(-1, 1) for j in range(n_genes)] for i in range(n_genes)]
    linear_influence = [random.uniform(-1, 1) for i in range(n_genes)]
    n_people = int(input("Enter the number of people in the biobank: "))
    linkage_strength = float(input("Enter gene-linkage strength from -1 to 1: "))

    print("Running simulation...")
    print(run_simulation(n_genes, allele_frequencies, epistasis_influence, linear_influence, n_people, linkage_strength))
