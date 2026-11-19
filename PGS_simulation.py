import random
import numpy

def logistic(x):
    return 1 / (1 + numpy.exp(-x))

def create_genotype(n_genes: int, allele_frequencies: list, epistasis_influence: list, linear_influence: list, linkage_matrix=None) -> tuple:

    """Create one person with optional pairwise gene linkage."""

    if linkage_matrix is None:
        linkage_matrix = numpy.zeros((n_genes, n_genes), dtype=float)

    linkage_matrix = numpy.asarray(linkage_matrix, dtype=float)
    if linkage_matrix.shape != (n_genes, n_genes):
        raise ValueError("linkage_matrix must be an n_genes by n_genes matrix")
    if numpy.any(linkage_matrix < -1) or numpy.any(linkage_matrix > 1):
        raise ValueError("linkage_matrix values must be between -1 and 1")

    genes = []
    latent_factors = numpy.random.random(n_genes)

    for i in range(n_genes):
        allele_count = 0
        pair_influences = []

        for j in range(n_genes):
            if i != j:
                strength = linkage_matrix[i][j]
                if strength >= 0:
                    pair_influences.append(strength * (latent_factors[j] - 0.5))
                else:
                    pair_influences.append((-strength) * (0.5 - latent_factors[j]))

        gene_probability = allele_frequencies[i]
        if pair_influences:
            gene_probability += sum(pair_influences) / n_genes
        gene_probability = min(1.0, max(0.0, gene_probability))

        for _ in range(2):
            if random.random() < gene_probability:
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

def create_biobank(n_genes: int, allele_frequencies: list, epistasis_influence: list, linear_influence: list, n_people: int, linkage_matrix=None):
    if linkage_matrix is None:
        linkage_matrix = numpy.random.uniform(-1, 1, size=(n_genes, n_genes))
        linkage_matrix = (linkage_matrix + linkage_matrix.T) / 2
        numpy.fill_diagonal(linkage_matrix, 0.0)

    people = []
    for i in range(n_people):
        people.append(create_genotype(n_genes, allele_frequencies, epistasis_influence, linear_influence, linkage_matrix))

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

def run_simulation(n_genes, allele_frequencies, epistasis_influence, linear_influence, n_people):
    """Run the entire simulation and return the comparison of models."""
    linkage_matrix = numpy.random.uniform(-1, 1, size=(n_genes, n_genes))
    linkage_matrix = (linkage_matrix + linkage_matrix.T) / 2
    numpy.fill_diagonal(linkage_matrix, 0.0)

    biobank = create_biobank(n_genes, allele_frequencies, epistasis_influence, linear_influence, n_people, linkage_matrix)
    model_comparison = compare_models(biobank, n_genes, n_people)

    return model_comparison

if __name__ == "__main__":
    n_genes = int(input("Enter the number of genes: "))
    allele_frequencies = [random.random() for i in range(n_genes)]
    epistasis_influence = [[random.uniform(-1, 1) for j in range(n_genes)] for i in range(n_genes)]
    linear_influence = [random.uniform(-1, 1) for i in range(n_genes)]
    n_people = int(input("Enter the number of people in the biobank: "))

    print("Running simulation...")
    print(run_simulation(n_genes, allele_frequencies, epistasis_influence, linear_influence, n_people))
