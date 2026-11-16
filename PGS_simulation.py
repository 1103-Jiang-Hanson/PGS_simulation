import random
import numpy

def logistic(x):
    return 1 / (1 + numpy.exp(-x))

def create_genotype(n_genes: int, allele_frequencies: list, epistasis_influence: list, linear_influence: list) -> tuple:

    '''allele_frequencies and linear_influence are int lists, length n_genes. epistasis_influence is an int 2x2 array, n_genes by n_genes.'''

    genes = []

    for i in range(n_genes):
        allele_count = 0
        for j in range(2):
            if random.random() < allele_frequencies[i]:
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

def create_biobank(n_genes: int, allele_frequencies: list, epistasis_influence: list, linear_influence: list, n_people: int):
    people = []

    for i in range(n_people):
        people.append(create_genotype(n_genes, allele_frequencies, epistasis_influence, linear_influence))

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
    """Run a simple PGS model using the correlations plus a mean-condition bias."""
    predictions = []
    mean_condition = numpy.mean([person[1] for person in biobank]) if biobank else 0.0

    for i in range(n_people):
        predicted_risk = mean_condition + sum(correlations[j] * biobank[i][0][j] for j in range(n_genes))
        predicted_condition = int(predicted_risk > 0.5)
        predictions.append(predicted_condition)

    return predictions

def run_advanced_pgs_model(coefficients, intercept, biobank, n_genes, n_people):
    """Run an advanced PGS model using the linear regression coefficients to predict conditions."""
    predictions = []
    for i in range(n_people):
        predicted_risk = intercept + sum(coefficients[j] * biobank[i][0][j] for j in range(n_genes))
        predicted_condition = int(predicted_risk > 0.5)  # Simple thresholding
        predictions.append(predicted_condition)

    return predictions

def evaluate_model(predictions, biobank):
    """Evaluate the model's predictions against the actual conditions in the biobank."""
    true_conditions = [person[1] for person in biobank]
    correct_predictions = sum(1 for pred, true in zip(predictions, true_conditions) if pred == true)
    accuracy = correct_predictions / len(true_conditions) if true_conditions else 0

    return accuracy

def compare_models(biobank, n_genes, n_people):
    """Compare the simple and advanced PGS models in a readable summary."""
    correlations = make_basic_pgs_model(biobank, n_genes, n_people)
    coefficients, intercept = make_advanced_pgs_model(biobank, n_genes, n_people)

    simple_predictions = run_simple_pgs_model(correlations, biobank, n_genes, n_people)
    advanced_predictions = run_advanced_pgs_model(coefficients, intercept, biobank, n_genes, n_people)

    simple_accuracy = evaluate_model(simple_predictions, biobank)
    advanced_accuracy = evaluate_model(advanced_predictions, biobank)

    return (
        "\n\n\n\n\nSimple model\n"
        f"  Accuracy: {simple_accuracy:.2%}\n\n"
        "Advanced model\n"
        f"  Accuracy: {advanced_accuracy:.2%}"
    )

def run_simulation(n_genes, allele_frequencies, epistasis_influence, linear_influence, n_people):
    """Run the entire simulation and return the comparison of models."""
    biobank = create_biobank(n_genes, allele_frequencies, epistasis_influence, linear_influence, n_people)
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
