import numpy as np
import matplotlib.pyplot as plt


def generate_multivariate_normal_sample(
    number_of_observations,
    mean_vector,
    covariance_matrix,
    random_seed=42
):
    """
    Generuje próbę z wielowymiarowego rozkładu normalnego.

    Parametry
    ----------
    number_of_observations:
        Liczba generowanych obserwacji.

    mean_vector:
        Wektor wartości oczekiwanych.

    covariance_matrix:
        Macierz kowariancji całej populacji.

    random_seed:
        Ziarno generatora liczb losowych.

    Zwraca
    -------
    Dwuwymiarową tablicę NumPy o wymiarach:
    liczba_obserwacji x liczba_zmiennych.
    """

    random_generator = np.random.default_rng(random_seed)

    sample = random_generator.multivariate_normal(
        mean=mean_vector,
        cov=covariance_matrix,
        size=number_of_observations
    )

    return sample


def split_sample_by_quantiles(sample, q, benchmark_index=0):
    """
    Dzieli próbę na trzy grupy na podstawie kwantyli zmiennej referencyjnej.

    Dla wartości q otrzymujemy grupy:

    dolna:
        dolne 100*q procent obserwacji,

    środkowa:
        środkowe 100*(1 - 2*q) procent obserwacji,

    górna:
        górne 100*q procent obserwacji.

    Dla q = 0.2 otrzymujemy podział około:

        20 procent / 60 procent / 20 procent.

    Parametry
    ----------
    sample:
        Macierz obserwacji.

    q:
        Udział populacji w każdej grupie ogonowej.
        Musi należeć do przedziału (0, 0.5).

    benchmark_index:
        Numer kolumny używanej jako zmienna referencyjna.
        W artykule jest to pierwsza współrzędna X_1,
        dlatego domyślnie benchmark_index = 0.

    Zwraca
    -------
    lower_group:
        Obserwacje z dolnej części rozkładu benchmarku.

    central_group:
        Obserwacje ze środkowej części rozkładu benchmarku.

    upper_group:
        Obserwacje z górnej części rozkładu benchmarku.

    lower_threshold:
        Empiryczny kwantyl rzędu q.

    upper_threshold:
        Empiryczny kwantyl rzędu 1-q.
    """

    if not 0.0 < q < 0.5:
        raise ValueError("Parametr q musi należeć do przedziału (0, 0.5).")

    benchmark = sample[:, benchmark_index]

    lower_threshold = np.quantile(benchmark, q)
    upper_threshold = np.quantile(benchmark, 1.0 - q)

    lower_mask = benchmark <= lower_threshold

    central_mask = (
        (benchmark > lower_threshold)
        & (benchmark < upper_threshold)
    )

    upper_mask = benchmark >= upper_threshold

    lower_group = sample[lower_mask]
    central_group = sample[central_mask]
    upper_group = sample[upper_mask]

    return (
        lower_group,
        central_group,
        upper_group,
        lower_threshold,
        upper_threshold
    )


def calculate_covariance_matrix(group):
    """
    Oblicza empiryczną macierz kowariancji w wybranej grupie.

    Parametr rowvar=False oznacza, że:

    - wiersze odpowiadają obserwacjom,
    - kolumny odpowiadają zmiennym.
    """

    if group.shape[0] < 2:
        raise ValueError(
            "Grupa musi zawierać co najmniej dwie obserwacje."
        )

    covariance_matrix = np.cov(
        group,
        rowvar=False,
        ddof=1
    )

    return covariance_matrix


def covariance_to_correlation(covariance_matrix):
    """
    Przekształca macierz kowariancji w macierz korelacji.

    Element macierzy korelacji ma postać:

        R_ij = Sigma_ij / sqrt(Sigma_ii * Sigma_jj)

    gdzie:

        Sigma_ij oznacza kowariancję zmiennych i oraz j,
        Sigma_ii i Sigma_jj oznaczają ich wariancje.
    """

    variances = np.diag(covariance_matrix)

    if np.any(variances <= 0.0):
        raise ValueError(
            "Wszystkie wariancje muszą być dodatnie."
        )

    standard_deviations = np.sqrt(variances)

    denominator = np.outer(
        standard_deviations,
        standard_deviations
    )

    correlation_matrix = covariance_matrix / denominator

    correlation_matrix = np.clip(
        correlation_matrix,
        -1.0,
        1.0
    )

    return correlation_matrix


def calculate_conditional_matrices(sample, q, benchmark_index=0):
    """
    Dla ustalonego q:

    1. dzieli próbę na trzy grupy,
    2. oblicza trzy warunkowe macierze kowariancji,
    3. oblicza trzy warunkowe macierze korelacji.

    Zwraca wyniki w słowniku.
    """

    (
        lower_group,
        central_group,
        upper_group,
        lower_threshold,
        upper_threshold
    ) = split_sample_by_quantiles(
        sample=sample,
        q=q,
        benchmark_index=benchmark_index
    )

    lower_covariance = calculate_covariance_matrix(
        lower_group
    )

    central_covariance = calculate_covariance_matrix(
        central_group
    )

    upper_covariance = calculate_covariance_matrix(
        upper_group
    )

    lower_correlation = covariance_to_correlation(
        lower_covariance
    )

    central_correlation = covariance_to_correlation(
        central_covariance
    )

    upper_correlation = covariance_to_correlation(
        upper_covariance
    )

    results = {
        "q": q,
        "lower_threshold": lower_threshold,
        "upper_threshold": upper_threshold,
        "lower_group_size": lower_group.shape[0],
        "central_group_size": central_group.shape[0],
        "upper_group_size": upper_group.shape[0],
        "lower_covariance": lower_covariance,
        "central_covariance": central_covariance,
        "upper_covariance": upper_covariance,
        "lower_correlation": lower_correlation,
        "central_correlation": central_correlation,
        "upper_correlation": upper_correlation
    }

    return results


def frobenius_distance(matrix_a, matrix_b):
    """
    Oblicza normę Frobeniusa różnicy dwóch macierzy.

    Norma Frobeniusa jest odpowiednikiem długości wektora
    dla macierzy:

        ||A||_F = sqrt(sum_ij A_ij^2)

    Im mniejsza wartość, tym bardziej podobne są macierze.
    """

    difference = matrix_a - matrix_b

    distance = np.linalg.norm(
        difference,
        ord="fro"
    )

    return distance


def calculate_balance_error(
    sample,
    q,
    benchmark_index=0,
    matrix_type="correlation"
):
    """
    Oblicza funkcję błędu równowagi dla ustalonego q.

    Porównujemy:

    - grupę dolną z grupą środkową,
    - grupę górną z grupą środkową.

    Funkcja błędu ma postać:

        d(q)
        =
        ||M_lower - M_central||_F
        +
        ||M_upper - M_central||_F

    gdzie M może oznaczać:

    - macierz korelacji,
    - macierz kowariancji.

    Parametr matrix_type może przyjmować wartości:

        "correlation"
        "covariance"
    """

    results = calculate_conditional_matrices(
        sample=sample,
        q=q,
        benchmark_index=benchmark_index
    )

    if matrix_type == "correlation":
        lower_matrix = results["lower_correlation"]
        central_matrix = results["central_correlation"]
        upper_matrix = results["upper_correlation"]

    elif matrix_type == "covariance":
        lower_matrix = results["lower_covariance"]
        central_matrix = results["central_covariance"]
        upper_matrix = results["upper_covariance"]

    else:
        raise ValueError(
            "matrix_type musi być równy "
            "'correlation' albo 'covariance'."
        )

    lower_central_distance = frobenius_distance(
        lower_matrix,
        central_matrix
    )

    upper_central_distance = frobenius_distance(
        upper_matrix,
        central_matrix
    )

    total_error = (
        lower_central_distance
        + upper_central_distance
    )

    return total_error


def find_optimal_q(
    sample,
    q_grid,
    benchmark_index=0,
    matrix_type="correlation"
):
    """
    Przeszukuje zadaną siatkę wartości q.

    Dla każdego q oblicza błąd równowagi i wybiera q,
    dla którego błąd jest najmniejszy.

    Jest to numeryczna realizacja:

        q_hat = arg min_q d(q)
    """

    errors = []

    for q in q_grid:
        error = calculate_balance_error(
            sample=sample,
            q=q,
            benchmark_index=benchmark_index,
            matrix_type=matrix_type
        )

        errors.append(error)

    errors = np.asarray(errors)

    optimal_index = np.argmin(errors)

    optimal_q = q_grid[optimal_index]

    optimal_error = errors[optimal_index]

    return optimal_q, optimal_error, errors


def print_matrix(name, matrix):
    """
    Wypisuje macierz z czytelnym formatowaniem.
    """

    print()
    print(name)
    print("-" * len(name))

    print(
        np.array2string(
            matrix,
            precision=4,
            suppress_small=True
        )
    )


def plot_error_function(q_grid, errors, optimal_q):
    """
    Rysuje wartość funkcji błędu w zależności od q.

    Minimum wykresu odpowiada optymalnemu podziałowi:

        q / (1 - 2q) / q.
    """

    plt.figure(figsize=(8, 5))

    plt.plot(
        q_grid,
        errors,
        linewidth=2
    )

    plt.axvline(
        optimal_q,
        linestyle="--",
        label=f"Optymalne q = {optimal_q:.4f}"
    )

    plt.axvline(
        0.198089616,
        linestyle=":",
        label="Teoretyczne q ≈ 0.1981"
    )

    plt.xlabel("q")
    plt.ylabel("Błąd równowagi")
    plt.title(
        "Porównanie warunkowych macierzy korelacji"
    )

    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(matrix, title):
    """
    Rysuje pojedynczą macierz korelacji jako mapę wartości.

    Zakres osi kolorów jest ustawiony od -1 do 1,
    ponieważ korelacje zawsze należą do tego przedziału.
    """

    plt.figure(figsize=(6, 5))

    image = plt.imshow(
        matrix,
        vmin=-1.0,
        vmax=1.0
    )

    plt.colorbar(image)

    number_of_variables = matrix.shape[0]

    plt.xticks(
        range(number_of_variables),
        [
            f"X{i + 1}"
            for i in range(number_of_variables)
        ]
    )

    plt.yticks(
        range(number_of_variables),
        [
            f"X{i + 1}"
            for i in range(number_of_variables)
        ]
    )

    for row_index in range(number_of_variables):
        for column_index in range(number_of_variables):
            plt.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.2f}",
                ha="center",
                va="center"
            )

    plt.title(title)
    plt.tight_layout()
    plt.show()


def main():
    """
    Główna część programu.
    """

    number_of_observations = 300_000

    mean_vector = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0
        ]
    )

    covariance_matrix = np.array(
        [
            [1.0, 0.8, 0.5, 0.3],
            [0.8, 1.0, 0.4, 0.2],
            [0.5, 0.4, 1.0, 0.6],
            [0.3, 0.2, 0.6, 1.0]
        ]
    )

    sample = generate_multivariate_normal_sample(
        number_of_observations=number_of_observations,
        mean_vector=mean_vector,
        covariance_matrix=covariance_matrix,
        random_seed=42
    )

    print("Rozmiar wygenerowanej próby:")
    print(sample.shape)

    full_covariance = calculate_covariance_matrix(
        sample
    )

    full_correlation = covariance_to_correlation(
        full_covariance
    )

    print_matrix(
        "Macierz kowariancji całej populacji",
        full_covariance
    )

    print_matrix(
        "Macierz korelacji całej populacji",
        full_correlation
    )

    theoretical_q = 0.198089616

    theoretical_results = calculate_conditional_matrices(
        sample=sample,
        q=theoretical_q,
        benchmark_index=0
    )

    print()
    print("Podział dla teoretycznego q")
    print("---------------------------")

    print(
        f"q = {theoretical_q:.6f}"
    )

    print(
        "Udziały grup:"
    )

    print(
        f"dolna: {100 * theoretical_q:.2f}%"
    )

    print(
        f"środkowa: "
        f"{100 * (1 - 2 * theoretical_q):.2f}%"
    )

    print(
        f"górna: {100 * theoretical_q:.2f}%"
    )

    print()
    print("Liczebności grup:")

    print(
        "dolna:",
        theoretical_results["lower_group_size"]
    )

    print(
        "środkowa:",
        theoretical_results["central_group_size"]
    )

    print(
        "górna:",
        theoretical_results["upper_group_size"]
    )

    print()
    print("Progi dla zmiennej referencyjnej X1:")

    print(
        "dolny próg:",
        theoretical_results["lower_threshold"]
    )

    print(
        "górny próg:",
        theoretical_results["upper_threshold"]
    )

    print_matrix(
        "Warunkowa macierz korelacji: dolna grupa",
        theoretical_results["lower_correlation"]
    )

    print_matrix(
        "Warunkowa macierz korelacji: środkowa grupa",
        theoretical_results["central_correlation"]
    )

    print_matrix(
        "Warunkowa macierz korelacji: górna grupa",
        theoretical_results["upper_correlation"]
    )

    print_matrix(
        "Warunkowa macierz kowariancji: dolna grupa",
        theoretical_results["lower_covariance"]
    )

    print_matrix(
        "Warunkowa macierz kowariancji: środkowa grupa",
        theoretical_results["central_covariance"]
    )

    print_matrix(
        "Warunkowa macierz kowariancji: górna grupa",
        theoretical_results["upper_covariance"]
    )

    q_grid = np.linspace(
        0.05,
        0.45,
        161
    )

    optimal_q_correlation, minimum_correlation_error, correlation_errors = (
        find_optimal_q(
            sample=sample,
            q_grid=q_grid,
            benchmark_index=0,
            matrix_type="correlation"
        )
    )

    optimal_q_covariance, minimum_covariance_error, covariance_errors = (
        find_optimal_q(
            sample=sample,
            q_grid=q_grid,
            benchmark_index=0,
            matrix_type="covariance"
        )
    )

    print()
    print("Wyniki optymalizacji")
    print("--------------------")

    print(
        "Optymalne q dla macierzy korelacji:",
        f"{optimal_q_correlation:.4f}"
    )

    print(
        "Odpowiadający podział:",
        f"{100 * optimal_q_correlation:.2f}% / "
        f"{100 * (1 - 2 * optimal_q_correlation):.2f}% / "
        f"{100 * optimal_q_correlation:.2f}%"
    )

    print(
        "Minimalny błąd korelacji:",
        f"{minimum_correlation_error:.6f}"
    )

    print()

    print(
        "Optymalne q dla macierzy kowariancji:",
        f"{optimal_q_covariance:.4f}"
    )

    print(
        "Odpowiadający podział:",
        f"{100 * optimal_q_covariance:.2f}% / "
        f"{100 * (1 - 2 * optimal_q_covariance):.2f}% / "
        f"{100 * optimal_q_covariance:.2f}%"
    )

    print(
        "Minimalny błąd kowariancji:",
        f"{minimum_covariance_error:.6f}"
    )

    optimal_results = calculate_conditional_matrices(
        sample=sample,
        q=optimal_q_correlation,
        benchmark_index=0
    )

    plot_error_function(
        q_grid=q_grid,
        errors=correlation_errors,
        optimal_q=optimal_q_correlation
    )

    plot_correlation_matrix(
        optimal_results["lower_correlation"],
        "Warunkowa macierz korelacji — dolna grupa"
    )

    plot_correlation_matrix(
        optimal_results["central_correlation"],
        "Warunkowa macierz korelacji — środkowa grupa"
    )

    plot_correlation_matrix(
        optimal_results["upper_correlation"],
        "Warunkowa macierz korelacji — górna grupa"
    )


if __name__ == "__main__":
    main()