import numpy as np 
from scipy import stats 
import math 
import pandas as pd 
import matplotlib.pyplot as plt

SEED = 3 
SIZE_100 = 100
SIZE_1000 = 1000 

# .rvs() — «random variates» — случайные величины. Генерирует случайные выборки 
# из указанного распределения вероятностей. 
# loc — нижняя граница распределения (минимальное значение, которое может принять СВ)
# scale — ширина интервала распределения. (В таком случае loc + scale = 15, это верх. граница распределения)
#
# Равн. распределение на интервале [5, 15]
uniform_100 = stats.uniform.rvs(loc=5, scale=10, size=SIZE_100, random_state=SEED)

# Распределение Бернулли, вероятность p = 0.7 
bernoulli_100 = stats.bernoulli.rvs(p=0.7, size=SIZE_100, random_state=SEED)

# Биноминальное распределение с n=20 (число испытаний), p=0.6 (вероятность успеха в каждом испытании) 
binominal_100 = stats.binom.rvs(n=20, p=0.6, size=SIZE_100, random_state=SEED)

# Нормальное распределение с параметрами mu=10, sigma=2, 
# где mu — матожидание, а sigma^2 — дисперсия. 
# Поскольку sigma здесь это также стандартное отклонение, то большая часть значений
# будет находится в пределах +-2стандартных отклоения от среднего. 
normal_100 = stats.norm.rvs(loc=10, scale=2, size=SIZE_100, random_state=SEED)

uniform_1000 = stats.uniform.rvs(loc=5, scale=10, size=SIZE_1000, random_state=SEED)
bernoulli_1000 = stats.bernoulli.rvs(p=0.7, size=SIZE_1000, random_state=SEED)
binominal_1000 = stats.binom.rvs(n=20, p=0.6, size=SIZE_1000, random_state=SEED)
normal_1000 = stats.norm.rvs(loc=10, scale=2, size=SIZE_1000, random_state=SEED)

def empirical_distribution_function(sample, x): 
	n = len(sample) 

	# Подсчитаем количество наблюдей <= x 
	count = sum(1 for val in sample if val <= x) 

	# значение ЭФР 
	edf_value = count / n 

	# Вычисляем эпсилон для 95%-го доверительного интервала 
  # Для 95% интервала alpha = 0.05, ln(2/0.05) ≈ ln(40)
	epsilon = np.sqrt(np.log(40) / (2 * n))

	# Верхняя граница доверительного интервала 
	lower_bound = max(0, edf_value - epsilon)

	# Нижняя граница доверительного интервала 
	upper_bound = min(1, edf_value + epsilon)

	return edf_value, lower_bound, upper_bound

def scipy_edf(sample, x):
	# Используем stats.percentileofscore для вычисления ЭФР
	# kind='weak' даёт процент значений <= x (т.е. F(x)), поэтому делим на 100
	edf_value = stats.percentileofscore(sample, x, kind='weak') / 100

	# Вычисляем эпсилон так же, как и ранее 
	n = len(sample)
	epsilon = np.sqrt(np.log(40) / (2 * n))
	
	lower_bound = max(0, edf_value - epsilon)
	upper_bound = min(1, edf_value + epsilon)
	
	return edf_value, lower_bound, upper_bound


def plot_distribution_functions(sample, title, cdf_func, x_range, edf): 
	"""
	sample — выборка
	title — название графика
	cdf_func — истинная функция распределения 
	x_range — пространство значений, в котором будет строится график 
	"""

	plt.figure(figsize=(10, 6))

	y_true = [cdf_func(x) for x in x_range]
	plt.plot(x_range, y_true, 'b-', label='Истинная фукнция распределения')

	y_empirical = []
	lower_bounds = [] 
	upper_bounds = []

	for x in x_range: 
		edf_value, lower_bound, upper_bound = edf(sample, x)
		y_empirical.append(edf_value)
		lower_bounds.append(lower_bound)
		upper_bounds.append(upper_bound)

	plt.plot(x_range, y_empirical, 'r-', label='Эмпирическая функция распределения')
	plt.plot(x_range, lower_bounds, 'g--', label='95% доверительный интервал')
	plt.plot(x_range, upper_bounds, 'g--')
	
	plt.title(title)
	plt.xlabel('x')
	plt.ylabel('F(x)')
	plt.grid(True)
	plt.legend()
	plt.tight_layout()

def create_all_plots():
    """
    Создаёт графики для всех распределений. 
    """

    # Блок графиков с использованием самописной EDF
		
    x_uniform = np.linspace(4, 16, 1000)
    plot_distribution_functions(
        uniform_100,
        f"Равномерное распределение (n={SIZE_100}) с помощью самописной EDF", 
        lambda x: stats.uniform.cdf(x, loc=5, scale=10), 
        x_uniform, 
        empirical_distribution_function
    )
    plt.savefig('uniform_100_custom.png')

    plot_distribution_functions(
        uniform_1000,
        f"Равномерное распределение (n={SIZE_1000}) с помощью самописной EDF", 
        lambda x: stats.uniform.cdf(x, loc=5, scale=10), 
        x_uniform, 
        empirical_distribution_function
    )
    plt.savefig('uniform_1000_custom.png')

    x_bernoulli = np.linspace(-0.5, 1.5, 3)
    plot_distribution_functions(
        bernoulli_100, 
        f"Распределение Бернулли (n={SIZE_100}, p=0.7) с помощью самописной EDF", 
        lambda x: stats.bernoulli.cdf(x, p=0.7), 
        x_bernoulli,
        empirical_distribution_function
    )
    plt.savefig('bernoulli_100_custom.png')

    plot_distribution_functions(
        bernoulli_1000, 
        f"Распределение Бернулли (n={SIZE_1000}, p=0.7) с помощью самописной EDF", 
        lambda x: stats.bernoulli.cdf(x, p=0.7), 
        x_bernoulli,
        empirical_distribution_function
    )
    plt.savefig('bernoulli_1000_custom.png')

    x_binom = np.arange(0, 22)
    plot_distribution_functions(
        binominal_100, 
        f"Биномиальное распределение (n={SIZE_100}, trials=20, p=0.6) с помощью самописной EDF",
        lambda x: stats.binom.cdf(x, n=20, p=0.6), 
        x_binom,
        empirical_distribution_function
    )
    plt.savefig('binom_100_custom.png')

    plot_distribution_functions(
        binominal_1000, 
        f"Биномиальное распределение (n={SIZE_1000}, trials=20, p=0.6) с помощью самописной EDF",
        lambda x: stats.binom.cdf(x, n=20, p=0.6), 
        x_binom,
        empirical_distribution_function
    )
    plt.savefig('binom_1000_custom.png')

    x_normal = np.linspace(4, 16, 1000)
    plot_distribution_functions(
        normal_100, 
        f"Нормальное распределение (n={SIZE_100}, μ=10, σ=2) с помощью самописной EDF",
        lambda x: stats.norm.cdf(x, loc=10, scale=2), 
        x_normal,
        empirical_distribution_function
    )
    plt.savefig('normal_100_custom.png')

    plot_distribution_functions(
        normal_1000, 
        f"Нормальное распределение (n={SIZE_1000}, μ=10, σ=2) с помощью самописной EDF",
        lambda x: stats.norm.cdf(x, loc=10, scale=2), 
        x_normal,
        empirical_distribution_function
    )
    plt.savefig('normal_1000_custom.png')

    # Использую встроенной SciPy EDF

    plot_distribution_functions(
        uniform_100,
        f"Равномерное распределение (n={SIZE_100}) с помощью SciPy EDF", 
        lambda x: stats.uniform.cdf(x, loc=5, scale=10), 
        x_uniform, 
        scipy_edf
    )
    plt.savefig('uniform_100_scipy.png')

    plot_distribution_functions(
        uniform_1000,
        f"Равномерное распределение (n={SIZE_1000}) с помощью SciPy EDF", 
        lambda x: stats.uniform.cdf(x, loc=5, scale=10), 
        x_uniform, 
        scipy_edf
    )
    plt.savefig('uniform_1000_scipy.png')

    plot_distribution_functions(
        bernoulli_100, 
        f"Распределение Бернулли (n={SIZE_100}, p=0.7) с помощью SciPy EDF", 
        lambda x: stats.bernoulli.cdf(x, p=0.7), 
        x_bernoulli,
        scipy_edf
    )
    plt.savefig('bernoulli_100_scipy.png')

    plot_distribution_functions(
        bernoulli_1000, 
        f"Распределение Бернулли (n={SIZE_1000}, p=0.7) с помощью SciPy EDF", 
        lambda x: stats.bernoulli.cdf(x, p=0.7), 
        x_bernoulli,
        scipy_edf
    )
    plt.savefig('bernoulli_1000_scipy.png')

    plot_distribution_functions(
        binominal_100, 
        f"Биномиальное распределение (n={SIZE_100}, trials=20, p=0.6) с помощью SciPy EDF",
        lambda x: stats.binom.cdf(x, n=20, p=0.6), 
        x_binom,
        scipy_edf
    )
    plt.savefig('binom_100_scipy.png')

    plot_distribution_functions(
        binominal_1000, 
        f"Биномиальное распределение (n={SIZE_1000}, trials=20, p=0.6) с помощью SciPy EDF",
        lambda x: stats.binom.cdf(x, n=20, p=0.6), 
        x_binom,
        scipy_edf
    )
    plt.savefig('binom_1000_scipy.png')

    plot_distribution_functions(
        normal_100, 
        f"Нормальное распределение (n={SIZE_100}, μ=10, σ=2) с помощью SciPy EDF",
        lambda x: stats.norm.cdf(x, loc=10, scale=2), 
        x_normal,
        scipy_edf
    )
    plt.savefig('normal_100_scipy.png')

    plot_distribution_functions(
        normal_1000, 
        f"Нормальное распределение (n={SIZE_1000}, μ=10, σ=2) с помощью SciPy EDF",
        lambda x: stats.norm.cdf(x, loc=10, scale=2), 
        x_normal,
        scipy_edf
    )
    plt.savefig('normal_1000_scipy.png')

create_all_plots()