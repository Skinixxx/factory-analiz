from sympy import Matrix, init_printing, symbols
from IPython.display import display, Math

# Включаем красивый вывод
init_printing(use_latex='mathjax')

# Данные из вашего запроса (D1 и D~1)
D1_data = [
    [0.00,  1.61,  0.69,  1.39,  1.79,  1.39],
    [-1.61, 0.00, -1.61, -1.10,  0.69, -1.10],
    [-0.69, 1.61,  0.00,  0.00, -1.39, -1.10],
    [-1.39, 1.10,  0.00,  0.00,  1.39,  0.69],
    [-1.79, -0.69, 1.39, -1.39,  0.00, -1.61],
    [-1.39, 1.10,  1.10, -0.69,  1.61,  0.00]
]

D_tilde_1_data = [
    [0.00,  1.61,  0.69,  1.39,  1.79,  1.39],
    [-1.61, 0.00, -0.92, -0.22,  0.18, -0.22],
    [-0.69, 0.92,  0.00,  0.69,  1.10,  0.69],
    [-1.39, 0.22, -0.69,  0.00,  0.41,  0.00],
    [-1.79, -0.18, -1.10, -0.41, 0.00, -0.41],
    [-1.39, 0.22, -0.69,  0.00,  0.41,  0.00]
]

# Преобразуем в SymPy-матрицы
D1 = Matrix(D1_data)
D_tilde_1 = Matrix(D_tilde_1_data)

# Функция для красивого вывода с меткой
def display_matrix_with_label(matrix, label):
    # Конвертируем матрицу в LaTeX-строку
    latex_str = matrix._repr_latex_()
    # Убираем \begin{equation}... если нужно — но display(Math) сам всё сделает
    display(Math(f"{label} = {latex_str}"))

# Выводим
display_matrix_with_label(D1, "D_1")
display_matrix_with_label(D_tilde_1, r"\tilde{D}_1")