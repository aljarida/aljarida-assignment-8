import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cdist
import os

matplotlib.use('Agg')

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

def generate_ellipsoid_clusters(distance, n_samples=100, cluster_std=0.5):
    np.random.seed(0)
    std_mod = 0.8
    covariance_matrix = np.array([[cluster_std, cluster_std * std_mod], [cluster_std * std_mod, cluster_std]])
    gen_samples = lambda: np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)

    X1, X2 = gen_samples(), gen_samples()
    X2 += np.array([distance, distance])
    y1, y2 = np.zeros(n_samples), np.ones(n_samples)

    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))
    return X, y

def fit_logistic_regression(X, y):
    assert(X.shape[1] == 2)
    model = LogisticRegression()
    model.fit(X, y)
    beta0 = model.intercept_[0]
    beta1, beta2 = model.coef_[0]
    return model, beta0, beta1, beta2

def calculate_logistic_loss(model, X, y_actual):
    y_pred = model.predict_proba(X)[:, 1]
    n = y_pred.shape[0]
    return - (1 / n) * np.sum(y_actual * np.log(y_pred) + (1 - y_actual) * np.log(1 - y_pred))

def calculate_slope(beta_1: float, beta_2: float) -> float:
    return - beta_1 / beta_2

def calculate_intercept(beta_0: float, beta_2: float) -> float:
    return - beta_0 / beta_2

def get_data(distance):
    X, y = generate_ellipsoid_clusters(distance=distance)
    m, b0, b1, b2 = fit_logistic_regression(X, y)
    s = calculate_slope(b1, b2)
    i = calculate_intercept(b0, b2)
    log_loss = calculate_logistic_loss(m, X, y)
    return {
            "beta_0": b0,
            "beta_1": b1,
            "beta_2": b2,
            "slope": s,
            "intercept": i,
            "log_loss": log_loss,
            "X": X,
            "y": y,
            "model": m,
            "margin_widths": []
           }


def generate_data_on_graph(axis, X, y) -> None:
    X0, X1 = X[:, 0], X[:, 1]
    sort = np.argsort(X0)
    colors = ['blue', 'red']
    cmap = clrs.ListedColormap(colors)
    axis.scatter(X0[sort], X1[sort], c=y[sort], cmap=cmap, alpha=0.6) 
    return None

def generate_contours_on_graph_and_margin_widths(distance, axis, xx0, xx1, Z):
    contour_levels = [0.7, 0.8, 0.9]
    alphas = [0.05, 0.1, 0.15]

    margin_widths = []
    for level, alpha in zip(contour_levels, alphas):
        c0_contour = axis.contourf(xx0, xx1, Z, levels=[0.0, 1 - level], colors=['blue'], alpha=alpha)
        c1_contour = axis.contourf(xx0, xx1, Z, levels=[level, 1.0], colors=['red'], alpha=alpha) 
        if level == 0.7:
            distances = cdist(c1_contour.collections[0].get_paths()[0].vertices,
                              c0_contour.collections[0].get_paths()[0].vertices,
                              metric='euclidean')
            min_distance = np.min(distances)
            margin_widths.append(min_distance)

    axis.set_title(f"Shift Distance = {round(distance, 4)}", fontsize=24)
    axis.set_xlabel("x1")
    axis.set_ylabel("x2")
    return margin_widths

def plot_equation_and_margin_width(axis, dd_i, X) -> None:
    X0_min, _, X1_min, X1_max = get_X0_X1_mins_maxs(X) 
    X1_range = X1_max - X1_min
    beta_0, beta_1, beta_2 = dd_i["beta_0"], dd_i["beta_1"], dd_i["beta_2"]
    intercept, slope = dd_i["intercept"], dd_i["slope"]
    min_distance = min(dd_i["margin_widths"])

    eq_text_1 = f"{beta_0:.2f} + {beta_1:.2f} * x1 + {beta_2:.2f} * x2 = 0\n"
    eq_text_2 = f"x2 = {slope:.2f} * x1 + {intercept:.2f}"
    equation_text = eq_text_1 + eq_text_2

    margin_text = f"Margin Width: {min_distance:.2f}"
    axis.text(X0_min + 0.3, X1_max - (0.2 * X1_range), equation_text,
             fontsize=12, color="black", ha='left',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
             )
    axis.text(X0_min + 0.3, X1_max - (0.4 * X1_range), margin_text,
             fontsize=12, color="black", ha='left',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
             )
    return None

def generate_decision_boundary(axis, dd_i, X) -> None:
    X0_min, X0_max, X1_min, X1_max = get_X0_X1_mins_maxs(X)
    x0 = np.linspace(X0_min, X0_max, 100)
    boundary = (dd_i["slope"] * x0) + dd_i["intercept"]
    mask = (boundary > X1_min) & (boundary < X1_max)
    x0_clipped, boundary_clipped = x0[mask], boundary[mask]
    axis.plot(x0_clipped, boundary_clipped, color='purple')
    return None

def get_X0_X1_mins_maxs(X):
    X0, X1 = X[:, 0], X[:, 1]
    X0_min, X0_max = X0.min(), X0.max()
    X1_min, X1_max = X1.min(), X1.max()
    return X0_min, X0_max, X1_min, X1_max

def get_xx0_xx1(X):
    X0_min, X0_max, X1_min, X1_max = get_X0_X1_mins_maxs(X)
    xx0, xx1 = np.meshgrid(np.linspace(X0_min, X0_max, 200), np.linspace(X1_min, X1_max, 200))
    return xx0, xx1

def calculate_probabilities_across_grid(X, model):
    xx0, xx1 = get_xx0_xx1(X)
    Z = model.predict_proba(np.c_[xx0.ravel(), xx1.ravel()])[:, 1]
    Z = Z.reshape(xx0.shape)
    return Z

def do_experiments(start, end, step_num):
    shift_distances = np.linspace(start, end, step_num)
    _, axes = plt.subplots(nrows=(step_num + 1) // 2, ncols=2, figsize=(20, 15))
    axes = axes.flatten()

    data_dictionary = {}
    for i, distance in enumerate(shift_distances):
        axis = axes[i]
        # Record / obtain necessary information  -- DONE
        data_dictionary[i] = get_data(distance)
        dd_i = data_dictionary[i]
        X, y = dd_i["X"], dd_i["y"]
        model = dd_i["model"]

        # Plot the data -- DONE
        generate_data_on_graph(axis, X, y)

        # Plot the decision boundary -- DONE
        generate_decision_boundary(axis, dd_i, X)

        # Plot fading red and blue contours for confidence levels -- DONE
        Z = calculate_probabilities_across_grid(X, model)
        xx0, xx1 = get_xx0_xx1(X)
        margin_widths = generate_contours_on_graph_and_margin_widths(distance, axis, xx0, xx1, Z)
        dd_i["margin_widths"].extend(margin_widths)

        # Plot contour decision boundary equation and margin width -- DONE
        plot_equation_and_margin_width(axis, dd_i, X)

        print(f"Axis {i} plotted.")

    plt.tight_layout(pad=2.0)
    plt.savefig(f"{result_dir}/dataset_plots.png")
    plt.close()

    _, axes = plt.subplots(4, 2, figsize=(24, 12))
    axes = axes.flatten()

    get_list = lambda x: [dd_i[x] for dd_i in data_dictionary.values()]

    axes[0].plot(shift_distances, get_list("beta_0"), marker="o", label="Beta 0")
    axes[1].plot(shift_distances, get_list("beta_1"), marker="o", label="Beta 1")
    axes[2].plot(shift_distances, get_list("beta_2"), marker="o", label="Beta 2")
    axes[3].plot(shift_distances, get_list("slope"), marker="o", label="Slope")
    axes[4].plot(shift_distances, get_list("intercept"), marker="o", label="Intercept")
    axes[5].plot(shift_distances, get_list("log_loss"), marker="o", label="Logistic Loss")
    
    widths = np.array(get_list("margin_widths")).reshape(-1, 1).tolist()
    axes[6].plot(shift_distances, widths, marker="o", label="Margin Width")

    for axis in axes:
        axis.legend()
        axis.grid(True)

    plt.tight_layout()
    plt.savefig(f"{result_dir}/parameters_vs_shift_distance.png")
    plt.close()

    print(f"Shift-distance graphs plotted and saved to {result_dir}")

if __name__ == "__main__":
    do_experiments(start=0.25, end=2.0, step_num=8)
