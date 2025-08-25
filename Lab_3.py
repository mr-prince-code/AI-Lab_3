from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC


from sklearn.datasets import make_moons

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV



from sklearn.datasets import make_moons, load_iris

from mlxtend.plotting import plot_decision_regions

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx], marker=markers[idx], label=cl, edgecolor='black')

    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='none', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o', s=100, label='test set')
    plt.legend(loc='upper left')
    plt.show()
    
    
#Exercise 1


iris = load_iris()
X = iris.data[:, [2, 3]]  # Petal length and width
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

print("Class distribution:", np.bincount(y_train))


def prepare_iris_data():
    """Load and prepare the Iris dataset."""
    print("=" * 60)
    print("EXERCISE 1: IRIS DATASET PREPARATION")
    print("=" * 60)
    
    # Load Iris dataset
    iris = load_iris()
    X = iris.data[:, [2, 3]]  # Petal length and width for 2D visualization
    y = iris.target
    
    print("Dataset Information:")
    print(f"Feature names: {iris.feature_names[2:4]}")
    print(f"Target names: {iris.target_names}")
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Class distribution in training set: {np.bincount(y_train)}")
    print(f"Class distribution in test set: {np.bincount(y_test)}")
    
    # Standardize features
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    
    # Combine for visualization
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    
    print(f"\nStandardization applied:")
    print(f"Training mean: {np.mean(X_train_std, axis=0)}")
    print(f"Training std: {np.std(X_train_std, axis=0)}")
    
    return X_train_std, X_test_std, y_train, y_test, X_combined_std, y_combined

def prepare_moons_data():
    """Generate and prepare the make_moons dataset."""
    print("\n" + "=" * 60)
    print("EXERCISE 1: MAKE_MOONS DATASET PREPARATION")
    print("=" * 60)
    
    # Generate nonlinear dataset
    X, y = make_moons(n_samples=200, noise=0.2, random_state=1)
    
    print("Dataset Information:")
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Class distribution in training set: {np.bincount(y_train)}")
    print(f"Class distribution in test set: {np.bincount(y_test)}")
    
    # Standardize features
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    
    # Combine for visualization
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    
    print(f"\nStandardization applied:")
    print(f"Training mean: {np.mean(X_train_std, axis=0)}")
    print(f"Training std: {np.std(X_train_std, axis=0)}")
    
    return X_train_std, X_test_std, y_train, y_test, X_combined_std, y_combined

def compare_classifiers(X_train_std, X_test_std, y_train, y_test, X_combined_std, y_combined, dataset_name):
    """Compare different classifiers on the given dataset."""
    print(f"\n" + "=" * 60)
    print(f"CLASSIFIER COMPARISON ON {dataset_name.upper()}")
    print("=" * 60)
    
    # Define classifiers
    classifiers = {
        'Perceptron': Perceptron(max_iter=1000, random_state=1),
        'Logistic Regression': LogisticRegression(random_state=1, max_iter=1000),
        'SVM (Linear)': SVC(kernel='linear', random_state=1),
        'SVM (RBF)': SVC(kernel='rbf', random_state=1),
        'Decision Tree': DecisionTreeClassifier(random_state=1),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=1),
        'K-NN (k=5)': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    test_start_idx = len(y_train)
    test_indices = range(test_start_idx, len(y_combined))
    
    for name, classifier in classifiers.items():
        print(f"\nTraining {name}...")
        
        # Train classifier
        classifier.fit(X_train_std, y_train)
        
        # Make predictions
        y_train_pred = classifier.predict(X_train_std)
        y_test_pred = classifier.predict(X_test_std)
        
        # Calculate accuracy
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        results[name] = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'classifier': classifier
        }
        
        print(f"  Training Accuracy: {train_accuracy:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        
        # Plot decision regions
        plot_decision_regions(
            X_combined_std, y_combined, classifier, 
            test_idx=test_indices, 
            title=f'{name} - {dataset_name} Dataset'
        )
#Exercise 2


ppn = Perceptron(max_iter=40, eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print('Accuracy:', accuracy_score(y_test, y_pred))

plot_decision_regions(X_combined_std, y_combined, classifier=ppn, test_idx=range(len(X_train), len(X_combined_std)))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')

#Exercise 3


lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr')
lr.fit(X_train_std, y_train)

y_pred = lr.predict(X_test_std)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Probabilities:', lr.predict_proba(X_test_std[:3]))

plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(len(X_train), len(X_combined_std)))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')

#exercise 4



svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)

y_pred = svm.predict(X_test_std)
print('Accuracy:', accuracy_score(y_test, y_pred))

plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(len(X_train), len(X_combined_std)))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')


#Exercise 5

X_moons, y_moons = make_moons(n_samples=100, random_state=123)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_moons, y_moons, test_size=0.3, random_state=1)

sc_m = StandardScaler()
X_train_m_std = sc_m.fit_transform(X_train_m)
X_test_m_std = sc_m.transform(X_test_m)
X_combined_m_std = np.vstack((X_train_m_std, X_test_m_std))
y_combined_m = np.hstack((y_train_m, y_test_m))

svm_rbf = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)
svm_rbf.fit(X_train_m_std, y_train_m)

y_pred_m = svm_rbf.predict(X_test_m_std)
print('Accuracy:', accuracy_score(y_test_m, y_pred_m))

plot_decision_regions(X_combined_m_std, y_combined_m, classifier=svm_rbf, test_idx=range(len(X_train_m), len(X_combined_m_std)))
plt.xlabel('Feature 1 [standardized]')
plt.ylabel('Feature 2 [standardized]')


#Gamma


# Moons dataset analysis with different gamma values
X_moons, y_moons = make_moons(n_samples=100, random_state=123)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_moons, y_moons, test_size=0.3, random_state=1)

sc_m = StandardScaler()
X_train_m_std = sc_m.fit_transform(X_train_m)
X_test_m_std = sc_m.transform(X_test_m)
X_combined_m_std = np.vstack((X_train_m_std, X_test_m_std))
y_combined_m = np.hstack((y_train_m, y_test_m))

# Test different gamma values
gamma_values = [0.01, 1, 100]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, gamma in enumerate(gamma_values):
    svm_rbf = SVC(kernel='rbf', random_state=1, gamma=gamma, C=1.0)
    svm_rbf.fit(X_train_m_std, y_train_m)
    
    y_pred_m = svm_rbf.predict(X_test_m_std)
    train_acc = accuracy_score(y_train_m, svm_rbf.predict(X_train_m_std))
    test_acc = accuracy_score(y_test_m, y_pred_m)
    
    print(f'Moons dataset - Gamma: {gamma}')
    print(f'Training Accuracy: {train_acc:.3f}')
    print(f'Test Accuracy: {test_acc:.3f}')
    print('---')
    
    # Plot decision regions
    plot_decision_regions(X_combined_m_std, y_combined_m, classifier=svm_rbf, 
                         test_idx=range(len(X_train_m), len(X_combined_m_std)), ax=axes[i])
    axes[i].set_title(f'Gamma = {gamma}\nTrain: {train_acc:.3f}, Test: {test_acc:.3f}')
    axes[i].set_xlabel('Feature 1 [standardized]')
    axes[i].set_ylabel('Feature 2 [standardized]')

plt.tight_layout()
plt.show()

# Iris dataset comparison
print("\n" + "="*50)
print("IRIS DATASET COMPARISON")
print("="*50)

# Load and prepare Iris data
iris = load_iris()
X_iris = iris.data[:, [0, 2]]  # Use sepal length and petal length for 2D visualization
y_iris = iris.target

X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_iris, y_iris, test_size=0.3, random_state=1)

sc_i = StandardScaler()
X_train_i_std = sc_i.fit_transform(X_train_i)
X_test_i_std = sc_i.transform(X_test_i)
X_combined_i_std = np.vstack((X_train_i_std, X_test_i_std))
y_combined_i = np.hstack((y_train_i, y_test_i))

# Linear SVM on Iris
svm_linear = SVC(kernel='linear', random_state=1, C=1.0)
svm_linear.fit(X_train_i_std, y_train_i)

y_pred_linear = svm_linear.predict(X_test_i_std)
linear_train_acc = accuracy_score(y_train_i, svm_linear.predict(X_train_i_std))
linear_test_acc = accuracy_score(y_test_i, y_pred_linear)

print(f'Linear SVM on Iris:')
print(f'Training Accuracy: {linear_train_acc:.3f}')
print(f'Test Accuracy: {linear_test_acc:.3f}')
print('---')

# RBF SVM with different gamma values on Iris
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, gamma in enumerate(gamma_values):
    svm_rbf_iris = SVC(kernel='rbf', random_state=1, gamma=gamma, C=1.0)
    svm_rbf_iris.fit(X_train_i_std, y_train_i)
    
    y_pred_rbf = svm_rbf_iris.predict(X_test_i_std)
    train_acc = accuracy_score(y_train_i, svm_rbf_iris.predict(X_train_i_std))
    test_acc = accuracy_score(y_test_i, y_pred_rbf)
    
    print(f'RBF SVM on Iris - Gamma: {gamma}')
    print(f'Training Accuracy: {train_acc:.3f}')
    print(f'Test Accuracy: {test_acc:.3f}')
    print('---')
    
    # Plot decision regions
    plot_decision_regions(X_combined_i_std, y_combined_i, classifier=svm_rbf_iris, 
                         test_idx=range(len(X_train_i), len(X_combined_i_std)), ax=axes[i])
    axes[i].set_title(f'Gamma = {gamma}\nTrain: {train_acc:.3f}, Test: {test_acc:.3f}')
    axes[i].set_xlabel('Feature 1 [standardized]')
    axes[i].set_ylabel('Feature 2 [standardized]')

plt.tight_layout()
plt.show()

# Compare linear vs RBF performance
print("\nCOMPARISON SUMMARY:")
print(f"Linear SVM on Iris - Test Accuracy: {linear_test_acc:.3f}")
for i, gamma in enumerate(gamma_values):
    svm_rbf_iris = SVC(kernel='rbf', random_state=1, gamma=gamma, C=1.0)
    svm_rbf_iris.fit(X_train_i_std, y_train_i)
    test_acc = accuracy_score(y_test_i, svm_rbf_iris.predict(X_test_i_std))
    print(f"RBF SVM (gamma={gamma}) on Iris - Test Accuracy: {test_acc:.3f}")


#Exercise 6


tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree.fit(X_train, y_train)  # No scaling needed

y_pred = tree.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))

X_combined = np.vstack((X_train, X_test))
plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(len(X_train), len(X_combined)))
plt.xlabel('Petal length')
plt.ylabel('Petal width')

dot_data = export_graphviz(tree, filled=True, rounded=True, class_names=['Setosa', 'Versicolor', 'Virginica'],
                          feature_names=['petal length', 'petal width'], out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_png('tree.png')

#Exercise 7


forest = RandomForestClassifier(criterion='gini', n_estimators=100, random_state=1, n_jobs=2)
forest.fit(X_train, y_train)

y_pred = forest.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Feature Importances:', forest.feature_importances_)

plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(len(X_train), len(X_combined)))
plt.xlabel('Petal length')
plt.ylabel('Petal width')

# With OOB score
forest_oob = RandomForestClassifier(criterion='gini', n_estimators=100, random_state=1, 
                                   oob_score=True, n_jobs=2)
forest_oob.fit(X_train, y_train)
print(f'OOB Score: {forest_oob.oob_score_:.3f}')
#Exrercise 8


print("\n=== Exercise 8: K-Nearest Neighbors ===")
# Vary n_neighbors
for k in [1, 5, 10]:
    knn = KNeighborsClassifier(n_neighbors=k, p=2, metric='minkowski')
    knn.fit(X_train_std, y_train)
    y_pred = knn.predict(X_test_std)
    print(f'KNN (k={k}, Euclidean) Accuracy: {accuracy_score(y_test, y_pred):.3f}')

# Manhattan distance
knn_manhattan = KNeighborsClassifier(n_neighbors=5, p=1, metric='minkowski')
knn_manhattan.fit(X_train_std, y_train)
y_pred_man = knn_manhattan.predict(X_test_std)
print(f'KNN (k=5, Manhattan) Accuracy: {accuracy_score(y_test, y_pred_man):.3f}')
#Exercise 9


param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train_std, y_train)
print('Best params:', grid.best_params_)
print('Best score:', grid.best_score_)

models = {'Perceptron': ppn, 'LogReg': lr, 'SVM': svm, 'Tree': tree, 'Forest': forest, 'KNN': knn}
for name, model in models.items():
    y_pred = model.predict(X_test_std if name in ['Perceptron', 'LogReg', 'SVM', 'KNN'] else X_test)
    print(f'{name} Accuracy: {accuracy_score(y_test, y_pred)}')