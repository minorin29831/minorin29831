import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pennylane as qml
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers

np.random.seed(42)

# データの準備
X, y = load_iris(return_X_y=True)

# ３つのうち２つのクラスのみ使用し、バイナリ分類問題に変換（各クラス50件ずつある）
X = X[:100]
y = y[:100]

# scaling
# StandardScalerは特徴量を標準化
# 平均を引いて標準偏差で割ることで平均０、標準偏差１に
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# 分類ラベルy（0or1）を−1と１にスケーリングして損失関数と整合するように
y_scaled = 2 * (y - 0.5)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled)

classical_model1 = SVC(kernel='linear')
classical_model2 = SVC(kernel='rbf', gamma='scale')
classical_model3 = SVC(kernel='poly', degree=3)
classical_model4 = SVC(kernel='sigmoid')

classical_model4.fit(X_train, y_train)

y_pred1 = classical_model4.predict(X_test)

accuracy1 = accuracy_score(y_test, y_pred1)
report1 = classification_report(y_test, y_pred1)

print("Accuracy1:", accuracy1)
print("\nClassification Report1:\n, report", report1)


n_qubits = len(X_train[0])
n_qubits

# 量子状態ベクトル進化をシミュレートするための高速な線形代数計算を行うカスタムバックエンドを搭載したデバイス
dev_kernel = qml.device("lightning.qubit", wires=n_qubits)

projector = np.zeros((2 ** n_qubits, 2 ** n_qubits))
projector[0, 0] = 1

# 量子カーネルの定義
# x1x2を量子状態として埋め込み、カーネルの計算を行い、出力は期待値として取得
@qml.qnode(dev_kernel)
def kernel(x1, x2):
    """The quantum kernel."""
    AngleEmbedding(x1, wires=range(n_qubits))
    qml.adjoint(AngleEmbedding)(x2, wires=range(n_qubits))
    return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))

kernel(X_train[0], X_train[0])

# カーネル行列の作成
def kernel_matrix(A, B):
    """Compute the matrix whose entries are the kernel
       evaluated on pairwise data from sets A and B."""
    return np.array([[kernel(a, b) for b in B] for a in A])

# SVMに渡して学習
svm = SVC(kernel=kernel_matrix).fit(X_train, y_train)

y_pred2 = svm.predict(X_test)

accuracy2 = accuracy_score(y_test, y_pred2)
report2 = classification_report(y_test, y_pred2)

print("Accuracy2:", accuracy2)
print("\nClassification Report2:\n, report", report2)