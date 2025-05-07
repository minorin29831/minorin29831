import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, hinge_loss
import pennylane as qml
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers

np.random.seed(42)

# データの準備
X, y = load_iris(return_X_y=True)

# ３つのうち２つのクラスのみ使用し、バイナリ分類問題に変換
X = X[:100]
y = y[:100]

# scaling
# StandardScalerは特徴量を標準化
# 平均を引いて標準偏差で割ることで平均０、標準偏差１に
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

# 時間計測スタート
start_time1 = time.time()

classical_model = SVC(kernel='rbf', gamma='scale')

accuracy_list1 = []
loss_list1 = []

# 繰り返し上手くできない、
for i in range(1, 21):
    classical_model.fit(X_train, y_train)
    y_pred1 = classical_model.predict(X_test)

    accuracy1 = accuracy_score(y_test, y_pred1)
    accuracy_list1.append(accuracy1)

    loss1 = hinge_loss(y_test, classical_model.decision_function(X_test))
    loss_list1.append(loss1)

    X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0.3, random_state=i)

plt.figure(figsize=(12, 5))

# 時間計測終了
classical_time = time.time() - start_time1

# 正解率の変化
plt.subplot(1, 2, 1)
plt.plot(range(1, 21), accuracy_list1, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Accuracy per Iteration')

# 損失関数の変化
plt.subplot(1, 2, 2)
plt.plot(range(1, 21), loss_list1, marker='o', color='red')
plt.xlabel('Iteration')
plt.ylabel('Hinge Loss')
plt.title('Hinge Loss per Iteration')

plt.tight_layout()
plt.show()

print("Accuracy1:", accuracy1)
print("Time1:", classical_time)




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
    AngleEmbedding(x1, wires=range(n_qubits))
    qml.adjoint(AngleEmbedding)(x2, wires=range(n_qubits))
    return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))

kernel(X_train[0], X_train[0])

# カーネル行列の作成
def kernel_matrix(A, B):
    return np.array([[kernel(a, b) for b in B] for a in A])

# 時間計測スタート
start_time2 = time.time()

# SVMに渡して学習
q_model = SVC(kernel=kernel_matrix).fit(X_train, y_train)

accuracy_list2 = []
loss_list2 = []

for i in range(1, 21):
    q_model.fit(X_train, y_train)
    y_pred2 = q_model.predict(X_test)

    accuracy2 = accuracy_score(y_test, y_pred2)
    accuracy_list2.append(accuracy1)

    loss2 = hinge_loss(y_test, classical_model.decision_function(X_test))
    loss_list2.append(loss2)

    X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0.3, random_state=i)

plt.figure(figsize=(12, 5))

# 時間計測終了
q_time = time.time() - start_time1

print("Accuracy2:", accuracy2)
print("Time2:", q_time)