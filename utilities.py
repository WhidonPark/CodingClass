import numpy as np
import matplotlib.pyplot as plt

def visualize_classifier(classifier, X, y):
    plt.figure()
    
    # plt.colorbar(label='class')
    
    xx, yy = np.meshgrid(np.arange(0, 10, 0.1),
                         np.arange(0, 10, 0.1))
    
    # print(xx.shape)
    grid = np.c_[xx.ravel(), yy.ravel()]  # 10000,2 의 형태로 변경  x,y 좌표 주고 물어보려고
    # print(f"grid[:10]:{grid[:200]}")
    # print(f"grid.shape:{grid.shape}")
    
    Z = classifier.predict(grid) 
    print(f"Z.shape : {Z.shape}")  # (10000,) 형태로 나와야함
    # print(Z[1000:2000])  # 처음 10개만 확인
    Z = Z.reshape(xx.shape)  # (100,100)
    print(f"reshaped Z.shape : {Z.shape}")
    
    
    plt.pcolormesh(xx, yy, Z, cmap='viridis', alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=100, edgecolors='black')
    plt.title("Logistic Regression Classifier (C=1)")
    plt.ylabel("y")
    
    # plt.xlim(0,10)
    # plt.legend()
    plt.grid()
    # plt.savefig('파일명.png')
    
    plt.show()
