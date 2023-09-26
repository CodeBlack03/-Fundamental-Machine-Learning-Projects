import numpy as np
from tqdm import tqdm, trange


class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components = None
    
    def fit(self, X) -> None:
        X = X-np.mean(X,axis=0)
        covariance_matrix = np.cov(X.T)
        eig_val,eig_vec = np.linalg.eig(covariance_matrix)
        indices = eig_val.argsort()[::-1]
        eig_vec = eig_vec[:,indices]
        self.components = eig_vec[:,:self.n_components]
    
    def transform(self, X) -> np.ndarray:
        return np.dot(X,self.components)

    def fit_transform(self, X) -> np.ndarray:
        # fit the model and transform the data
        self.fit(X)
        return self.transform(X)


class SupportVectorModel:
    def __init__(self) -> None:
        self.w = None
        self.b = None
    
    def _initialize(self, X) -> None:
        self.w=np.random.uniform(0,1,size=len(X[0]))*0.01
        self.b=0.1

    def fit(
            self, X, y, 
            learning_rate: float,
            num_iters: int,
            C: float = 1.0,
    ) -> None:
        self._initialize(X)
        loss_per_epoch=[]
        errors=[]
        index_list = list(range(0,len(X)))
        # fit the SVM model using stochastic gradient descent
        for epoch in range(1, num_iters + 1):
            error = 0
            loss=0
            np.random.shuffle(index_list)
            
            for i in index_list:
                
                if (y[i]*(np.dot(X[i], self.w)+self.b)) < 1:
                    #misclassified update for ours weights
                    self.w = self.w + learning_rate * ( C*(X[i] * y[i])+ (-1*(self.w)) )
                    self.b = self.b + learning_rate*(C*y[i])
                else:
                    #correct classification, update our weights
                    self.w = self.w - learning_rate * (self.w)
                    self.b = self.b
                
    
    def predict(self, X,y) -> np.ndarray:
        # make predictions for the given data
        ypred=np.zeros(len(X))
        for i in range(len(X)):
            ypred[i]=np.dot(self.w,X[i])
        return ypred+self.b

    def accuracy_score(self, X, y) -> float:
        # compute the accuracy of the model (for debugging purposes)
        return np.mean((self.predict(X,y)) == (y))


class MultiClassSVM:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.models = []
        for i in range(self.num_classes):
            self.models.append(SupportVectorModel())
    
    def fit(self, X, y, **kwargs) -> None:
        # first preprocess the data to make it suitable for the 1-vs-rest SVM model
        # then train the 10 SVM models using the preprocessed data for each class
        for i in range(self.num_classes):
            # create a binary target variable for class i
            y_binary = np.full(y.shape, fill_value = -1)
            y_binary[y == i] = 1
            
            # train an SVM model for class i
            self.models[i].fit(X, y_binary, **kwargs)

    def predict(self, X,y) -> np.ndarray:
        # pass the data through all the 10 SVM models and return the class with the highest score
        scores = np.zeros((X.shape[0], self.num_classes))
        for i in range(self.num_classes):
            scores[:, i] = self.models[i].predict(X,y)
        
        y_pred = np.argmax(scores, axis=1)
        #print("y_pred = ",y_pred)
        return y_pred

    def accuracy_score(self, X, y) -> float:
        # compute the accuracy of the model (for debugging purposes)
        return np.mean((self.predict(X,y) ) == (y ))
    
    def precision_score(self, X, y) -> float:
        precision_scores = []
        y_pred=self.predict(X,y)
        for i in range(self.num_classes):
            tp = np.sum((y == i) & (y_pred == i))
            fp = np.sum((y != i) & (y_pred == i))
            precision = tp / (tp + fp)
            precision_scores.append(precision)
        
        return np.mean(precision_scores)
        
    
    def recall_score(self, X, y) -> float:
        recall_scores = []
        y_pred=self.predict(X,y)
        for i in range(self.num_classes):
            tp = np.sum((y == i) & (y_pred == i))
            fn = np.sum((y != i) & (y_pred != i))
            r_score = tp / (tp + fn)
            recall_scores.append(r_score)
        
        return np.mean(recall_scores)
    
    def f1_score(self, X, y) -> float:
        return  2* self.precision_score(X,y) * self.recall_score(X,y)/ (self.precision_score(X,y) + self.recall_score(X,y))
