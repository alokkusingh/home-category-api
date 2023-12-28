import joblib

class ExpenseCategorizer:
    # model = joblib.load("/Users/aloksingh/git/home-category-api/model/expense-categorization-mdl.joblib");
    # count_vect = joblib.load("/Users/aloksingh/git/home-category-api/model/expense-categorization-vct.joblib");

    def __init__(self):
        print("ExpenseCategorizer initialized")
        self.model = joblib.load("/Users/aloksingh/git/home-category-api/model/expense-categorization-LSVC-mdl.joblib");
        self.count_vect = joblib.load("/Users/aloksingh/git/home-category-api/model/expense-categorization-vct.joblib");
        print(self.model)

    def getCategory(self, head):
        print(head)
        return self.model.predict(self.count_vect.transform([head]).toarray())
