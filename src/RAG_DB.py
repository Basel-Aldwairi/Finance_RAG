from pymongo import MongoClient
from numpy.linalg import norm
import numpy as np

# Helps in comparison
def cosine_similarity(a, b):
    a = np.array(a).flatten()
    b = np.array(b).flatten()
    return np.dot(a, b) / (norm(a) * norm(b))

class RAGDataBase:

    # Database address and name
    def __init__(self,uri = 'mongodb://127.0.0.1:27017', db_name = 'rag_bank_database'):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.questions = self.db['questions']

        # Deletion types
        self.delete_one = 0
        self.delete_many = 1
        self.delete_all = 2

    def insert_question(self,question, answer, embeddings= None):
        doc = {
            'question': question,
            'answer': answer,
            'embeddings': embeddings.tolist() if embeddings is not None else None
        }
        self.questions.insert_one(doc)

    def get_all_questions(self):
        return list(self.questions.find({},{'_id':0}))


    def search_question(self,question_embeddings,threshold = 0.8):
        docs = self.get_all_questions()
        if not docs:
            return None, None

        embeddings = np.array([doc['embeddings'] for doc in docs])

        # Checks similarity between question embedding and the embeddings of all the questions in the Database
        similarities = [cosine_similarity(question_embeddings,e) for e in embeddings]
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        # Gets best match if the similarity is above the threshold
        if best_score >= threshold:
            best_match = docs[best_idx]
            return best_match['question'], best_match['answer']
        return None, None

    def delete(self,embedded_question,deletion_type):

        question, answer = self.search_question(embedded_question)
        if deletion_type == self.delete_all:
            self.questions.delete_many({})
            return True

        if question:
            match deletion_type:
                case self.delete_one:
                    self.questions.delete_one({'question': {'$regex': question, '$options': 'i'}})
                    return True
                case self.delete_many:
                    self.questions.delete_many({'question': {'$regex': question, '$options': 'i'}})
                    return True
        return False