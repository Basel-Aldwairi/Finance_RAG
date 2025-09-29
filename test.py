from pymongo import MongoClient

class TestDB:
    def __init__(self,uri='mongodb://localhost:27017', db_name  = 'test_db'):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.questions = self.db['questions']
        self.delete_one = 0
        self.delete_many = 1
        self.delete_all = 2

    def save_question(self, question, answer, embedding = None):
        doc = {
            'question': question,
            'answer' : answer,
            'embedding' : embedding.tolist() if embedding is not None else None
        }
        self.questions.insert_one(doc)

    def get_all_questions(self):
        return list(self.questions.find({}, {'_id': 0}))

    def search_question(self, question):
        question_list = self.get_all_questions()
        for i in question_list:
            if i['question'] == question:
                return i['answer']
        return None
    def delete_question(self,question,delete_option):
        match delete_option:
            case self.delete_one:
                self.questions.delete_one({'question':{'$regex':question,'$options':'i'}})
            case self.delete_many:
                self.questions.delete_many({'question':{'$regex':question,'$options':'i'}})
            case self.delete_all:
                self.questions.delete_many({})

    def drop_collection(self):
        self.db.drop_collection('questions')

    def drop_database(self):
        self.client.drop_database('test_db')