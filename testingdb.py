from test import TestDB

db = TestDB()

db.save_question('What is AI?','AI is something')

# db.delete_question('AI',delete_option=db.delete_all)
print(len(db.get_all_questions()))
ldb = db.get_all_questions()

for i,d in enumerate(ldb):
    print(f'{i} : {d}')

db.drop_collection()
db.drop_database()
print(db.get_all_questions())