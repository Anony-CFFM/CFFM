import pandas as pd
import scipy.sparse as sp
import numpy as np
import pickle as pickle
from sklearn.utils import shuffle


def read_ratings():
    data_origin = pd.read_csv('BX-Book-Ratings.csv', encoding='latin-1')
    print('Before compressed:\n', data_origin.info())

    # data_origin = data_origin.dropna(axis=0, how='any', inplace=False)
    # data_origin.to_csv('ratings2.csv', index=False, header=True)

    # data_map = pd.DataFrame(columns=['User-ID', 'ISBN'])
    # for index, row in data_origin.iterrows():
    #     splits = row[0].replace('"','').split(';')
    #     # print([splits[0], splits[1],splits[2]])
    #     if len(splits)==3:
    #         if splits[2] == "0":
    #             print(index)
    #             data_map.loc[index] = [splits[0], splits[1]]
    # data_map.to_csv('ratings.csv', index=False, header=True)


def read_books():
    data_origin = pd.read_csv('BX-Books.csv', encoding='latin-1', error_bad_lines=False)
    print('Before compressed:\n', data_origin.info())

    # data_origin = data_origin.dropna(axis=0, how='any', inplace=False)
    # data_origin.to_csv('books.csv', index=False, header=True)

    # data_map = pd.DataFrame(columns=['ISBN', 'Author','Publisher'])
    # for index, row in data_origin.iterrows():
    #     print(index)
    #     splits = row[0].replace('"','').split(';')
    #     data_map.loc[index] = [splits[0], splits[2],splits[4]]
    # data_map.to_csv('BX-Books3.csv', index=False, header=True)


def read_users():
    data_origin = pd.read_csv('BX-Users.csv', encoding='latin-1', error_bad_lines=False)
    print('Before compressed:\n', data_origin.info())

    # data_origin = data_origin.dropna(axis=0, how='any', inplace=False)
    # data_origin.to_csv('users.csv', index=False, header=True)


    # data_map = pd.DataFrame(columns=['User-ID', 'Location', 'Age'])
    # for index, row in data_origin.iterrows():
    #     print(row[0])
    #     # splits = row[0].replace('"', '').split(';')
    #     # print(index)
    # data_origin.to_csv('BX-Users2.csv', index=False, header=True)

    # data_map = pd.DataFrame(columns=['User-ID', 'Location', 'Age'])
    # with open('BX-Users.csv',mode='r',encoding='latin-1',errors=None) as file:
    #     index = 0
    #     all_index = 1
    #     line = file.readline()
    #     while (line):
    #         if(all_index % 5000 == 0):
    #             data_map.to_csv('BX-Users4.csv', index=False, header=True)
    #         splits = file.readline().replace('"', '').replace('\n', '').split(';')
    #         all_index += 1
    #         if len(splits) == 3:
    #             location = splits[1].split(',')
    #             if len(location) == 3:
    #                 location = location[2].strip()
    #                 if splits[2] != 'NULL':
    #                     print(index)
    #                     data_map.loc[index] = [splits[0],location, splits[2]]
    #                     index += 1


    # data_map.to_csv('BX-Users4.csv', index=False, header=True)
    # data_origin = pd.read_csv('BX-Users2.csv', encoding='latin-1', error_bad_lines=False)
    # print('Before compressed:\n', data_origin.info())
    # for index, row in data_origin.iterrows():
    #     print(row)



def test():
    left = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'], 'B': ['B0', 'B1', 'B2', 'B3']})
    mid = pd.DataFrame({'A': ['A0', 'A1'], 'C': ['C0', 'C1']})
    right = pd.DataFrame({'B': ['B1', 'B2'], 'D': ['D1', 'D2']})
    result = pd.merge(left, mid, on='A')
    print(result)
    result = pd.merge(result, right, on='B')
    print(result)


def merge():
    data_users = pd.read_csv('users.csv', encoding='latin-1', error_bad_lines=False)
    data_ratings = pd.read_csv('ratings.csv', encoding='latin-1', error_bad_lines=False)
    data_books = pd.read_csv('books.csv', encoding='latin-1', error_bad_lines=False)

    result1 = pd.merge(data_ratings, data_users, on='User-ID')
    print('Before compressed:\n', result1.info())
    result2 = pd.merge(result1, data_books, on='ISBN')
    print('Before compressed:\n', result2.info())

    result2.to_csv('book_crossing_all.csv', index=False, header=True)


def data_split():
    data_origin = pd.read_csv('book_crossing_all.csv', encoding='latin-1', error_bad_lines=False)
    print('Before compressed:\n', data_origin.info())

    # a = data_origin[data_origin['User-ID'].duplicated(keep='first')].index
    # print(a)

    # number of user-id : 347648
    # number of ISBN : 240456

    # RangeIndex: 374495 entries, 0 to 374494
    # Data columns (total 6 columns):
    # User-ID      374495 non-null int64
    # ISBN         374495 non-null object
    # Location     374495 non-null object
    # Age          374495 non-null int64
    # Author       374495 non-null object
    # Publisher    374495 non-null object



    data_map = pd.DataFrame(columns=['User-ID', 'ISBN', 'Location', 'Age', 'Author', 'Publisher'])
    i, j, k, l, m, n = 0, 0, 0, 0, 0, 0
    User_ID_map, ISBN_map, Location_map, Age_map, Author_map, Publisher_map = {}, {}, {}, {}, {}, {}

    book_info = dict()
    user_info = dict()
    user_book = sp.dok_matrix((347649, 240457), dtype=np.float32)

    for index, row in data_origin.iterrows():
        print('index:', index)
        if row['User-ID'] not in User_ID_map:
            User_ID_map[row['User-ID']] = i
            User_ID = i
            i += 1
        else:
            User_ID = User_ID_map[row['User-ID']]

        if row['ISBN'] not in ISBN_map:
            ISBN_map[row['ISBN']] = j
            ISBN = j
            j += 1
        else:
            ISBN = ISBN_map[row['ISBN']]

        if row['Location'] not in Location_map:
            Location_map[row['Location']] = k
            Location = k
            k += 1
        else:
            Location = Location_map[row['Location']]

        if row['Age'] not in Age_map:
            Age_map[row['Age']] = l
            Age = l
            l += 1
        else:
            Age = Age_map[row['Age']]

        if row['Author'] not in Author_map:
            Author_map[row['Author']] = m
            Author = m
            m += 1
        else:
            Author = Author_map[row['Author']]

        if row['Publisher'] not in Publisher_map:
            Publisher_map[row['Publisher']] = n
            Publisher = n
            n += 1
        else:
            Publisher = Publisher_map[row['Publisher']]

        data_map.loc[index] = [User_ID, ISBN, Location, Age, Author, Publisher]
        user_book[User_ID, ISBN] = 1.0
        user_info[User_ID] = (Location, Age)
        book_info[ISBN] = (Author, Publisher)

    print('User_ID:', len(User_ID_map))
    print('ISBN:', len(ISBN_map))
    print('Location:', len(Location_map))
    print('Age:', len(Age_map))
    print('Author:', len(Author_map))
    print('Publisher:', len(Publisher_map))
    # User_ID: 26847
    # ISBN: 134039
    # Location: 144
    # Age: 124
    # Author: 56089
    # Publisher: 9093


    pickle.dump(user_book, open('user_book.pkl', 'wb'))
    pickle.dump(user_info, open('user_info.pkl', 'wb'))
    pickle.dump(book_info, open('book_info.pkl', 'wb'))
    data_map.to_csv('all_positive_data.csv', index=False, header=True, sep=' ')


def negative_data():

    print('negative!')

    data_origin = pd.read_csv('all_positive_data.csv', encoding='latin-1', sep=' ')
    print('Before compressed:\n', data_origin.info())

    user_book = pickle.load(open('user_book.pkl', 'rb'))
    user_info = pickle.load(open('user_info.pkl', 'rb'))
    book_info = pickle.load(open('book_info.pkl', 'rb'))

    negative_data = pd.DataFrame(columns=['User-ID', 'ISBN', 'Location', 'Age', 'Author', 'Publisher'])

    num_negatives = 2
    num_users = 26847
    num_items = 134039
    all_index = 0

    for index, row in data_origin.iterrows():
        print('index:', index)
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (row[0], j) in user_book:
                j = np.random.randint(num_items)
            (Location, Age) = user_info[row[0]]
            (Author, Publisher) = book_info[j]

            negative_data.loc[all_index] = [row[0], j, Location, Age, Author, Publisher]
            all_index += 1

    negative_data.to_csv('all_negative_data.csv', index=False, header=True, sep=' ')
    print('Before compressed:\n', negative_data.info())
    # User-ID      748990 non-null object
    # ISBN         748990 non-null object
    # Location     748990 non-null object
    # Age          748990 non-null object
    # Author       748990 non-null object
    # Publisher    748990 non-null object





def train_validation_test():

    # all_positive = pd.read_csv('all_positive_data.csv', encoding='latin-1', sep=' ')
    # all_negative = pd.read_csv('all_negative_data.csv', encoding='latin-1', sep=' ')
    # print('loadover!')
    #
    data_all = pd.DataFrame(columns=['rating','User-ID', 'ISBN', 'Location', 'Age', 'Author', 'Publisher'])
    #
    # all_positive.insert(0, 'rating', [1 for _ in range(all_positive.shape[0])])
    # all_negative.insert(0, 'rating', [-1 for _ in range(all_negative.shape[0])])
    #
    # all = pd.concat([all_positive, all_negative]).reset_index()
    # all = shuffle(all, random_state=2019).reset_index()
    # all.to_csv('book_crossing_without_1.csv', index=False, header=True, sep=' ')
#   _______________________________
#     all = pd.read_csv('book_crossing_without_1.csv', encoding='latin-1', sep=' ')
#     print('Before compressed:\n', all.info())
#     length = all.shape[0]
#     each_scale = int(length/10)
#     for i in range(10):
#         print('finish:',i)
#         if (i != 9):
#             # data_split = all[i*each_scale:(i+1)*each_scale]
#             print(i)
#         else:
#             data_split = all[i*each_scale:]
#             data_split.to_csv('split/book_crossing_without_1.'+ str(i) +'.csv', index=False, header=True, sep=' ')

    flag = 9
    all = pd.read_csv('split/book_crossing_without_1.'+ str(flag) +'.csv', encoding='latin-1', sep=' ')
    #
    # # length = all.shape[0]
    # # train = data_all[:each_scale*7]
    # # validation = data_all[each_scale * 7:each_scale*9]
    # # test = data_all[each_scale * 9:]
    # #
    for i in range(all.shape[0]):
        print('flag:',flag,'\tindex!:',i)
        rating = all.at[i,'rating']
        User_ID = all.at[i, 'User-ID']
        ISBN = all.at[i, 'ISBN']
        Location = all.at[i, 'Location']
        Age = all.at[i, 'Age']
        Author = all.at[i, 'Author']
        Publisher = all.at[i, 'Publisher']

        data_all.loc[i] = [rating,str(User_ID)+':1',str(ISBN)+':1',str(Location)+':1',str(Age)+':1',str(Author)+':1',str(Publisher)+':1']

    data_all.to_csv('split/book_crossing_with_1.' + str(flag) + '.csv', index=False, header=True, sep=' ')

    # print(data_all)
    #
    # length = data_all.shape[0]
    # each_scale = int(length/10)
    # train = data_all[:each_scale*7]
    # validation = data_all[each_scale * 7:each_scale*9]
    # test = data_all[each_scale * 9:]
    #
    # print('writing...')
    #
    # data_all.to_csv('bookcrossing.all.libfm',index=False, header=False, sep=' ')
    # train.to_csv('bookcrossing.train.libfm', index=False, header=False, sep=' ')
    # validation.to_csv('bookcrossing.validation.libfm', index=False, header=False, sep=' ')
    # test.to_csv('bookcrossing.test.libfm', index=False, header=False, sep=' ')
    #
    print('finish!')



def dataset_construction():
    length = 0
    data_all = pd.read_csv('split/book_crossing_with_1.' + str(0) + '.csv', encoding='latin-1', sep=' ')

    for i in range(1,10):
        a = pd.read_csv('split/book_crossing_with_1.'+ str(i) +'.csv', encoding='latin-1', sep=' ')
        data_all = pd.concat([data_all, a])
        # print(a)
        # length += a
        # print(length)
    print('Before compressed:\n', data_all.info())





    # length = data_all.shape[0]
    # each_scale = int(length/10)
    # train = data_all[:each_scale*7]
    # validation = data_all[each_scale * 7:each_scale*9]
    # test = data_all[each_scale * 9:]
    #
    # train.to_csv('train.libfm', index=False, header=False, sep=' ')
    # validation.to_csv('validation.libfm', index=False, header=False, sep=' ')
    # test.to_csv('test.libfm', index=False, header=False, sep=' ')



def final_split():
    data_all = pd.read_csv('split/book_crossing_with_1.' + str(0) + '.csv', encoding='latin-1', sep=' ')
    length = data_all.shape[0]
    each_scale = int(length / 10)
    train = data_all[:each_scale * 7]
    validation = data_all[each_scale * 7:each_scale * 9]
    test = data_all[each_scale * 9:]

    temp_train = train.sample(frac=0.15, replace=False, random_state=2021)
    temp_validation = temp_train.sample(frac=0.98, replace=False, random_state=2021)
    temp_test = temp_train.sample(frac=0.98, replace=False, random_state=2021)

    train = shuffle(pd.concat([train, temp_train]), random_state=2019)
    validation = shuffle(pd.concat([validation, temp_validation]), random_state=2019)
    test = shuffle(pd.concat([test, temp_test]), random_state=2019)

    # print('Before compressed:\n', temp.info())


    train.to_csv('book-crossing.train.libfm', index=False, header=False, sep=' ')
    validation.to_csv('book-crossing.validation.libfm', index=False, header=False, sep=' ')
    test.to_csv('book-crossing.test.libfm', index=False, header=False, sep=' ')


def data_assign():
    import pandas as pd

    df1 = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'B': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    df2 = pd.DataFrame({'A': [11, 12, 13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30], 'B': [11, 12, 13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],
                        'C': [11, 12, 13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]})

    # train_data = pd.DataFrame(columns=['rating','User-ID', 'ISBN', 'Location', 'Age', 'Author', 'Publisher'])

    data_all = pd.DataFrame(columns=['D','A', 'B', 'C'])

    df1.insert(0, 'D', [1 for _ in range(df1.shape[0])])
    df2.insert(0, 'D', [-1 for _ in range(df2.shape[0])])
    df = pd.concat([df1, df2]).reset_index()
    print(df)

    num_columns = df.shape[1]


    for i in range(df.shape[0]):
        A = df.at[i, 'A']
        B = df.at[i, 'B']
        C = df.at[i, 'C']
        D = df.at[i, 'D']
        data_all.loc[i] = [D,str(A)+':1',str(B)+':1',str(C)+':1']
    print(data_all)

    data_all = shuffle(data_all,random_state=2019)

    length = data_all.shape[0]
    each_scale = int(length/10)
    train = data_all[:each_scale*7]
    validation = data_all[each_scale * 7:each_scale*9]
    test = data_all[each_scale * 9:]

    train.to_csv('train.libfm', index=False, header=False, sep=' ')
    validation.to_csv('validation.libfm', index=False, header=False, sep=' ')
    test.to_csv('test.libfm', index=False, header=False, sep=' ')



    # for index, row in df.iterrows():
    #     print('index:', index)


def test2():
    all_positive = pd.read_csv('all_positive_data.csv', encoding='latin-1', sep=' ')
    all_negative = pd.read_csv('all_negative_data.csv',encoding='latin-1', sep=' ')

    print(all_negative.shape[0]+all_positive.shape[0])


    # User_ID: 26847
    # ISBN: 134039
    # Location: 144
    # Age: 124
    # Author: 56089
    # Publisher: 9093

def extra_deal():
    # data_all = pd.read_csv('all_negative_data.csv', encoding='latin-1', sep=' ')
    # print('Before compressed:\n', data_all.info())
    # length = data_all.shape[0]
    # each_scale = int(length/7)
    #
    # for i in range(7):
    #     print('finish:',i)
    #     if (i != 6):
    #         data_split = data_all[i*each_scale:(i+1)*each_scale]
    #     else:
    #         data_split = data_all[i*each_scale:]
    #     data_split.to_csv('splits/negative_data_without_1.'+ str(i) +'.csv', index=False, header=True, sep=' ')

    name='negative_data'
    flag = 6
    all = pd.read_csv('splits/'+name+'_without_1.'+ str(flag) +'.csv', encoding='latin-1', sep=' ')


    data = pd.DataFrame(columns=['rating', 'User-ID', 'ISBN', 'Location', 'Age', 'Author', 'Publisher'])

    ISBN_start = 26847
    Location_start = 160886
    Age_start = 161030
    Author_start = 161154
    Publisher_start = 217243

    for i in range(all.shape[0]):
        print('name:',name,'\t','flag:',flag,'\tindex!:',i)
        User_ID = all.at[i, 'User-ID']
        ISBN = all.at[i, 'ISBN']
        Location = all.at[i, 'Location']
        Age = all.at[i, 'Age']
        Author = all.at[i, 'Author']
        Publisher = all.at[i, 'Publisher']

        data.loc[i] = [1,str(User_ID)+':1',str(ISBN+ISBN_start)+':1',str(Location+Location_start)+':1',str(Age+Age_start)+':1',str(Author+Author_start)+':1',str(Publisher+Publisher_start)+':1']

    print('Before compressed:\n', data.info())
    print(name,'\t',str(flag))
    data.to_csv('splits/'+name+'_with_1.' + str(flag) + '.csv', index=False, header=True, sep=' ')

def merge_data():
    name = 'positive_data'
    all = pd.read_csv('splits/'+name+'_with_1.0'+ '.csv', encoding='latin-1', sep=' ')
    for i in range(1,5):
        print('p:',i)
        temp = pd.read_csv('splits/'+name+'_with_1.' + str(i) + '.csv', encoding='latin-1', sep=' ')
        all = pd.concat([all,temp])
    print('positive compressed:\n', all.info())
    all.to_csv('splits/new_all_'+name+'.csv', index=False, header=True, sep=' ')
    print('finish positive!')

    name_ = 'negative_data'
    all = pd.read_csv('splits/' + name_ + '_with_1.0' + '.csv', encoding='latin-1', sep=' ')
    for i in range(1, 7):
        print('n:',i)
        temp = pd.read_csv('splits/' + name_ + '_with_1.' + str(i) + '.csv', encoding='latin-1', sep=' ')
        all = pd.concat([all, temp])
    print('negative compressed:\n', all.info())
    all.to_csv('splits/new_all_' + name_ + '.csv', index=False, header=True, sep=' ')
    print('finish negative!')




def assign_data():
    positive_data = pd.read_csv('splits/new_all_positive_data.csv', encoding='latin-1', sep=' ')
    negative_data = pd.read_csv('splits/new_all_negative_data.csv', encoding='latin-1', sep=' ')

    print('load finish!')
    positive_data = shuffle(positive_data, random_state=2019)
    negative_data = shuffle(negative_data, random_state=2019)

    length_positive = positive_data.shape[0]
    each_scale_positive = int(length_positive/10)
    train_positive = positive_data[:each_scale_positive*7]
    validation_positive = positive_data[each_scale_positive * 7:each_scale_positive*9]
    test_positive = positive_data[each_scale_positive * 9:]

    length_negative = negative_data.shape[0]
    each_scale_negative = int(length_negative/10)
    train_negative = negative_data[:each_scale_negative*7]
    validation_negative = negative_data[each_scale_negative * 7:each_scale_negative*9]
    test_negative = negative_data[each_scale_negative * 9:]

    train_data = shuffle(pd.concat([train_positive,train_negative]), random_state=2020)
    validation_data = shuffle(pd.concat([validation_positive, validation_negative]), random_state=2020)
    test_data = shuffle(pd.concat([test_positive, test_negative]), random_state=2020)


    print('train_data\n',train_data.info())
    print('validation_data\n', validation_data.info())
    print('test_data\n', test_data.info())

    train_data.to_csv('book-crossing.train.libfm', index=False, header=True, sep=' ')
    validation_data.to_csv('book-crossing.validation.libfm', index=False, header=True, sep=' ')
    test_data.to_csv('book-crossing.test.libfm', index=False, header=True, sep=' ')
    print('finish!')


def modify_column():
    negative_data = pd.read_csv('splits/new_all_negative_data.csv', encoding='latin-1', sep=' ')
    negative_data['rating'] = negative_data['rating'].apply(lambda x: -1)
    negative_data.to_csv('splits/new_all_negative_data.csv', index=False, header=True, sep=' ')
    print(negative_data)




if __name__ == '__main__':
    assign_data()
    # modify_column()
    # assign_data()
    # merge_data()
    # extra_deal()
    # User_ID: 26847
    # ISBN: 134039
    # Location: 144
    # Age: 124
    # Author: 56089
    # Publisher: 9093



    # test2()
    # dataset_construction()
    # train_validation_test()
    # negative_data()
    # test()
    # read_ratings()
    # read_books()
    # read_users()
    # merge()
    # data_split()
