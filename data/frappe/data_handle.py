import pandas as pd



if __name__ == '__main__':
    data_origin = pd.read_csv('frappe.train.libfm',sep=' ',names=['R','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10'])
    print('Before compressed:\n', data_origin.info())
    posi_ = data_origin[data_origin.R == 1]
    nega_ = data_origin[data_origin.R == -1]
    length = posi_.shape[0]
    print(length)
    nega_ = nega_.sample(length,random_state=2019)
    print(nega_)
    # for index, row in nega_.iterrows():
    #     print(row[0])
        # print(index)