import re

# for extracting data and class per line in the file
pattern = '(?P<data_name>.+)\\|\s*(?P<class>\d*)'
regex = re.compile(pattern)

# data set file names
# convention used was C(reate) R(etrieve) U(pdate) D(elete) for variable names
create_task_dataset = 'Create.txt'
delete_task_dataset = 'Delete.txt'
retrieve_task_dataset = 'List.txt'


def get_data_from_file(filename=''):
    try:
        file = open(filename, "r")

        data = []

        for line in file:
            result = re.match(regex, line)

            data_name = result.group('data_name')
            data_name = data_name.strip()

            data.append(data_name)

        return data

    except:
        print('File does not exist.')


def get_data_and_class_from_file(filename=''):
    try:
        file = open(filename, "r")

        data = []

        for line in file:
            result = re.match(regex, line)

            data_name = result.group('data_name')
            data_name = data_name.strip()

            data_class = result.group('class')
            data_class = int(data_class)

            data.append(
                        dict(
                             data_name=data_name,
                             data_class=data_class
                             )
                        )

        return data

    except:
        print('File does not exist.')


def get_training_data():
    create_data = get_data_from_file(create_task_dataset)
    delete_data = get_data_from_file(delete_task_dataset)
    retrieve_data = get_data_from_file(retrieve_task_dataset)

    # append data sets
    training_data = []
    training_data.extend(create_data)
    training_data.extend(delete_data)
    training_data.extend(retrieve_data)

    return training_data
