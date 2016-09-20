import re

# for extracting data and class per line in the file
pattern = '(?P<name>.+)\\|\s*(?P<class>\d*)'
regex = re.compile(pattern)

# data set file names
# convention used was C(reate) R(etrieve) U(pdate) D(elete) for variable names
create_task_dataset = 'Create.txt'
delete_task_dataset = 'Delete.txt'
retrieve_task_dataset = 'List.txt'
dummy_dataset = "Dummy.txt"


def read_from_file(filename=''):
    try:
        file = open(filename, "r")

        data = []
        classes = []

        for line in file:
            result = re.match(regex, line)

            data_name = result.group('name')
            data_name = data_name.strip()

            data_class = result.group('class')
            data_class = int(data_class)

            data.append(data_name)
            classes.append(data_class)

        file.close()

        return dict(names=data,
                    classes=classes)

    except:
        print('File does not exist.')


def get_data():
    dummy_data_set  = read_from_file(dummy_dataset)
    create_data_set = read_from_file(create_task_dataset)
    delete_data_set = read_from_file(delete_task_dataset)
    retrieve_data_set = read_from_file(retrieve_task_dataset)
    
    # append data sets
    data = []
    data.extend(create_data_set['names'])
    data.extend(delete_data_set['names'])
    data.extend(retrieve_data_set['names'])
    data.extend(dummy_data_set['names'])
    
    # append classes
    classes = []
    classes.extend(create_data_set['classes'])
    classes.extend(delete_data_set['classes'])
    classes.extend(retrieve_data_set['classes'])
    classes.extend(dummy_data_set['classes'])

    return dict(names=data,
                classes=classes)
