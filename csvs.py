import csv

def read_csv(name):
    f = open(name, 'r', encoding='utf-8')
    rdr = list(csv.reader(f))
    rdr.pop(0)
    f.close()
    return rdr

def write_csv(data, name):
    f = open(name, 'w', encoding='utf-8', newline='')
    f.write("prediction\n")
    for i in range(len(data)):
        f.write(str(data[i]))
        f.write('\n')
    f.close()