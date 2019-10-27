import json

def format_to_JSON(keyword, og_path, new_path):
    '''
    Converts file of JS objects into JSON file with a single JS object whose
    key is 'keyword' and whose value is a list of the JS objects.

    'og_path' is the path of the unformatted file and and 'new_path' is the
    desired path of the formatted file.
    '''

    unformmated_file = open(og_path, "r")
    d = {keyword: []}

    for line in unformmated_file.readlines():
        obj = json.loads(line)
        d[keyword].append(obj)

    unformmated_file.close()

    json_file_str = json.dumps(d)

    formatted_file = open(new_path, "w")
    formatted_file.write(json_file_str)
    formatted_file.close()

# format_to_JSON("businesses", "../yelp_dataset/unformatted/business.json", "../yelp_dataset/formatted/business.json")
