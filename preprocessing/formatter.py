import json

def format(og_path, new_path):
    unformmated_file = open(og_path, "r")
    biz_dict = {"businesses": []}

    for line in unformmated_file.readlines():
        business = json.loads(line)
        biz_dict["businesses"].append(business)

    unformmated_file.close()

    json_file_str = json.dumps(biz_dict)

    formatted_file = open(new_path, "w")
    formatted_file.write(json_file_str)
    formatted_file.close()

# format("../yelp_dataset/unformatted/business.json", "../yelp_dataset/formatted/business.json")
