from os.path import join


def get_activity_data_info(dir):

    dir_path = dir
    dir = dir_path.split("/")[-1]
    data_filepath = None
    feature_cols = None
    with open(join(dir_path, dir + ".xml"), "r") as f:
        xml = f.read().split("\n")
        for i, line in enumerate(xml):
            if "<feature_columns>" in line:
                feature_cols = xml[i + 1].strip().replace(" ", "").split(",")
            elif "file_path" in line:
                data_filepath = (line.replace("<file_path>", "")
                                     .replace("</file_path>", "")
                                     .strip())
    
    return data_filepath, feature_cols