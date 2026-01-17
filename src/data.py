import json


PATH_TO_JSON_FILE = "../fastener_dataset/annotations/instances_default.json"
PATH_TO_IMAGES_FOLDER = "../fastener_dataset/images"


def extract_data():
    with open(PATH_TO_JSON_FILE, "r") as f:
        data = json.load(f)

    images = data["images"]

    annotations = data["annotations"]

    last_annotation = annotations[-1]

    paths = []
    labels = []
    labels_for_subclasses = []

    for annotation in annotations:
        image_id = annotation["image_id"]

        file_name = ""

        for image in images:
            if str(image["id"]) == str(image_id):
                file_name = image["file_name"]
                break

        paths.append(PATH_TO_IMAGES_FOLDER + "/" + file_name.replace(":", "_"))
        labels.append(annotation["category_id"] - 1)

    return paths, labels
