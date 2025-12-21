import json


PATH_TO_JSON_FILE = "../fastener_dataset/annotations/instances_default.json"
PATH_TO_IMAGES_FOLDER = "../fastener_dataset/images"

def extract_data():
    with open(PATH_TO_JSON_FILE, "r") as f:
        data = json.load(f)

    images = data["images"]
    print("Number of images: ",len(images))
    print("Attributes: ",list(images[0].keys()))
    print()

    annotations = data["annotations"]
    print("Number of annotations (they refer to images):",len(annotations))
    print("Attributes: ", list(annotations[0].keys()))
    print()
    last_annotation = annotations[-1]
    print("IDs don't always match!")
    print(last_annotation["id"],"!=",last_annotation["image_id"])
    paths = []
    labels = []
    labels_for_subclasses = []

    for annotation in annotations:
        image_id = annotation["image_id"]

        file_name = ""

        #sram me ove petlje, vjerovatno se moze nekako aproksimirat uz pomoc image_id-a
        for image in images:
            if str(image["id"]) == str(image_id):
                file_name = image["file_name"]
                break

        #u jsonu se nalaze krivi nazivi slika pa je potrebno ispraviti
        paths.append(PATH_TO_IMAGES_FOLDER + "/" + file_name.replace(":","_"))
        labels.append(annotation["category_id"]-1)
        #labels_for_subclasses.append(annotation["category"])

    print("Size of paths and labels should be the same: ", len(paths), len(labels), len(labels_for_subclasses))
    
    return paths, labels