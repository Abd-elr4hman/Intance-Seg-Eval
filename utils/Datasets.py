
class COCO():
    def __init__(self, data):
        self.data= data
        self.cocoJSON_images= self.data['images']
        self.cocoAnnotations= self.data['annotations']
        self.cocoCategories= self.data['categories']

    def get_annotaions(self):
        return self.cocoAnnotations

    def get_images(self):
        return self.cocoJSON_images

    def get_categories(self):
        return self.cocoCategories


    def get_images_IDS_filenames(self):
        """Takes coco annotaion images and returns lists of image IDS and filenames"""
        image_IDs=[]
        image_filenames=[]

        for image in self.cocoJSON_images:
            image_IDs.append(image['id'])
            image_filenames.append(image['file_name'])

        return image_IDs, image_filenames

    def retrieve_image_GT(self, image_ID:int):
        """retrieve ground truth instanve annotations for a single image"""
        needed_image_annotations=[]
        for annotation_instance in self.cocoAnnotations:
            if annotation_instance['image_id']==image_ID:
                needed_image_annotations.append(annotation_instance)

        return needed_image_annotations
