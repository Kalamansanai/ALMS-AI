import json

class Config:
    
    def __init__(self):
        f = open('config.json')
        data = json.load(f)

        self.dataset_path = data['dataset_path']

        self.image_width = data['image_width']
        self.image_height = data['image_height']
        self.item_number = data['item_number']

        self.resize_width = data['resize_width']
        self.resize_height = data['resize_height']
        
        self.epochs = data['epochs']
        self.batch_size = data['batch_size']
        self.loss = data['loss']
        self.metrics = data['metrics']
        self.validation_split = data['validation_split']
        self.shuffle = data['shuffle']
        self.learning_rate = data['learning_rate']

        f.close()   
        print("Configurations loaded")
        