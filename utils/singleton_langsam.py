from lang_sam import LangSAM

class LangSAMSingleton:
    _instance = None
    _model = None

    @staticmethod
    def get_instance():
        if LangSAMSingleton._instance is None:
            LangSAMSingleton()
        return LangSAMSingleton._instance

    def __init__(self):
        if LangSAMSingleton._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            LangSAMSingleton._instance = self
            self._model = self.load_model()  # Your model loading function
            print("Model loaded successfully!")

    def load_model(self):
        # Your model loading logic here (e.g., using PyTorch, TensorFlow, etc.)
        print("Loading model...")
        model = LangSAM()  # Replace with actual model loading code
        return model

    def get_model(self):
        return self._model