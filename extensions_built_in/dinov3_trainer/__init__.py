from toolkit.extension import Extension


class DINOv3TrainerExtension(Extension):
    # uid must be unique, it is how the extension is identified
    uid = "dinov3_trainer"

    # name is the name of the extension for printing
    name = "DINOv3 Segmentation Trainer"

    # This is where your process class is loaded
    # keep your imports in here so they don't slow down the rest of the program
    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from jobs.process.DINOv3TrainProcess import DINOv3TrainProcess
        return DINOv3TrainProcess


AI_TOOLKIT_EXTENSIONS = [
    DINOv3TrainerExtension
]